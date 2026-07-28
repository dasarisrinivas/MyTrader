# Replay Engine v3 — Multi-Leg Design Audit

**Scope:** build the measurement instrument only. No strategy research.
**Unfreeze:** multi-leg replay ONLY (the single sanctioned change).
**Unchanged:** production, engine v2, promotion constitution v2.1, all strategies.

---

## 0. Feasibility — VERIFIED before design (not assumed)

| Question | Evidence | Verdict |
|---|---|---|
| Is the 2nd leg of historical spreads recoverable? | `suggested_trade` contains e.g. `"Buy 693C / Sell 698C exp APR26"`, `"Buy 660P / Sell 655P"`, `"Long Straddle: Buy 685C + Buy 685P exp APR26"` | **YES — parseable** |
| Cross-expiry quotes (calendars)? | exp 20260727/20260731/20260821 all HTTP 200 with correct term structure (740C = 5.23 / 9.01 / 15.69) | **YES** |
| v3 greeks endpoints? | `/v3/option/history/{greeks,all_greeks,implied_volatility}` → **404** | **NO — self-compute BS** (already have `flow_research/greeks.py`) |
| Leg synchronization risk? | NBBO is a **uniform 1-second grid** (23,402 rows/session ≈ 6.5h×3600) identically for every contract | **LOW — both legs quote on the same 1s ticks** |
| Endpoint respects params? | 700C=43.37 / 740C=5.23 / 780C=0.01 — correct moneyness; term structure correct | **YES — v2 results stand** |

**⚠ Verified data defect:** for `LONG_STRADDLE`, the `strike` column (680.0) does
**NOT** match the actual legs (`685C + 685P`). The strike column is an ATM/VWAP
reference, not a leg. **The parser MUST read `suggested_trade`, never the strike
column.** (This is precisely how a mis-graded family would silently persist.)

---

## 1. Supported structures

| Structure | Legs | Support | Notes |
|---|---|---|---|
| Vertical debit (bull call, bear put) | 2 | **YES — v3.0** | All existing spread signals are debit |
| Vertical credit (bear call, bull put) | 2 | **YES — v3.0** | No historical signals exist; synthetic construction (§6) |
| Straddle / strangle | 2 long | **YES — v3.0** | Parser must handle `+` form |
| Calendar / diagonal | 2, diff expiry | **YES — v3.1** | Cross-expiry verified; adds expiry-pair cache keys |
| Iron condor | 4 | **OPTIONAL — v3.2** | Composition of two verticals; defer |

---

## 2. Replay accuracy — pricing model

**Convention (conservative, matches v2):**
- **Entry:** buy leg at **ASK**, sell leg at **BID** → worst-case net debit / least credit.
- **Exit:** sell long at **BID**, buy short at **ASK**.
- **Commission:** $0.65 per contract per leg per side → vertical round trip = **$2.60**.
- Real spread orders often fill *better* than legging each side worst-case, so this
  convention is **deliberately conservative — it biases AGAINST promotion**, the
  correct direction for a promotion instrument.

**Both legs, bid/ask evolution:** each leg priced from its own NBBO series at the
**same 1-second timestamp**; if either leg lacks a valid NBBO at that instant the
observation is **unreplayable and itemized** (constitution v2.1 G8 rules apply
unchanged).

**Max loss / max profit (and they double as invariants):**
- Debit vertical: `max_loss = debit`, `max_profit = width − debit`
- Credit vertical: `max_profit = credit`, `max_loss = width − credit`
- Straddle: `max_loss = total debit`, `max_profit` unbounded

**Expiry settlement:** both legs settled at **intrinsic** vs the SPY close on the
expiry date (ThetaData stock EOD), never a stale quote.

**Greeks:** self-computed Black-Scholes per leg (`greeks_src="COMPUTED"`);
structure greeks = signed sum of leg greeks. Vendor greeks unavailable (404).

---

## 3. Architecture

```
shree/research/replay_engine_v3.py          # NEW FILE — v2 untouched
    Leg(action, right, strike, expiry, qty)
    Structure(kind, legs, width, entry_dt, exit_dt)
    parse_structure(suggested_trade, expiry_date) -> Structure | None
    replay_multileg(cache, structure, ex_model) -> MultiLegResult
    MultiLegResult(entry_debit, exit_credit, net_dollar, ret_pct,
                   max_loss, max_profit, per_leg[], unreplayable_reason)
```
- **Imports** `QuoteCache`, `_prevailing`, `ExecutionModel` from v2 — **does not
  modify them**. v2 stays byte-identical, so every single-leg scorecard is
  reproducible bit-for-bit (requirement 4).
- v3 is additive: `RESEARCH_ENGINE_V3_VERSION = "v3.0-multi-leg"`, reported in
  provenance alongside the v2 hash.

---

## 4. Preservation of single-leg results
- `replay_engine.py` is **not edited**. Its file hash stays `4d3239df40707c2a`.
- Existing scorecards, `data/promotion_scorecard.json`, and all prior verdicts
  remain valid and reproducible.
- The promotion constitution is **not** amended by this work.

---

## 5. Eligibility — unchanged until validation passes
**G10 stays in force: multi-leg families remain INELIGIBLE.** Only after the full
test plan (§7) passes may a constitution amendment (v2.2) narrow G10 — and that
amendment requires the standard version bump + migration notes + historical
re-comparison.

---

## 6. HONEST SCOPE CAVEAT — what v3 does and does not unlock
- v3 does **NOT** make existing spread families promotable. They have **7
  sessions** vs the required **≥30**, and are missing the `high_vix` regime — they
  fail COVERAGE regardless of how accurately they are priced.
- All existing spread signals are **debit** (long premium). **Zero credit-spread
  signals exist historically.** Therefore the short-premium research direction
  requires **synthetic construction**: build hypothetical credit spreads at
  historical signal timestamps and replay them with the instrument.
- **Therefore v3's justification is RESEARCH CAPABILITY, not near-term promotion.**
  Under the rule "a framework change must change a promotion decision or it is
  rejected," v3 is admissible only because it is the sanctioned unfreeze enabling
  the #1 backlog direction — not because it promotes anything today.

---

## 7. Test plan — must pass before any research use

| # | Test | Pass criterion |
|---|---|---|
| V1 | **Invariant bounds** | every replayed P&L within `[−max_loss, +max_profit]`; zero violations |
| V2 | **Leg-sum identity** | multi-leg net == (long-leg net) − (short-leg net) computed independently via v2, to the cent |
| V3 | **Cross-vendor** | entry debit vs IB-recorded `entry_mid` on spread signals; median deviation < 5% |
| V4 | **Parser correctness** | 30 hand-checked `suggested_trade` strings; 100% exact legs; **MUST flag the straddle strike-column mismatch** |
| V5 | **Expiry settlement** | intrinsic settlement matches manual calc on known expiries |
| V6 | **Determinism** | identical inputs → identical outputs; stable dataset hash across runs |
| V7 | **Unreplayable itemization** | every failure itemized with reason (G8 v2.1 compliance) |

Any V1/V2 failure = engine defect, blocks all use.

---

## 8. Scope estimate
| Item | Effort |
|---|---|
| Structure/Leg model + `suggested_trade` parser | 0.5 d |
| v3 replay core (2-leg, entry/exit/expiry settlement) | 1.0 d |
| Validation suite V1–V7 | 1.0 d |
| Calendar support (v3.1) | 0.5 d |
| Run + report | 0.5 d |
| **Total** | **≈ 3.5 days** |

---

## 9. Risks
| Risk | Severity | Mitigation |
|---|---|---|
| **Parser fragility** (free-text `suggested_trade`) | HIGH | strict regex, fail-closed, unparseable itemized not guessed; V4 |
| **Straddle strike-column mismatch** (VERIFIED real) | HIGH | parse text only, never the strike column; V4 asserts it |
| **Early assignment** on short leg (American SPY) | MED | **NOT MODELED — documented limitation.** Rare/non-adverse for defined-risk debit spreads; must be stated in every v3 report |
| Leg desynchronization | LOW | uniform 1-second NBBO grid; both legs keyed to same tick |
| Optimistic fills | LOW | worst-side convention biases against promotion |
| Small spread sample (7 sessions) | HIGH for promotion | acknowledged in §6 — v3 is a research instrument, not a promotion unlock |
| Scope creep into strategy research | MED | this document forbids it; instrument only |

---

## 10. Recommendation
**BUILD v3.0** (verticals + straddle/strangle), defer calendars to v3.1 and iron
condors to v3.2. Gate all research behind V1–V7. Keep G10 in force until they
pass. Expect v3 to change **zero** promotion decisions in the near term — its
value is enabling the short-premium research direction, which is the only
untested direction where the measured, monotonic force (theta) works *for* the
position rather than against it.
