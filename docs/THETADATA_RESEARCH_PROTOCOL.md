# ThetaData — One-Month Falsification Protocol (decision memo)

**Status:** RESEARCH DECISION. No strategy. No trading. No production change.
TC / confidence / governor / allocation / risk / exits / production behavior —
**all frozen, untouched.**

**Question this memo answers:** is ONE month of ThetaData justified?

**Stance:** try to DISPROVE that options flow is useful. Assume no edge.
Buy the test only because the test is cheap and decisive — not because we
expect a pass. Pre-commit to stopping on FAIL.

**Date:** 2026-07-23

---

## 1. Executive summary

We already proved (free, via IB) that the flow-research pipeline works end to
end on real tape: ingest → NBBO merge → Lee-Ready aggressor → BS greeks →
measurements, classification spot-checked correct. We also proved IB **cannot**
supply the depth the test needs: 1000-tick cap, second resolution, paced,
consolidated (no venue) → clean coverage collapsed to ~40s of one session, and
sweeps are unreconstructable.

The bottleneck is now data, not code. The decision is whether to pay ThetaData
~$80–160 for one month to run the SAME validation battery we used to kill TC,
retroactively, on our existing 657 signals.

Key economic fact: a one-month subscription grants **historical** access to the
entire Apr–Jul signal window (ThetaData history spans years; you are not limited
to the paid month). So one month buys an **immediate, permanent yes/no** on the
whole options-flow direction. Asymmetric payoff: trivial cost, decisive answer.
A FAIL ends months of temptation; a PASS gates only a shadow observation layer.

**Recommendation (detailed in §10): BUY ONE MONTH — as a falsification test,
pre-registered PASS/FAIL, no renewal without a PASS, accept NO.**

---

## 2. What ThetaData uniquely provides (Task 1)

Fields/capabilities beyond what IB gives us, and why each matters to the test.
(Pricing/tier approximate — verify at purchase. Intraday historical trades+quotes
are the higher options tier.)

| Capability | IB reality (measured) | ThetaData | Why it matters |
|---|---|---|---|
| **True OPRA ms timestamps** | second resolution | millisecond | Aggressor + lead/lag need sub-second ordering of trade vs quote. Second-resolution smears many prints into one instant → classification noise. |
| **Historical NBBO (quote) depth** | ~40s clean coverage/pull, paced | full session, ms, multi-year | Lee-Ready aggressor requires the prevailing NBBO at every trade. Without deep quote history there is no honest aggressor label at scale. **This is the single blocking gap.** |
| **Full historical trade tape** | 1000-tick cap, paced | complete session, bulk | Need every print across the whole session, all 657 signal windows — not a 40s tail. |
| **Exchange / venue per print** | consolidated last | per-venue OPRA code | Real sweep detection = same contract+side crossing multiple venues in a tight window. IB's consolidated tape makes this impossible. |
| **Condition codes** | limited | OPRA condition flags | Filter spread legs / auctions / late prints out of net measures — the #1 way flow products lie. |
| **Sweep / aggressor reconstruction** | not reliable | reconstructable from venue+cond+NBBO | The "informed urgency" proxy. Untestable on IB. |
| **Historical greeks + IV** | not served historically | delta/gamma/theta/vega + IV time series | Delta-weighted flow and IV-weighted side need greeks AT the print instant. We self-computed BS on IB as a fallback; vendor greeks are a cleaner cross-check. |
| **Historical open interest** | awkward | daily OI series | Open-vs-close estimate (new risk vs unwind) needs OI deltas. |
| **Historical options chain** | per-contract, heavy | full chain snapshots | Reconstruct the whole chain at a signal instant (walls, concentration) without thousands of per-contract IB calls. |
| **Multi-session bulk replay** | impractical (pacing) | designed for it | The battery needs 40+ sessions retro-attached. IB would take days and trip pacing; ThetaData bulk-downloads it. |

Bottom line: ThetaData supplies exactly the two things IB cannot — **deep
historical NBBO** (→ honest aggressor at scale) and **venue+condition detail**
(→ real sweeps and clean filtering) — plus greeks/IV/OI/chain history for free
alongside. Without these, the test cannot be run at all; that is the whole
justification for spending.

---

## 3. Research architecture (Tasks 2 & 3)

Bounded, one month, observation-only. Nothing touches production.

Pipeline already built (`shree/flow_research/`, 18 tests passing). New work is
only the ThetaData adapter + bulk downloader + OOS/correction additions to the
validator.

```
ThetaData terminal (local REST/WS)
      │  bulk historical: trades + NBBO quotes + greeks + OI, SPY 0-2DTE,
      │  Apr 1 – Jul 23 (the exact 657-signal window)
      ▼
spy_flow_prints   (raw tape, own DB — never production)
      │  classify (Lee-Ready on real NBBO) + venue-sweep + block + condition filter
      ▼
shadow_flow       (measurements attached to context)
      │  retro-attach to EVERY existing record:
      │    • 657 dispatched signals   (approved + the outcomes already stored)
      │    • all rejected / blocked   (blocked_signals.jsonl — the control)
      │    • all production fills      (real P&L)
      │    • fixed 13x/session intervals (unconditional, unbiased sample)
      ▼
validate.py  →  battery + statistical safeguards  →  PASS / FAIL
```

Reuse (Task 3): we do NOT gather new signals. We attach flow to the data we
already own — 657 signals with outcomes, the blocked shadow book, production
fills. That is why one month is enough: the outcomes already exist; we are only
adding the missing input column and re-running math.

---

## 4. Validation methodology (Task 4)

The SAME battery that killed TC, per flow feature, at horizons 1/5/15/30 min
and to-exit:

- **Forward return** — underlying return by feature quintile; monotone lift or nothing.
- **Option P&L proxy** — executor-real fill model (spreads/slippage baked in), top vs bottom quintile.
- **MFE / MAE** — favorable/adverse excursion asymmetry vs null.
- **Barrier tests** — P(+0.5% before −0.5%), top vs bottom quintile.
- **Event studies** — align on high-|feature| prints; average forward path vs baseline.
- **Information lift** — added-R² / partial correlation of feature AFTER residualizing on the existing price features (RSI, VWAP, ORB, ADX, P/C, VIX, confidence). New variance only.
- **Correlation** — Spearman of each flow feature vs each existing feature. HIGH corr = restatement, not information → discard.
- **Regime dependence** — stability across VIX buckets and across months. Works-in-one-month = overfit.
- **Lead/lag** — does flow LEAD price (predictive) or LAG it (confirmation/chase)? Cross-correlation at ±k minutes. A lagging feature is the TC failure mode again.
- **Statistical significance** — permutation/shuffle p-values, Wilson bounds on any win-rate, effect sizes (not just p).

Features under test (measurements, not signals): pc_prem_imbalance, dw_flow,
net_call/put_prem, sweep_intensity, block_prem, oc_open_ratio,
expiry_concentration, strike_repetition, atm_vs_wing, iv_weighted_side.

Primary pre-registered hypothesis (declared before code, to bound data-snooping):
> **H1: delta-weighted aggressor flow (`dw_flow`) in the 30-min window before a
> signal has positive information lift on that signal's forward option P&L,
> after residualizing on existing price features.**
All other features are secondary/exploratory and carry a heavier multiple-
comparisons penalty.

---

## 5. Statistical safeguards (Task 5) — assume every positive is false

| Attack | Defense |
|---|---|
| **Multiple comparisons** | ~11 features × 5 horizons × several tests = dozens of p-values. Control FDR with Benjamini-Hochberg; the PRIMARY H1 must clear Bonferroni alone. A feature that only "passes" as one of 50 tests is noise. |
| **Look-ahead bias** | Snapshot uses only prints with `ts ≤ context ts`. Greeks/IV from data available at that instant. No exit-time information in any pre-signal feature. |
| **Survivorship bias** | Include rejected + all signals + unconditional intervals. The interval sample has zero signal-selection. If "edge" exists only in the selected signal set, it fails. |
| **Overfitting / data snooping** | Pre-registered H1 (§4). Limited degrees of freedom: fixed windows, fixed quintiles, no per-feature tuning. Shuffle control must flatten any claimed effect. |
| **Small sample** | 657 signals → ~130/quintile. Underpowered for subtle effects — a deliberate FEATURE: if the effect needs >657 obs to see, it is too small to trade. Interval sample (~1500+) backstops power for the unbiased test. |
| **Regime dependence** | Time-split OUT-OF-SAMPLE: train Apr–May, test Jun–Jul (purged/embargoed to kill window overlap leakage). A feature found in-sample must survive on the held-out later period. Also require stability across ≥2 VIX regimes. |
| **Lead/lag confound (beta not alpha)** | Residualize on the contemporaneous underlying move (the CALL_SWEEP trap). Lead/lag test must show flow LEADS, not lags. |

If a result cannot survive ALL of these, it is recorded as FAIL. No exceptions,
no "but it's close."

---

## 6. PASS / FAIL — pre-registered BEFORE any code (Task 6)

**PASS only if ALL six hold:**

1. ✓ **Statistically significant incremental information** — primary H1 clears
   Bonferroni (α=0.05 across the primary family); permutation p < 0.05.
2. ✓ **Survives out-of-sample** — effect present on the Jun–Jul holdout with the
   same sign, magnitude within ~½ of in-sample, purged/embargoed.
3. ✓ **Not already explained by current indicators** — information lift > 0 after
   residualizing on all existing features; max |corr| with any existing feature < ~0.5.
4. ✓ **Improves prediction on the 657 shadow signals** — top-vs-bottom quintile
   forward-outcome spread is significant on the real signal set, not only intervals.
5. ✓ **Materially separates winners from losers** — barrier-win top quintile > 50%
   AND meaningfully above bottom quintile; Wilson-lower-bounded.
6. ✓ **Economically meaningful after costs** — quintile P&L spread exceeds realistic
   0DTE spread+slippage (executor-real fill model), not just gross.

**Any one fails → FAIL. Ties → FAIL.**

---

## 7. If FAIL (Task 7)

- Stop options-flow research immediately.
- Do not renew ThetaData.
- Do not invent a new flow hypothesis to rescue it.
- Record the negative result in the audit docs and move on.
- A clean NO is a WIN — it ends months of temptation for ~$100.

---

## 8. If PASS (Task 8)

- Recommend ONLY an observation layer. No strategy, no trading rule.
- Flow runs in **shadow mode** exactly like the blocked-signal book: logged next
  to every signal, influencing nothing.
- It must prove itself forward, live, for a pre-set window (≥20 sessions,
  governor-style evidence tier) BEFORE any proposal to let it touch a production
  decision. That proposal would be a separate memo, separately gated.

---

## 9. Estimated implementation effort

Pipeline, schema, features, validator, tests already exist and are proven on real
IB data. Remaining, one-time:

| Item | Effort |
|---|---|
| ThetaData adapter (`ThetaDataSource.prints()` — REST/bulk → Print) | ~1 day |
| Bulk historical downloader (SPY 0–2DTE trades+quotes+greeks+OI, Apr–Jul) | ~1 day |
| OOS split + BH/Bonferroni + lead/lag additions to `validate.py` | ~1 day |
| Timezone/alignment hardening (signal `sent_at` ↔ ET print clock — a real trap hit on IB) | ~0.5 day |
| Run battery + write result memo | ~1–2 days |
| **Total** | **~4–6 working days** |

---

## 10. Estimated research duration

Comfortably inside the one paid month:

- Days 1–3: subscribe, stand up terminal, bulk-download the Apr–Jul window, ingest.
- Days 4–6: retro-attach snapshots, run battery + safeguards.
- Days 7–10: adversarial destruction pass, OOS, write PASS/FAIL memo.

~2 weeks of the month used; the rest is buffer. **Decision delivered within the
single billing cycle** — that is the core reason one month suffices.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Apr–Jul tick history missing/gappy at ThetaData | Spot-check coverage on day 1 for a few known-active sessions BEFORE the deep pull; if gappy, abort within the free-look. |
| 657 signals too few for subtle effects | Interval sample (~1500+) for power; and "too small to see = too small to trade" is an acceptable FAIL. |
| Timezone/alignment error fabricates or destroys overlap | Dedicated hardening step (§9); the IB run already exposed `sent_at` TZ ambiguity — fix before trusting any join. |
| Multiple-comparisons false positive | BH/FDR + Bonferroni on primary; shuffle control; OOS holdout. |
| Terminal/bulk operational overhead eats the month | Effort budgeted; downloader is scriptable and restartable. |
| Scope creep into "let's just try a strategy" | Hard rule: observation only, PASS gates a shadow layer, nothing else. |
| Sunk-cost pressure after paying | Pre-registered FAIL criteria + pre-commitment to stop. The whole memo exists to make stopping the default. |

---

## Free-tier pre-flight (verified 2026-07-23, $0, before subscribing)

Confirmed against the running terminal on the FREE tier, so the paid month is
pure execution:

- **API shape:** v3 REST on :25503, CSV. Params `symbol/expiration=YYYYMMDD/
  strike/right/start_date/end_date`. Adapter built + 25 tests green.
- **Paywall location:** `trade`/`trade_quote` = Standard (403 on free); `quote`/
  `ohlc`/`open_interest` = Value; discovery + EOD + stock-close = free. → **buy
  Standard**, Value cannot pull trades.
- **Venue + condition codes present** (`bid_exchange`/`ask_exchange`/
  `*_condition`) → real sweeps + spread-leg filtering (IB could not).
- **Coverage:** every sampled Apr 1–Jul 22 session has 157–238 strikes — window
  fully populated.
- **Timezone = ET** (max free-EOD `last_trade` hour = 16, not 20).
- **Bulk = omit strike/right** → whole chain per call (~240 vs ~5000 requests).
- **Python 3.9 vs library:** the official `thetadata` pip lib needs Py 3.12; the
  pipeline runs on 3.9, so we use the REST adapter for this run. Migrate to the
  library only if a future forward-capture layer is built.

## Final verdict

**BUY THETADATA FOR ONE-MONTH RESEARCH.**

Not because we expect edge — we expect FAIL. Buy it because it is the only source
that can run the honest test, the test is decisive within one billing cycle, and
the payoff is asymmetric: ~$100 buys a permanent yes/no on a research direction
that has otherwise cost months.

Binding conditions:
- Pre-registered PASS/FAIL (§6) locked before the adapter is written.
- One month only. No renewal without a PASS.
- FAIL → stop the entire options-flow direction, no new hypothesis.
- PASS → shadow observation layer only, separately gated. Nothing touches TC,
  confidence, governor, allocation, risk, exits, or production behavior.
