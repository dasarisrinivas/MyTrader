# Counterfactual blocked-trade analysis (2026-05-26)

## Question

For the trades the strategy's gates **reject**, what would have happened if they'd
been allowed? Which gates earn their keep (block net-losers) and which might be
throwing away profit (a frequency opportunity)?

## Method

The live decision log (`manager_decisions.jsonl`) turned out to be unusable for this:
it's an **advisory shadow log** (`override=True` on 98.5% of records, zero real
approvals, each signal logged ~4.6× across many backtest runs) — the backtest's
actual trades go through `RiskManager`, not that manager. So I used the faithful
method instead: **disable each gate, re-run the 26-month live-faithful book, and the
PnL/trade delta is exactly what that gate's blocked trades would have done** (path-
dependent, like the live engine). Baseline = current live config (shorts-off, ADX-12,
all blocks on): **392 trades, +$2,142** over Mar 2024 – May 2026.

## Per-gate counterfactual (gate OFF − baseline, 26 months, MES $5/pt)

| Gate disabled | Δ trades | Δ PnL | $/blocked trade | Verdict |
|---|---|---|---|---|
| Monday block | +118 | −$337 | −$2.9 | Value-adding, but regime-dependent (see below) |
| Late-afternoon block (≥20:00 UTC) | +13 | −$295 | −$22.7 | **Strongly value-adding** — blocks few, very bad trades |
| Medium-ATR gate | +32 | −$63 | −$2.0 | ~Neutral (mildly value-adding) |
| Exhaustion cooldown | ~0 | ~$0 | — | Inert — rarely binds |
| **All four off** | **+175** | **−$623** | **−$3.6** | Collectively protect ~$623 |
| *(HTF 30m filter)* | — | — | — | Inert in backtest (needs live 30m feed) |

(Previously established by the same toggle method: the **opening block** and **Friday
trend-cont block** are both value-adding; **disabling shorts** added +$562; **ADX 18→12**
was the one *relaxation* that added accretive trades.)

## Findings

1. **No gate is value-destroying.** Every block either blocks net-losing trades or is
   inert. There is **no free frequency** to be had by relaxing a gate — the blocked
   trades are, in aggregate, net-negative (−$3.6/trade across 175 of them). The gate
   stack is well-calibrated to the 26-month data.
2. **Highest value per blocked trade: the late-afternoon block** (−$22.7/trade). It
   blocks only ~13 trades over two years but they're sharply negative. Cheapest, most
   surgical gate — definitely keep.
3. **The Monday block is the biggest frequency suppressor** (+118 trades / +30% if
   removed) and is net value-adding (−$337 to remove), **but its edge is reversing**:

   | Monday block removed | Δ trades | Δ PnL |
   |---|---|---|
   | 2024 (c1) | +46 | −$269 (block strongly helps) |
   | Dec24–Aug25 (c2) | +35 | −$155 (helps) |
   | Aug25–May26 (c3) | +37 | **+$87 (block now *hurts*)** |

   Mondays were toxic in 2024/early-2025 but turned profitable in the last ~9 months.
   It's still net-positive to keep over the full sample, but this is the one gate worth
   **monitoring** — if Mondays keep improving, relaxing it is the most likely future
   frequency gain (it unlocks the most trades).
4. **Medium-ATR and exhaustion do almost nothing** (−$63 and ~$0). They could be
   simplified away with negligible PnL impact, but there's no reason to.

## Recommendation

Keep all gates as-is — none is leaking profit, so the counterfactual confirms the
current configuration rather than opening a new lever. The single thing to **watch**
is the Monday block: re-check it in a quarter or two; if Mondays stay positive, that's
the cleanest place to add frequency. Net across the whole frequency program, the only
change that improved the book remains **ADX 18→12** (already live).

Artifacts: `g_alloff_c{1,2,3}`, `g_monday_c{1,2,3}`, `g_lateaft_c{1,2,3}`,
`g_medatr_c{1,2,3}`, `g_exhaust_c2` in the working outputs.
