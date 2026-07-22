# NQ Edge Hunt — real IB data, scientific disproof attempt — 2026-07-21

Same protocol used to kill MES, applied to NQ. Goal: **disprove** that NQ has a
tradeable intraday price-only edge. Real IB data only (ContFuture NQ, read-only
cid=17, live bot untouched). Cost 0.75 pt round-trip (≈1 tick slippage/side +
commission; ~0.6–3.8% of ATR depending on timeframe).

---

## Headline

**NQ is a random walk at every timeframe (VR≈1.0, Hurst≈0.5, AC≈0), exactly
like MES. Cross-asset is dead (QQQ doesn't lead NQ, VXN unlocks nothing).
UNLIKE MES, a 3m/5m RSI mean-reversion signal shows apparent edge that survives
costs — but it lives only in 1–2 months of data, is absent at 15m over a full
year, and does not survive multiple-testing correction. It is a PROMISING LEAD,
not a proven edge.**

Verdict: **BUILD NQ SIGNAL BOT** (to forward-validate the 3m/5m lead, not to
trade).

---

## Phase 1 — Market structure (all timeframes, within-session)

| TF | bars | med ATR | cost %ATR | VR(2) | VR(6) | lag-1 AC | Hurst |
|---|---|---|---|---|---|---|---|
| 1m | 2250 | 19.9 | 3.8% | 1.02 | 1.02 | +0.017 | 0.54 |
| 2m | 4725 | 30.6 | 2.4% | 0.99 | 0.99 | +0.004 | 0.51 |
| 3m | 3150 | 38.6 | 1.9% | 0.99 | 1.05 | −0.004 | 0.55 |
| 5m | 3690 | 49.0 | 1.5% | 1.00 | 0.96 | +0.005 | 0.48 |
| 10m | 5691 | 58.7 | 1.3% | 0.97 | 0.93 | −0.016 | 0.53 |
| 15m | 7514 | 61.3 | 1.2% | 0.96 | 0.92 | −0.006 | 0.55 |
| 30m | 3758 | 90.9 | 0.8% | 1.00 | 0.95 | +0.013 | — |
| 60m | 3997 | 122.4 | 0.6% | 1.00 | 1.11 | +0.018 | — |

**Random walk everywhere.** VR≈1, Hurst≈0.5, autocorr≈0 at every timeframe. NQ
is **not** structurally different from MES. It neither trends nor mean-reverts at
the level of bar-return autocorrelation. Note NQ's cost-as-%ATR is *lower* than
MES (bigger ATR), so NQ gets a *fairer* chance — and still shows no structure.

---

## Phase 2 — Price-only rules, net of cost (K=6). tuple = (PF, t, n)

| TF | RSI14 | RSI2 | EMA× | Donchian | Momentum |
|---|---|---|---|---|---|
| 1m | 0.95,−0.3 | 0.96,−0.5 | 0.85,−0.6 | 0.77,−2.3 | 0.87,−2.7 |
| 2m | 0.99,−0.1 | 0.82,−2.5 | 0.90,−0.6 | 0.89,−1.4 | 0.98,−0.4 |
| **3m** | **1.55,+3.0** | 0.98,−0.2 | 1.22,+0.7 | 0.96,−0.3 | 1.01,+0.3 |
| **5m** | **1.41,+2.7** | 1.04,+0.4 | 0.80,−0.9 | 0.97,−0.3 | 0.98,−0.5 |
| 10m | 0.86,−1.7 | 0.88,−1.8 | 0.86,−0.7 | 0.91,−0.8 | 1.10,+2.2 |
| 15m | 0.90,−1.4 | 0.91,−1.4 | 0.93,−0.3 | 0.75,−1.5 | 1.04,+0.9 |
| 30m | 0.88,−1.1 | 1.09,+0.8 | 1.67,+1.8 | — | 0.87,−1.4 |
| 60m | 0.71,−1.7 | 0.97,−0.2 | — | — | — |

Almost everything loses or is insignificant. **Two cells stand out: 3m RSI
(PF 1.55, t=3.0) and 5m RSI (PF 1.41, t=2.7).**

### The 3m/5m RSI lead — attacked

| Test | 5m RSI | 3m RSI |
|---|---|---|
| Full | PF 1.41, t=2.74 | PF 1.55, t=2.99 |
| Train (1st half) | PF 1.09, t=0.52 | PF 1.34, t=1.32 |
| Test (2nd half, recent) | **PF 1.81, t=3.37** | PF 1.75, t=2.87 |
| Long only | PF 1.42, t=2.06 | PF 1.78, t=3.07 |
| Short only | PF 1.39, t=1.80 | PF 1.34, t=1.29 |
| Double cost (1.5pt) | PF 1.38, t=2.57 | PF 1.51, t=2.84 |

**In its favor:** survives doubled costs; works both sides; holds (strengthens)
in the recent half.
**Against it (decisive):**
1. **Sample is 1–2 months only** (IB caps intraday history: 5m ≈ 2 months, 3m ≈ 1
   month). n≈450/330. Cannot establish a *durable* edge — only "worked in
   June–July 2026." The "OOS-stronger" pattern = the edge concentrated in the
   most recent weeks = **regime-recent, not proven-durable.**
2. **Absent at 15m over a full year** (RSI PF 0.90, losing). A real RSI-reversion
   edge should survive at 15m; it doesn't. The effect exists only where history
   is short.
3. **Fails multiple-testing correction** (Phase 5 below).

---

## Phase 3 — Cross-asset (dead, same as MES)

- **QQQ → NQ:** contemporaneous corr 1.000 (same underlying); **lead corr(QQQ_t,
  NQ_t+1) = −0.011.** QQQ does not predict next-bar NQ.
- **VXN regime:** NQ lag-1 AC ≈ 0 in both regimes (low-VXN −0.057, high-VXN
  −0.003). No regime-conditional edge.
- (SPY/ES vs NQ would be the same story — all track the same index complex
  contemporaneously with no exploitable lead.)

---

## Phase 5 — Multiple-testing correction (kills the full-sample hits)

45 cells scanned (9 TF × 5 systems). Bonferroni at α=0.05 → **need |t| > 3.26.**
- 3m RSI full-sample t = 2.99 → **below threshold.**
- 5m RSI full-sample t = 2.74 → **below threshold.**
- Expected false positives at |t|>1.96 over 45 tests = **2.2.** Finding two cells
  at t≈2.7–3.0 is exactly what pure noise produces.

Only the 5m *recent-half* sub-sample (t=3.37) exceeds 3.26 — but a sub-sample of
a 2-month set is not independent evidence.

**Conclusion: no NQ price-only edge survives strict correction on the full
available sample.** The 3m/5m RSI signal is a lead, not a result.

---

## Final questions — direct answers

1. **Any statistically significant intraday edge?** Not one that survives multiple-testing on the full sample. A 3m/5m RSI signal is suggestive but unproven.
2. **Is NQ fundamentally different from MES?** **No** on structure (both random walks, both dead cross-asset). **Slightly** in that NQ's short-TF RSI shows a lead MES never did — but on too little data to trust.
3. **Best timeframe?** For the apparent signal, 3m/5m. For trustworthy structure, all are random walks.
4. **Survive costs?** The 3m/5m signal survives even doubled cost. So cost is not what kills it.
5. **Survive multiple-testing?** **No** (t below Bonferroni 3.26 on full sample).
6. **Trade my own money on it?** **No.** 1–2 months of data, fails correction, absent at 15m/full-year.
7. **If yes, why?** N/A.
8. **If no, why not?** Insufficient history + fails correction + no long-horizon corroboration = indistinguishable from a recent regime.
9. **Simplest architecture required?** A signal-only bot that logs a locked 5m RSI(14) 30/70 reversion signal (both sides, 1ATR/1ATR, K=6) and scores it forward — pre-registered, so future data is true out-of-sample. ≤6 params.
10. **Abandon liquid index futures entirely?** For **directional price-only** intraday: the evidence says these instruments are efficient (MES + NQ both random walks). The 3m/5m NQ lead is the only thread worth pulling, and only via **forward** validation. If it dies forward, then yes — abandon price-only index-futures intraday and move to non-OHLCV data (order-flow/internals) or a different market.

---

## Phase 8 — If no price edge: what needs non-OHLCV data

| Idea | Data | IB provides? | History | Effort | P(success) |
|---|---|---|---|---|---|
| Order-flow / cumulative delta | tick bid/ask | partial (tick) | limited | high | low-med |
| DOM / bid-ask imbalance | L2 depth | live only, no history | none retro | high | low-med |
| Volume/Market Profile | tick volume@price | yes (tick) | limited | med | low-med |
| TICK / ADD / VOLD internals | index feeds | yes (as indices) | yes | med | med |
| Event-driven (FOMC/CPI/NFP) | calendar + bars | bars yes, calendar external | yes | low | med |

None is a quick win; each needs new data plumbing.

---

## FINAL VERDICT

# BUILD NQ SIGNAL BOT

Not "BUILD NQ BOT" (no proven edge, would risk capital on 2 months). Not
"ABANDON" (a real, cost-surviving, both-sides 3m/5m RSI lead exists — too
interesting to discard). The disciplined move: **run a signal-only NQ bot that
emits and forward-scores a pre-registered 5m RSI reversion signal.** Locked
params → every future bar is honest out-of-sample. If it holds for ~3 months
forward at PF>1.3 net, *then* consider trading. If it decays like everything
else, abandon price-only index-futures intraday and go to internals/order-flow.

---

## Caveman verdict

Hunt NQ like hunt MES. Structure same: coin flip every timeframe, VR ~1, Hurst
half, no bounce no run. QQQ no lead NQ. VXN no help. Same dead index as MES.

BUT — one track in snow: 3m and 5m RSI bounce show profit, survive double cost,
work both side, stronger in recent moon. Not noise-shaped like MES was.

Still — track only 1-2 moon old (IB give no more short-bar history). Not show at
15m over full year. Fail Bonferroni (need t>3.26, got 2.7-3.0). Two lucky cell
from forty-five throw = expect two. So: maybe real, maybe recent-regime luck.
Cannot tell from old bone.

So not trade. Not abandon. **BUILD NQ SIGNAL BOT.** Lock the 5m RSI rule, log it,
score forward. Future bar = honest test. Three moon forward hold → real. Fade →
dead, then leave price-only index futures for good.

Evidence beat hope. NQ earn ONE more look — forward, log-only, no money. Not
because hope, because the 3m/5m track survive every attack except "too little
history," and only forward data fix that.
