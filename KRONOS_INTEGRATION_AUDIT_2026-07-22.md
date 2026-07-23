# Kronos + MES Bot Integration Audit — 2026-07-22

Question: can Kronos (candlestick foundation model) give the MES bot real edge?
Method: **I ran Kronos on real MES 15m data and measured it** — not theory.

---

## TASK 1 — What Kronos is (verified from repo + running it)

| Property | Finding |
|---|---|
| Predicts | Future candlesticks (OHLCV) autoregressively, N bars ahead |
| Inputs | DataFrame `[open,high,low,close]` (+volume/amount optional) + timestamps |
| Probabilistic? | Yes — temperature/nucleus sampling; average multiple paths → P(up) |
| Sizes | mini 4M / small 25M / base 102M / large 500M (large closed) |
| Context | 512 tokens (small/base), 2048 (mini). Example uses 400 lookback |
| Local? | Yes — PyTorch, HF Hub, MIT license |
| Hardware | CPU works. **Measured on this Mac (CPU): load ~4s, next-bar predict ~0.6–1.2s at 20 sample paths** |
| Live every minute? | Yes, latency-wise trivial (~1s) |
| MES data? | Yes — fed it MES OHLCV directly, no problem |
| Repo's own claim | "**not a production-ready quantitative trading system**"; no edge metrics published |

Runs fine. The question was never "does it run" — it's "does it predict MES."

---

## TASK 2/3 — Does it have edge? (A/B test, actually run)

Walk-forward on **real IB MES 15m RTH, 1 year, 484 forecasts** spread across the
year. Context 256 bars → predict next bar, 20 sample paths → direction + P(up).
Net cost 0.65 pt. Trade Kronos's predicted direction vs realized.

| Metric | Kronos-small (25M) | Kronos-base (102M) |
|---|---|---|
| n forecasts | 484 | 162 |
| Directional accuracy | **51.2%** | **53.1%** |
| corr(pred, actual) | +0.075 | +0.061 |
| Net-of-cost meanR | **−0.032** | +0.003 |
| Net PF | **0.90** | ~1.0 |
| t-stat | **−0.89** | +0.05 |
| Gross (pre-cost) meanR | +0.028 (t=0.78) | — |

**Confidence subsets (does Kronos conviction help? — Option A/D):**
- Top 50% most-confident predictions: 51.2% acc, net −0.013R.
- Top 25%: **49.6% acc** (worse than 50%), net −0.046R.
- Top 9%: 55.1% (n=49 — one cherry from several cuts, noise).

**Kronos's own confidence does not identify better predictions.** Non-monotonic
→ it can't be used as a filter or a sizing signal.

### What this means for the ranked options
1. **Option A (entry confirmation)** — best-fit in theory, **useless in fact.**
   51% agreement with the future ≈ coin. Confirming a signal with a coin adds
   nothing; it removes ~as many winners as losers.
2. **Option D (risk sizing)** — dead. Confidence is non-monotonic (top-25% worse).
3. **Option B (regime detector)** — untested here, but MES 15m is a proven random
   walk (VR≈0.93); there is no directional regime to detect.
4. **Option C (exit improvement)** — same forecast, same 51% accuracy; no basis
   to time exits.

All four rank equally: **no measurable value**, because the forecast itself has
no edge.

---

## TASK 4 — Data pipeline (spec, for completeness — not recommended to build)

```
IB Gateway → MES 15m bars → last 256 bars as OHLCV DataFrame →
KronosPredictor.predict(pred_len=1, sample_count=20) → P(up) & pred_close →
[would feed strategy] → (NO order manager — signal-only account)
```
- Timeframe 15m · lookback 256 · horizon 1 bar · features OHLCV · output next-bar
  close distribution → P(up). Latency ~1s/bar on CPU. **Feasible, just pointless.**

---

## Why Kronos can't help here (the real reason)

Six prior audits proved MES 15m is a **random walk on real IB data**: variance
ratio ≈0.93, Hurst ≈0.55, lag-1 autocorr ≈−0.017. A random walk has **no
conditional structure to forecast** — the best possible next-bar prediction is
"≈ last close." Kronos, trained on 45 exchanges, dutifully outputs forecasts, but
on this instrument they land at 51–53% accuracy = coin. A 4× bigger model (base
vs small) moved accuracy from 51.2% to 53.1% with t=0.05 — i.e. **not at all.**
More parameters cannot manufacture signal that the instrument does not contain.

The tiny gross tilt (+0.028R, t=0.78) is (a) statistically insignificant and
(b) smaller than transaction costs → **unusable even if real.**

---

## TASK 5 — CAVEMAN VERDICT: **IGNORE**

- Ran Kronos on 484 real MES 15m bars. Accuracy **51.2%** = coin.
- Bigger model (102M): **53.1%**, t=0.05. No improvement. Size is not the problem.
- Net of costs it **loses** (PF 0.90). Gross tilt is below costs and t<1.
- Kronos confidence is **non-monotonic** — can't filter or size with it.
- Correlation with the future ≈ **+0.07** ≈ zero.
- MES 15m is a **random walk** (VR 0.93, Hurst 0.55) — nothing to forecast.
- Options A/B/C/D all inherit the same dead forecast; none adds edge.
- "AI = edge" is the exact assumption you told me to reject. Evidence rejects it.
- It runs great (~1s CPU). Running well ≠ predicting well.
- **Smallest useful integration = none.** Adding Kronos = more complexity, zero proven edge.

**Kill the idea.** Not because Kronos is bad tech (it's fine, MIT, fast), but
because you cannot forecast a coin. If the instrument ever shows real structure
(e.g. the NQ 3m/5m lead under forward validation), *then* a forecaster is worth
re-testing — on that data, with this same measured-edge bar, not on faith.

---

## Reproduce

```
# repo cloned to scratch; models auto-download from HF (MIT)
python3 /tmp/kronos_edge.py     # 484-point walk-forward, writes /tmp/kronos_edge_results.json
python3 /tmp/kronos_base.py     # base-model confirmation
```

## Caveman verdict

Take AI model. Feed real MES bar. Ask: next bar up or down?
Model guess right 51 of 100. Coin guess 50. Big model guess 53, still coin.
Net of cost, model lose money. Model sure-of-itself guess = not better guess.

Why? MES 15m is coin (proven, six audit, variance ratio 0.93). Coin have no
pattern. No brain, big or small, forecast coin. Kronos run fast, run local, run
every minute — but predict nothing here.

**IGNORE. Kill idea.** Not add AI to coin flip. If some market later show real
pattern, bring model back and measure again. Until then: model stay in cave.
Evidence beat hype. Every time.
