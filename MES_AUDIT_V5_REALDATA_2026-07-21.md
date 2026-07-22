# MES Audit V5 — Verdict on REAL IB data (settles V2/V3/V4) — 2026-07-21

V4 said every quantitative conclusion was built on a corrupt replay series and
the gating task was to pull a clean, real, single-timeframe price series. **Done.**
Pulled 1 year of real MES 15m bars straight from IB (continuous contract,
read-only, spare client id — live bot untouched) and re-ran everything.

Data: `ContFuture MES` 15m **RTH, 7,514 bars, 252 sessions, 2025-07-22 → 2026-07-21.**
Forward returns computed within-session only (never across the overnight gap),
net of 0.65 pt round-trip cost.

---

## Market structure — SETTLED

| Test | Replay (V3) | **Real IB data (V5)** |
|---|---|---|
| Variance ratio VR(2/4/6/12) | 0.41/0.29/0.23/0.18 | **0.97 / 0.96 / 0.93 / 0.91** |
| lag-1 autocorrelation | −0.59 | **−0.017** |

**MES 15m RTH is a random walk.** VR≈0.93 (≈1.0), autocorr ≈0. It does **not**
mean-revert and does **not** trend at this timeframe. V3's "structurally
mean-reverting" was a pure artifact of the scrambled/mixed-timeframe replay
series (V4's suspicion, now proven on real data). The −0.59 autocorr was
stitching noise; real data gives −0.017.

---

## Edge hunt — nothing survives (net of cost, real data)

| System | n | WR | meanR | PF | t |
|---|---|---|---|---|---|
| RSI14 30/70 (K=6) | 1064 | 45.5% | −0.037 | 0.94 | −0.77 |
| EMA9/21 cross | 221 | 45.7% | −0.063 | 0.90 | −0.54 |
| Donchian-20 | 230 | 42.6% | −0.152 | 0.77 | −1.49 |
| ORB breakout (K=6) | 489 | 53.0% | −0.002 | 1.00 | −0.03 |
| ORB fade | 489 | 43.8% | −0.114 | 0.82 | −1.59 |
| Gap-fade | 249 | 45.4% | −0.082 | 0.91 | −0.56 |
| RSI2 10/90 (K=2) | 1994 | 46.4% | −0.070 | 0.81 | **−3.26** |
| Random null (matched) | — | — | −0.058 | — | — |

Best cell out of ~30 tested: RSI14 **20/80** K=6 → PF 1.47, but **t=1.64 on
n=136** — the false positive you expect from running 30 tests (no
multiple-testing correction would keep it). RSI2 10/90 is significantly
*negative* (t=−3.26) → short-horizon **momentum**, the opposite of reversion.
ORB breakout = 1.00 (dead random).

**Conclusion: no simple linear rule — trend OR reversion — produces a
cost-surviving, statistically-significant edge on one year of real MES 15m RTH
data.** Every strategy clusters on the random null. The instrument is efficient
at this timeframe against these tools.

---

## Every prior audit claim, final grade (on real data)

| Claim | Source | Final verdict |
|---|---|---|
| MES 15m mean-reverts | V3 | **FALSE** — random walk (VR 0.93) |
| RSI mean-reversion has edge / "right direction" | V2, V3 | **FALSE** — PF 0.94 net, loses, = random |
| A simple system beats the bot | V2 | **FALSE on real data** — all simple systems lose net |
| Bot has no demonstrated edge | V1–V4 | **TRUE** — and there is no 15m edge to have |
| Over-engineering / RAG crash / correlated filters / survivorship / native fills lose | V1–V4 | **TRUE** — from code/config/fills, unaffected |
| Signal-only is the correct posture | V3, V4 | **TRUE — strongest reason yet: no 15m linear edge exists** |

The replay-based audits (V2/V3) manufactured an edge that isn't there. V4 caught
the data bug; V5 proves the real answer with clean IB data.

---

## What this means

- **Do not build the RSI bot.** Do not rebuild the trend bot. Neither has edge on
  real 15m data.
- **Signal-only remains correct**, now for the deepest reason: at 15m with
  EMA/RSI/ORB/Donchian, MES offers no linear edge net of costs. Risk no capital
  on these signals.
- If a real edge exists, it is **not** in 15m linear price rules. Candidates,
  ranked, that this analysis did NOT rule out:
  1. **Different timeframe** — the random-walk result is specific to 15m RTH. Test 1m, 5m, daily. (data: same IB pull at other bar sizes)
  2. **Volatility, not direction** — VR≈1 kills directional edge but says nothing about forecastable variance. Test HAR-RV on 15m realized vol.
  3. **Overnight / ETH microstructure** — I have 3 months of ETH bars; RTH-only may hide gap/globex structure. Needs more ETH history.
  4. **Cross-asset lead** (ES/NQ/VIX → MES) — untested, needs synchronized feeds.
  5. **Event-conditioned** (FOMC/CPI/open-drive days) — regime-specific, not a blanket rule.

The V4 research list stands; its #1 (get clean data) is now **complete** and its
#2 (is 15m a random walk?) is **answered: yes.**

---

## Reproduce

```
python3 tools/download_mes_clean.py         # writes /tmp/mes_15m_{rth,eth}.parquet (cid=17, read-only)
# analysis inline in this session's history; VR + net-of-cost system sweep on the RTH parquet
```

## Caveman verdict — real bone, final

Got real MES bar from IB. One year. Clean. No replay rot.

Real bone say: **15m MES is coin flip.** VR 0.93. No bounce, no run. Every simple
rule — RSI, EMA, ORB, Donchian, gap-fade — lose to cost, sit on random line. Best
one (RSI 20/80) is luck from trying thirty rule, t=1.6, die under honest math.

So: V2 wrong (RSI no edge). V3 wrong (no mean-revert). V4 right (data was rotten).
V1 right about bloat but strategy has no edge to fix. **No 15m linear edge exist.
Trade nothing. Build nothing new here.** If edge live somewhere, it not in 15m
price line — look other timeframe, look volatility, look overnight, look
cross-asset. Signal bot log and wait. Truth beat every audit. This the truth.
