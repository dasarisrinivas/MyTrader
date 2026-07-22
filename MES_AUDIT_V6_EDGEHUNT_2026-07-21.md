# MES Audit V6 — Edge Hunt across timeframes + cross-asset (real IB data) — 2026-07-21

V5 proved 15m RTH is a random walk. V6 asks: **is there ANY timeframe or
cross-asset relationship where a measurable edge lives?** Pulled real IB data at
1m/5m/15m/30m/60m (RTH + ETH) plus SPY and VIX, and tested structure + simple
rules net of costs + cross-asset lead/regime.

All data read-only from IB (cid=17), live signal bot untouched.

---

## The one-line answer

**No. MES is a random walk at every timeframe from 1m to 60m, no price-only
linear rule survives costs at any of them, SPY does not lead MES, and VIX regime
does not unlock directional edge. There is no demonstrated intraday linear edge
to find.**

---

## Cross-timeframe structure + net-of-cost rules (K=6, 0.65 pt round trip)

| TF | bars | med ATR | cost as %ATR | VR(2) | VR(6) | lag-1 AC | RSI | EMA× | Donchian |
|---|---|---|---|---|---|---|---|---|---|
| 1m RTH | 2250 | 3.1pt | **21%** | 1.00 | 0.93 | −0.001 | 0.76 | 0.62 | 0.45 (t−6.5) |
| 5m RTH | 3690 | 7.2pt | 9% | 0.97 | 0.92 | −0.028 | 1.17 (t+1.2) | 0.58 | 0.91 |
| 15m RTH | 7514 | ~11pt | ~6% | 0.97 | 0.93 | −0.017 | 0.94 | 0.90 | 0.77 |
| 30m RTH | 3758 | 16.7pt | 3.9% | 1.00 | 0.96 | +0.013 | 0.84 | 1.11 (t+0.3) | — |
| 60m RTH | 3997 | 23.8pt | 2.7% | 1.01 | 1.15 | +0.004 | 1.11 (t+0.6) | — | — |
| 1m ETH | 5795 | 1.8pt | 36% | 1.00 | 0.93 | −0.000 | 0.63 | 0.54 | 0.39 (t−12) |
| 5m ETH | 12001 | 4.3pt | 15% | 0.97 | 0.95 | −0.032 | 0.95 | 0.70 | 0.66 |

**Read:**
- **Variance ratio ≈ 1.0 and lag-1 autocorr ≈ 0 at every timeframe.** Random walk
  everywhere. No timeframe mean-reverts or trends.
- **Every simple rule loses or is insignificant net of costs.** The few PF>1 cells
  (5m RSI 1.17, 30m EMA 1.11, 60m RSI 1.11) all have **t<1.3** and are cherry-picked
  from ~20 cells — noise, not edge.
- **Shorter is strictly worse:** cost drag rises from 2.7% of ATR (60m) to 21–36%
  (1m) while structure never improves. Going faster pays more to trade the same
  random walk.

---

## Cross-asset (the last places edge could hide)

**SPY → MES lead-lag** (6,250 aligned 15m bars):
- Contemporaneous corr 0.999 (same underlying).
- **LEAD corr(SPY_t, MES_t+1) = −0.011** → SPY does not predict next-bar MES.
- Trading SPY-sign → next MES: PF 0.76, **t = −7.5** (significantly loses).

**VIX regime** (median VIX 17.2):
- Volatility expands with VIX (|ret| 10.7bp high vs 6.4bp low) — expected, not
  directionally tradeable.
- lag-1 AC ≈ 0 in both regimes (low-VIX −0.037, high-VIX +0.006). No
  regime-conditional mean-reversion.

Both dead. MES is efficient against price-and-VIX-only information.

---

## Direct answers to the 10 questions

1. **Is the 15-minute MES strategy fundamentally dead?** **Yes.** Random walk (VR≈0.93), every rule loses net of costs.
2. **Stop investing engineering time in it?** **Yes.** Optimizing a random walk is optimization theater.
3. **What timeframe deserves investigation next?** **None of 1m–60m.** All are random walks; shorter ones are worse after costs.
4. **Is 1-minute worth pursuing?** **No.** Worst cost drag (21–36% of ATR), same random walk, most negative rule results. It is the worst choice, not the hidden edge.
5. **If no timeframe shows edge, what hypotheses next?** Only the ones this analysis could **not** test with bar-OHLCV — and each needs data/products I don't have:
   - **Order-flow / footprint / tape** (bid-ask aggression, cumulative delta) — requires L2/tick data.
   - **Market internals / breadth** (TICK, ADD, VOLD) — requires index feeds.
   - **Event-conditioned** (FOMC/CPI/open-drive days) — requires an economic calendar overlay.
   - **Volatility as the traded quantity** (options/vol structures) — a different product, not directional MES futures.
   These are the only unexplored frontiers. None is a quick win; each is a research project with new data dependencies.
6. **If you had to build a profitable MES signal bot today, what would it look like?** You **cannot** build one from price-only intraday bars — the evidence says no such edge exists. The honest build is **not a strategy bot** but a **research + signal-logging harness**: log candidate signals, score them forward, and only promote one if it clears out-of-sample PF>1.15 net, multiple-testing corrected. Trade nothing until then.
7. **Simplest architecture for that edge?** The current signal-only bot + a nightly outcome scorer + a clean multi-timeframe data pipeline (the `download_mes_*` tools). No strategy complexity, because no strategy has earned it.
8. **Delete immediately?** Already done: execution stack, RAG/LLM, sentiment, VX, optimizer, manager-veto, 1m-era strategies, risk manager (~61k lines). Next: the 120-knob config (no edge to tune).
9. **Rebuild?** Nothing to rebuild into a strategy yet — there is no validated edge to build around. Rebuild the *research process*, not the strategy.
10. **What would I personally build for my own money?** A signal-logging + validation harness on MES, and I would **not risk capital on any intraday price-only rule** — the data says it's a coin flip net of costs. I would spend the research budget on the Q5 frontiers (order-flow, breadth, event days), and if none produced a forward-validated edge, I would trade MES intraday **not at all** and redeploy the effort to a market/instrument that isn't efficient against my tools.

---

## Architecture scorecard (unchanged, confirmed)

| Subsystem | Verdict |
|---|---|
| Execution/RAG/LLM/sentiment/VX/optimizer/manager-veto/risk/1m-strategies | **DELETE** ✅ done |
| signal_bot, feature_engineer, logging, download tools | **KEEP** |
| es_fifteen_min strategy | **DELETE/RETIRE** — no edge at 15m or any TF |
| 120-knob config | **DELETE most** — nothing to tune on a random walk |
| Confidence field on signals | **DEMOTE to descriptive** — it is theater without a calibrated edge |

---

## Fake complexity / survivorship (confirmed, all prior findings hold)

120 `ft_` knobs, 63 dated post-hoc tweaks, filters killed on 0/5-WR weeks,
3-decimal confidence with no calibration — all curve-fitting on a random walk.
The whole tuning history was optimizing noise.

---

## Roadmap

| # | Task | Hrs | Risk | Files | Validation | Commit |
|---|---|---|---|---|---|---|
| 1 | Nightly signal-outcome scorer (log-only) | 3 | low | new `tools/score_signals.py` | replays own output | `feat(signal): nightly outcome scorer` |
| 2 | Retire es_fifteen_min; signal bot emits a neutral baseline + regime only | 2 | low | `shree/signal_bot/` | tests | `refactor(signal): retire dead 15m strategy` |
| 3 | Delete 120-knob config; keep ≤10 descriptive params | 2 | low | `config.yaml`, `shree/config/` | import + tests | `refactor(config): strip un-tunable knobs` |
| 4 | Order-flow / breadth data spike (research, not code) | 8 | med | new research dir | VR/edge test on new data | `research(mes): order-flow feasibility` |
| 5 | Decision: if no frontier edge in 30d, stop intraday MES | — | — | — | scorecard | — |

### 30 / 60 / 90 day

- **30:** ship scorer; retire the dead 15m strategy; strip config. Bot = pure signal logger.
- **60:** run one Q5 frontier spike (order-flow or breadth) with real data; VR/edge-test it the same way.
- **90:** if a frontier clears OOS PF>1.15 net (multiple-test corrected), build the minimal bot for it. **If not, stop trading MES intraday** — the evidence does not support it, and honesty beats sunk cost.

---

## Caveman verdict — hunt over

Chase edge across every timeframe. 1 minute, 5, 15, 30, 60. Day and night.
Every one: **coin flip.** VR ~1. No bounce, no run. Every price rule bleed to
cost. Short timeframe worst — pay most, get same coin.

Ask SPY to lead MES: SPY say nothing (corr −0.01). Ask VIX for regime: vol grow
but direction stay coin. No cross-asset edge.

So truth, final and cold: **no intraday price-only edge in MES. Not 15m, not any
timeframe, not from SPY, not from VIX.** Stop tuning. Stop chasing timeframe.
Random walk cannot be optimized.

Where edge maybe hide: order-flow tape, market breadth, event day, volatility
product. All need new data, none proven, none quick. Until one prove itself:
**trade nothing.** Signal bot log and score. Build no strategy on a coin.

If own money: I not trade MES intraday on price alone. Data say coin flip. Put
research on order-flow/breadth; if still coin, hunt different market. Do not feed
money to a random walk because code already exist.
