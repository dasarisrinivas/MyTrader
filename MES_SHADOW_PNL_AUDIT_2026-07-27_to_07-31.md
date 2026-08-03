# MES Shadow Signal P&L Audit — Week of Mon 2026-07-27 → Fri 2026-07-31

Shadow signals only. **No live trades — account is Cash, bot places no orders.**
All P&L below is hypothetical replay of the actual emitted signal stream against
real MES 1-minute market data pulled from IB. No signal logic modified. No
cherry-picking: every actionable signal in the window is included, losers and all.

---

## 1. Signal inventory

Source: `logs/mes_signals.jsonl` (bot's own shadow stream).

- **389 rows evaluated** during the week → **13 actionable** (12 BUY, 1 SELL), 376 HOLD.
- Contract quantity: signals carry no size → **1 MES contract assumed**.

| Day | Rows | BUY | SELL | HOLD |
|---|---|---|---|---|
| Mon 07-27 | 92 | 0 | 0 | 92 |
| Tue 07-28 | 83 | 4 | 0 | 79 |
| Wed 07-29 | 61 | 0 | 0 | 61 |
| Thu 07-30 | 92 | 8 | 0 | 84 |
| Fri 07-31 | 61 | 0 | 1 | 60 |

By strategy: EMA9_PB_LONG 7 · TREND_CONT_LONG 4 · EMA21_PB_LONG 1 · EMA21_PB_SHORT 1.
Direction: **12 long / 1 short** (long-biased; short side largely disabled in config).

---

## 2. Replay methodology

| Assumption | Value |
|---|---|
| Entry | next 1-min bar **open** after signal timestamp (can't fill at bar close) |
| Slippage | **1 tick (0.25 pt) adverse each side** — pay ask on buy, hit bid on sell |
| Fees | **$0.85/side → $1.70 round turn** (IBKR MES all-in: commission + exchange + reg) |
| Contract | 1 MES · tick 0.25 = $1.25 · 1 pt = $5 |
| Exit priority | 1) stop 2) target 3) EOD flatten at RTH close (20:00 UTC / 15:00 CT) |
| Intrabar tie | if a 1-min bar touches both stop and target → **stop assumed first** (conservative) |
| Data | IB MES 1-min bars (13,381 bars, 07-20 → 08-03) |

---

## 3. Per-trade results

| # | Signal (CT) | Dir | Setup | Entry | Exit | Pts | Gross | Net | R | Hold | Exit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 07-28 10:15 | BUY | EMA21_PB_LONG | 7453.50 | 7470.50 | +17.00 | +85.00 | **+83.30** | +1.03 | 8m | TARGET |
| 2 | 07-28 11:00 | BUY | TREND_CONT_LONG | 7481.00 | 7462.09 | −18.91 | −94.54 | **−96.24** | −1.01 | 132m | STOP |
| 3 | 07-28 11:45 | BUY | TREND_CONT_LONG | 7476.00 | 7466.75 | −9.25 | −46.25 | **−47.95** | −0.60 | 194m | EOD_FLAT |
| 4 | 07-28 12:00 | BUY | EMA9_PB_LONG | 7484.75 | 7470.63 | −14.12 | −70.61 | **−72.31** | −1.02 | 9m | STOP |
| 5 | 07-30 09:30 | BUY | TREND_CONT_LONG | 7440.00 | 7423.58 | −16.42 | −82.12 | **−83.82** | −1.02 | 17m | STOP |
| 6 | 07-30 11:15 | BUY | TREND_CONT_LONG | 7442.00 | 7465.51 | +23.51 | +117.57 | **+115.87** | +1.21 | 166m | TARGET |
| 7 | 07-30 12:00 | BUY | EMA9_PB_LONG | 7452.00 | 7461.59 | +9.59 | +47.96 | **+46.26** | +0.42 | 118m | TARGET |
| 8 | 07-30 12:15 | BUY | EMA9_PB_LONG | 7450.75 | 7468.37 | +17.62 | +88.11 | **+86.41** | +1.23 | 111m | TARGET |
| 9 | 07-30 12:45 | BUY | EMA9_PB_LONG | 7453.00 | 7469.78 | +16.78 | +83.89 | **+82.19** | +1.27 | 82m | TARGET |
| 10 | 07-30 13:15 | BUY | EMA9_PB_LONG | 7455.75 | 7471.12 | +15.37 | +76.87 | **+75.17** | +1.23 | 53m | TARGET |
| 11 | 07-30 13:30 | BUY | EMA9_PB_LONG | 7459.25 | 7472.14 | +12.89 | +64.44 | **+62.74** | +0.94 | 39m | TARGET |
| 12 | 07-30 14:00 | BUY | EMA9_PB_LONG | 7464.25 | 7475.31 | +11.06 | +55.29 | **+53.59** | +0.82 | 9m | TARGET |
| 13 | 07-31 10:15 | SELL | EMA21_PB_SHORT | 7476.25 | 7464.25 | +12.00 | +60.00 | **+58.30** | +3.00 | 6m | TARGET |

> **Note on #13's R = +3.00:** the fill came in 3.00 pts worse than the signal price
> (7476.25 vs 7473.25), which *shrank* actual risk to 4.0 pts vs the intended 7.0.
> The high R is a fill artifact, not superior signal quality.

---

## 4. Daily performance

| Date | Trades | Wins | Losses | Gross | Fees | Net | Win rate |
|---|---|---|---|---|---|---|---|
| Mon 07-27 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | — |
| Tue 07-28 | 4 | 1 | 3 | −126.40 | 6.80 | **−133.20** | 25.0% |
| Wed 07-29 | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | — |
| Thu 07-30 | 8 | 7 | 1 | +452.00 | 13.60 | **+438.40** | 87.5% |
| Fri 07-31 | 1 | 1 | 0 | +60.00 | 1.70 | **+58.30** | 100.0% |
| **TOTAL** | **13** | **9** | **4** | **+385.60** | **22.10** | **+363.50** | **69.2%** |

---

## 5. Overall performance

| Metric | Value |
|---|---|
| Total trades | 13 |
| Net P&L | **+$363.50** |
| Gross profit / gross loss | +$663.82 / −$300.32 |
| Fees | $22.10 |
| Profit factor | **2.21** |
| Win rate | **69.2%** (9W/4L) |
| Average winner / loser | +$73.76 / −$75.08 |
| Expectancy | **+$27.96/trade (+0.58 R)** |
| Max drawdown | **−$300.32** |
| Largest winning day / losing day | +$438.40 / −$133.20 |
| Longest win / loss streak | 8 / 4 |
| Average hold | 73 min |

Exit reasons: TARGET 9 (+$663.82) · STOP 3 (−$252.37) · EOD_FLAT 1 (−$47.95).

---

## 6. Strategy breakdown

| Strategy | Trades | Net P&L | EV/trade | Win rate | PF |
|---|---|---|---|---|---|
| EMA9_PB_LONG | 7 | **+$334.04** | +$47.72 | 85.7% | 5.62 |
| EMA21_PB_LONG | 1 | +$83.30 | +$83.30 | 100% | ∞ |
| EMA21_PB_SHORT | 1 | +$58.30 | +$58.30 | 100% | ∞ |
| **TREND_CONT_LONG** | 4 | **−$112.14** | −$28.03 | 25.0% | **0.51** |

- **Profitable:** EMA9_PB_LONG — but 7 of 7 fired on a *single* sustained uptrend on 07-30. One market event, not seven independent edges.
- **Losing / drawdown driver:** TREND_CONT_LONG (PF 0.51, 25% WR) — caused the entire 07-28 loss day and the worst single trade (−$96.24).

---

## 7. Time analysis (signal time, US/Central)

| Session | Trades | Net | EV | Win rate |
|---|---|---|---|---|
| OPEN 09-10 | 1 | −$83.82 | −$83.82 | 0% |
| MORNING 10-12 | 5 | +$113.28 | +$22.66 | 60% |
| LUNCH 12-13 | 4 | +$142.55 | +$35.64 | 75% |
| AFTERNOON 13-14 | 2 | +$137.91 | +$68.95 | 100% |
| CLOSE 14+ | 1 | +$53.59 | +$53.59 | 100% |

- Best window: **13:00–14:00 CT** (+$68.95 EV) — n=2. Worst: **09:00–10:00 CT** — n=1.
- **These n's are 1–5. No time-of-day conclusion is statistically usable.** Do not gate on this.

---

## 8. Risk analysis

**The headline number is not achievable on 1 contract.**

- Signals overlap heavily: taking every signal requires **up to 6 concurrent open positions** (07-30 stacked 8 signals in 4.5 hours). Peak margin ≈ **$8,400** (MES ~$1,400/contract overnight).
- Worst single trade: −$96.24. Worst day: −$133.20. Longest loss streak: 4 (all 07-28).
- Max drawdown on the take-everything path: **−$300.32**.
- Required account (1-contract, realistic): margin $1,400 + drawdown buffer ≥ 3× worst day → **~$3,000–5,000 minimum**; for the 6-contract stacked version, **~$12,000+**.

**Would it survive live trading?** On this week's data it would not have *blown up* —
but it also has not demonstrated an edge (see §10). A daily loss limit of $250 would
**not** have been hit (worst day −$133.20).

---

## 9. Baseline A vs B

| Path | Trades | Net | EV | Win rate | PF | Max DD |
|---|---|---|---|---|---|---|
| **A) Every shadow signal** | 13 | **+$363.50** | +$27.96 | 69.2% | 2.21 | −$300.32 |
| **B) 1-position-at-a-time** (production order-lock) | 5 | **+$77.41** | +$15.48 | 60.0% | 1.43 | −$180.06 |

B skips signals #3,4,7,8,9,10,11,12 (position already open).

**The production filter (one position at a time) cuts net P&L by 79%** — because the
one profitable event (the 07-30 trend) was captured *repeatedly* in path A by stacking
6 contracts into the same move. That is leverage on a single event, not diversified edge.
B is the honest number for a 1-contract account: **+$77.41 over a full week.**

---

## 10. VERDICT — MES SHADOW AUDIT

### Evidence strength: **VERY WEAK**

| Test | Result |
|---|---|
| Trade-level significance | mean +$27.96, sd $74.29, **t = +1.36** → not significant (need ~2.0) |
| R-level significance | mean +0.58R, **t = +1.75** → not significant |
| Independent day-level | only **3 active days**, t = **+0.72** → meaningless |
| **Remove the single 07-30 trend day** | **−$74.90 net, 40% win rate, EV −$14.98** |

**The entire week's profit is one trending Thursday.** Strip that one day and the
week loses money. 13 trades clustered into 3 days is not 13 independent bets — it is
closer to **3 bets, of which 1 won big.**

### Does MES have positive expectancy?
**Not demonstrated.** A +$363.50 week with t=1.36 is indistinguishable from luck.
This sits against six prior audits showing MES 15m is a **random walk** on real IB data
(variance ratio ≈0.93, Hurst ≈0.55, lag-1 autocorr ≈−0.017) where **no** price-only rule
survived costs. One good week does not overturn a year of data — it is exactly the kind
of variance a zero-edge system produces ~1 week in 3.

### Ready for paper trading?
**It already IS the safest form of paper trading** — signal-only, no orders, no capital
at risk. Keep it there. **Not ready for live capital.**

### Required before any live deployment
1. **≥100 non-overlapping trades** forward-logged (currently 13, heavily overlapping) — at least 3–4 months at this signal rate.
2. **Out-of-sample PF > 1.3 net of costs** sustained, with **t > 2** on independent observations.
3. **Fix the concurrency question** — decide 1-contract sequential (the honest path, EV +$15.48) vs stacked. Stacked results must never be quoted for a 1-contract account.
4. **Investigate TREND_CONT_LONG** (PF 0.51, 25% WR, drove all losses) — but on 4 trades, do **not** disable it yet. Disabling a setup on n=4 is exactly the curve-fitting sin already documented in this config (63 dated post-hoc tweaks, filters killed on 0/5 weeks).
5. **No parameter changes from this week.** One week is not evidence.

---

## Caveman verdict

Week look good on paper: **+$363.50, PF 2.21, win 69%.** Caveman not celebrate.

Three problem:
1. **All money come from one day.** Take away Thursday → week LOSE $74.90, win rate drop to 40%. Not thirteen bet. Three bet, one win big.
2. **Cannot get $363 with one contract.** Signals pile up — need SIX contract at once. Real one-contract path = **+$77.41**. Production lock cut 79% of profit.
3. **t = 1.36.** Not significant. Coin flip week look like this all the time.

Good part: TREND_CONT_LONG lose money (PF 0.51) while EMA9 pullback win — but seven of those seven fire on same one trend. One event, not seven edge.

**Verdict: keep shadow. No live money. Change nothing.** Thirteen trade over three day
prove nothing. Six audit already say MES 15m is coin. One green week not beat one year
of data. Log more week. Judge at hundred trade, not thirteen.
