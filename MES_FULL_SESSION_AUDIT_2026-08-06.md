# MES Shadow Engine — Full Session Validation Audit — 2026-08-06

**RESEARCH ONLY.** No production code, strategy logic, threshold, exit, confidence
model or filter was modified during this audit. The frozen engine is analyzed
exactly as it ran.

Sources: `logs/mes_signals.jsonl` (**1,049 emitted rows**, 2026-07-21 → 2026-08-06),
`logs/bot.log` gate diagnostics, real IB MES 1-minute bars (13,800 bars, full ETH).
Shadow only — Cash account, the bot places no orders. All P&L is hypothetical.

> **Config boundary in this window.** On 2026-08-04 23:42 CST the overnight
> allowlist was opened (`ft_overnight_allowed_signals: ["EMA9_PB_LONG"] → []`) at
> the user's explicit request. The audit therefore spans **two configurations** and
> segments results by era. This supersedes the 2026-08-03 audit.

---

## 1. SESSION COVERAGE VERIFICATION — **FULL SESSION (ETH + RTH). Expectation MET.**

**The engine evaluates the entire CME Globex session.** Evaluations per ET hour,
every hour of the clock:

| ET hr | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Evaluated | 35 | 43 | 44 | 44 | 44 | 44 | 44 | 44 | 46 | 49 | 48 | 48 |

| ET hr | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Evaluated | 49 | 48 | 48 | 46 | 40 | 13 | 37 | 48 | 48 | 48 | 47 | 44 |

Hour 17 ET is low (13) — that is the **CME daily maintenance halt** (17:00–18:00 ET);
no bars exist to evaluate.

**Confirming evidence:** 7 of 26 replayed signals fired outside core RTH, including
two in **US pre-market (08:00/08:15 ET)** that the previous configuration would have
suppressed. An RTH-gated engine could not produce these.

### 2. Enforcement points (file + line)

| # | Location | Purpose | Effective state |
|---|---|---|---|
| 1 | `shree/signal_bot/bot.py:102` | `useRTH=False` on IB fetch | **full ETH data ingested** |
| 2 | `shree/strategies/es_fifteen_min.py:583-590` | builds `_rth_start`/`_rth_end` | `config.yaml:705-708` = 00:00–23:59 ET |
| 3 | `es_fifteen_min.py:883` | `OUTSIDE_RTH` HOLD gate | **disarmed — 0 occurrences in logs** |
| 4 | `es_fifteen_min.py:573-580` | builds `_entry_start_et`/`_entry_end_et` | `config.yaml:667-670` = 00:00–23:59 ET |
| 5 | `es_fifteen_min.py:889` | `OUTSIDE_ENTRY_WINDOW` HOLD gate | **disarmed — 0 occurrences in logs** |
| 6 | `es_fifteen_min.py:1275-1291` | overnight family allowlist | **NOW EMPTY → gate skipped entirely** |
| 7 | `es_fifteen_min.py:373, 744` | OR anchored to 09:30 ET (`_core_rth_start`) | structural, unchanged (see §5) |

**No session filter is currently restricting signal families.** Gates 3 and 5 exist
in code but are dead branches (grep: `OUTSIDE_RTH` = 0, `OUTSIDE_ENTRY_WINDOW` = 0).
Gate 6 is now inert because `if _is_overnight_pb and self._overnight_allowed_signals:`
is falsy on an empty set.

**Intentional or accidental?** Entirely **intentional**. Gates 3/5 were widened by
config (documented `# was 9:30` / `# was 16:00` comments). Gate 6 was opened
deliberately on 2026-08-05 with rationale recorded inline in `config.yaml:673-684`.
**No accidental regression found.**

---

## 3. SIGNAL GENERATION TIMELINE

### Hourly (exchange time, US/Eastern)

| ET Hr | Evald | Signals | Long | Short | % tot | AvgConf | AvgADX | Net $ | EV | PF | Win % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00–07 | 342 | 0 | — | — | 0% | — | — | — | — | — | — |
| **08** | 46 | 2 | 2 | 0 | 7.7% | 0.70 | 26.8 | **+106.93** | +53.47 | ∞ | 100% |
| 09 | 49 | 0 | — | — | 0% | — | — | — | — | — | — |
| 10 | 48 | 1 | 1 | 0 | 3.8% | 0.70 | 35.3 | −83.82 | −83.82 | 0.00 | 0% |
| 11 | 48 | 3 | 2 | 1 | 11.5% | 0.70 | 22.8 | +67.49 | +22.50 | 1.91 | 66.7% |
| **12** | 49 | **7** | 7 | 0 | **26.9%** | 0.70 | 27.4 | −10.74 | −1.53 | 0.94 | 42.9% |
| 13 | 48 | 5 | 5 | 0 | 19.2% | 0.70 | 29.1 | +82.10 | +16.42 | 1.62 | 60% |
| 14 | 48 | 2 | 2 | 0 | 7.7% | 0.70 | 32.5 | +137.91 | +68.95 | ∞ | 100% |
| 15 | 46 | 1 | 1 | 0 | 3.8% | 0.70 | 32.9 | +53.59 | +53.59 | ∞ | 100% |
| 16–18 | 90 | 0 | — | — | 0% | — | — | — | — | — | — |
| 19 | 48 | 3 | 3 | 0 | 11.5% | 0.70 | 29.0 | −2.19 | −0.73 | 0.97 | 66.7% |
| 20 | 48 | 2 | 2 | 0 | 7.7% | 0.70 | 28.6 | +57.85 | +28.92 | ∞ | 100% |
| 21–23 | 139 | 0 | — | — | 0% | — | — | — | — | — | — |

**Confidence is a constant 0.70 on every single signal** — it carries zero
discriminating information. There is **no "quality score" field** in the emitted
stream; reported as absent rather than invented.

### 30-minute buckets (ET) — buckets with signals

| Bucket ET | Sigs | % tot | Long | Short | AvgConf | Net $ | EV | Win % |
|---|---|---|---|---|---|---|---|---|
| 08:00–08:29 | 2 | 7.7% | 2 | 0 | 0.70 | +106.93 | +53.47 | 100% |
| 10:30–10:59 | 1 | 3.8% | 1 | 0 | 0.70 | −83.82 | −83.82 | 0% |
| 11:00–11:29 | 2 | 7.7% | 1 | 1 | 0.70 | +141.60 | +70.80 | 100% |
| 11:30–11:59 | 1 | 3.8% | 1 | 0 | 0.70 | −74.11 | −74.11 | 0% |
| 12:00–12:29 | 4 | 15.4% | 4 | 0 | 0.70 | −0.01 | −0.00 | 50% |
| 12:30–12:59 | 3 | 11.5% | 3 | 0 | 0.70 | −10.73 | −3.58 | 33.3% |
| 13:00–13:29 | 3 | 11.5% | 3 | 0 | 0.70 | +60.36 | +20.12 | 66.7% |
| 13:30–13:59 | 2 | 7.7% | 2 | 0 | 0.70 | +21.74 | +10.87 | 50% |
| 14:00–14:29 | 1 | 3.8% | 1 | 0 | 0.70 | +75.17 | +75.17 | 100% |
| 14:30–14:59 | 1 | 3.8% | 1 | 0 | 0.70 | +62.74 | +62.74 | 100% |
| 15:00–15:29 | 1 | 3.8% | 1 | 0 | 0.70 | +53.59 | +53.59 | 100% |
| 19:30–19:59 | 3 | 11.5% | 3 | 0 | 0.70 | −2.19 | −0.73 | 66.7% |
| 20:00–20:29 | 2 | 7.7% | 2 | 0 | 0.70 | +57.85 | +28.92 | 100% |

### Session breakdown

| Session | Sigs | Win % | Net $ | EV | PF | Avg hold | t |
|---|---|---|---|---|---|---|---|
| Overnight (18–03 ET) | 5 | 80.0% | +55.66 | +11.13 | 1.86 | 113m | +0.55 |
| Europe (03–08 ET) | **0** | — | — | — | — | — | — |
| **US Pre-mkt (08–09:30)** | 2 | 100% | +106.93 | +53.47 | ∞ | 81m | +43.75 |
| RTH Morning (09:30–12) | 4 | 50.0% | −16.33 | −4.08 | 0.90 | 27m | −0.09 |
| Lunch (12–14 ET) | 12 | 50.0% | +71.35 | +5.95 | 1.22 | 80m | +0.32 |
| Afternoon (14–15 ET) | 2 | 100% | +137.91 | +68.95 | ∞ | 46m | +11.09 |
| Closing Hr (15–16 ET) | 1 | 100% | +53.59 | +53.59 | ∞ | 9m | — |
| Post-close (16–18 ET) | **0** | — | — | — | — | — | — |

---

## 4. FULL SHADOW P&L — every emitted signal, no filtering

**Assumptions:** fill = next 1-min bar open after signal ts; slippage 1 tick (0.25 pt)
adverse each side; fees $1.70 round turn (IBKR MES all-in); 1 contract; exits
stop → target → **120-min time stop** (engine's own `ft_max_hold_bars: 8`); stop
assumed first on intrabar ties.

### ⚠️ Two figures — read both

IB serves only ~10 days of 1-minute history, so **3 actionable signals from 07-22
fall outside the bar window.** All three are **known losers** (−$50.21, −$46.36,
−$49.41 = −$145.98), measured in the 2026-07-22 daily audit. Excluding them
**flatters the result**. Both versions are reported; **B is the honest one.**

| Metric | A) replayed only (n=26) | **B) ALL 29 actionable** |
|---|---|---|
| Net P&L | +$409.11 | **+$263.13** |
| Win rate | 65.4% (17W/9L) | **58.6% (17W/12L)** |
| Profit factor | 1.74 | **1.38** |
| EV / trade | +$15.74 | **+$9.07** |
| Gross profit / loss | +$958.68 / −$549.57 | +$958.68 / −$695.55 |
| Max drawdown | −$392.16 | ≤ −$392.16 |
| Longest losing streak | 5 | 5+ |
| Average winner / loser | +$56.39 / −$61.06 | +$56.39 / −$57.96 |
| Average hold | 73 min | — |
| **t-statistic** | **+1.34** | **lower than +1.34** |

Exit reasons (n=26): TARGET 12 · STOP 7 · TIME_STOP 7.

**Even the flattered version has t = +1.34 — below the ~2.0 significance bar.**

---

## 5. TIME-OF-DAY PERFORMANCE

**Strongest:** 08:00 ET (+$53.47 EV, n=2), 14:00 ET (+$68.95 EV, n=2), 15:00 ET (n=1).
**Weakest:** 10:00 ET (−$83.82, n=1), 12:00 ET (7 signals — the densest hour — at PF 0.94, essentially flat).

**Statistically significant windows: NONE.** Every hour has n ≤ 7. The eye-catching
`t=+43.75` (pre-market) and `t=+11.09` (afternoon) are artifacts of **2 same-direction
trades in one move** — not evidence. No time-of-day gating is justified by this data.

---

## 6. MISSING-SESSION DETECTION

Zero-signal windows, all **evaluated normally** (~44–48 bars/hour):

| Window (ET) | Evaluated | Cause — evidence |
|---|---|---|
| 00:00–07:00 | 342 | **`B:no_OR` / `E:no_OR` 43–44 per hour.** Opening Range is anchored to 09:30 ET (`es_fifteen_min.py:744`); before the RTH open, OR-breakout families are *structurally impossible*. Remaining families blocked by unresolved EMA stack in thin liquidity. |
| 16:00–18:00 | 90 | Includes the **CME maintenance halt** (17:00–18:00 ET, only 13 evaluations — no bars exist). |
| 21:00–23:00 | 139 | `B/E:no_cross` + `C:stack` — OR exists but price never crosses; EMA stack unresolved. |

Ruled out with evidence: market closed (no — bars present except the halt), session
filter (no — all gates disarmed, §2), missing data (no — §7), indicator unavailable
(no — ADX/ATR present and varying every row), scheduler asleep (no — 96.5% exact
15-min cadence), polling stopped (no), shadow disabled (no), **bug (no)**.

---

## 7. MARKET DATA COVERAGE — clean

- **Cadence:** 1,011 of 1,048 inter-row gaps are exactly 15 min (**96.5%**). Polling frequency unchanged.
- **Bar completeness:** 600 1-min bars per ET hour across all 24 hours **except hour 17 ET = 0** (CME daily halt — correct).
- **Gaps > 20 min: 24 total, all explained** — recurring 75-min gaps at 16:00→17:15 ET (maintenance halt), 30-min gaps at ~23:30→00:00 (session date-roll fetch timing), and weekend closures.
- **Indicators:** updating continuously overnight (ADX/ATR present and varying).
- **Duplicates:** 1,045 unique `bar_ts` of 1,049 rows → 4 re-emits on the same bar (bot restarts). Cosmetic; all are HOLD rows, no P&L impact.
- **Ticks:** not applicable — the engine consumes 15-minute bars, not ticks.

---

## 8. RTH vs FULL SESSION

| Slice | Sigs | Win % | Net $ | EV | PF | Max DD | t |
|---|---|---|---|---|---|---|---|
| **Core RTH (09:30–16 ET)** | 19 | 57.9% | +$246.51 | +$12.97 | 1.51 | −$327.12 | +0.86 |
| **Outside RTH (ETH)** | 7 | 85.7% | +$162.59 | +$23.23 | **3.50** | $0.00 | +1.45 |
| **FULL SESSION** | 26 | 65.4% | +$409.11 | +$15.73 | 1.74 | −$392.16 | +1.34 |

### Config-era split (the live A/B)

| Era | Sigs | Win % | Net $ | EV | PF | t |
|---|---|---|---|---|---|---|
| ALLOWLIST (pre 08-04) | 24 | 62.5% | +$302.17 | +$12.59 | 1.55 | +1.00 |
| **FULL_ETH (post 08-04)** | 2 | 100% | +$106.93 | +$53.47 | ∞ | +43.75 |

**Does overnight improve performance?** On these numbers, ETH looks *better* than RTH
(PF 3.50 vs 1.51). **This must not be believed yet:**
- n=7, t=+1.45 — not significant.
- 4 of the 7 ETH trades are the same `EMA9_PB_LONG` firing on consecutive bars in one 08-02 move — **one event, not four**.
- The 2 FULL_ETH-era signals are 2 `TREND_CONT_LONG` on consecutive bars in one 08-05 pre-market move — **one event, not two**.
- This is the **opposite sign** to the controlled 2-month replay in `ETH_SUPPRESSION_RESEARCH_2026-08-05.md`, which measured the incremental overnight set at **−$135.20, PF 0.78** over 26 signals. That study had a clean RTH control and 4× the incremental sample. **When two studies disagree, trust the one with the control and the larger sample.**

---

## 9. SIGNAL DENSITY

```
00–07 ET |  0    (evaluated 342, zero signals)
08:00 ET |  2 ###########
09:00 ET |  0    (evaluated 49, zero signals)
10:00 ET |  1 ######
11:00 ET |  3 #################
12:00 ET |  7 ########################################   <- peak, PF 0.94
13:00 ET |  5 #############################
14:00 ET |  2 ###########
15:00 ET |  1 ######
16–18 ET |  0    (evaluated 90, incl. CME halt)
19:00 ET |  3 #################
20:00 ET |  2 ###########
21–23 ET |  0    (evaluated 139, zero signals)
```

**Clustering is the dominant risk characteristic.** 73% of signals fall in 10:00–15:00
ET; the rest in two narrow bands (08:00 and 19:00–20:00 ET). Within days the clustering
is worse: 07-30 produced 8 signals in one move, 08-02 produced 4 in 45 minutes, 08-05
produced 2 in 15 minutes. **26 trades behave like roughly 8–10 independent events**,
which is why the equity curve swings hard (−$392 drawdown on a +$409 gross result).

**Cumulative P&L path:** −$302.54 (07-26 trough) → +$3.38 (07-30) → +$182.38 (08-02)
→ +$409.11 (08-05). Underwater for the first third of its life.

---

## 10. ROOT-CAUSE ANALYSIS

**Question: "If signals occur only during RTH, explain why."**
**The premise is false.** 7 of 26 signals fired outside RTH, including 2 in pre-market.

Causes of the *residual* concentration in RTH, ranked by evidence:

1. **RTH-anchored Opening Range** (`es_fifteen_min.py:373, 744`) — OR-breakout families B/E cannot fire before 09:30 ET. Confirmed by 43–44 `no_OR` blocks per overnight hour. **Intentional** (in-code comment documents a MAY 12 2026 revert after a widened session window corrupted the OR).
2. **EMA-stack conditions rarely resolve in thin overnight liquidity** — a market property, not a code restriction. Supported by the measured ETH/RTH feature ratios (ATR 0.63×, range 0.50×, volume 0.14× — `ETH_SUPPRESSION_RESEARCH_2026-08-05.md`).
3. **~~Overnight family allowlist~~** — **no longer a cause.** Opened to `[]` on 2026-08-05; 0 `OVERNITE_ALLOWLIST` blocks since.

**No scheduling restriction, no market-hours filter, no replay restriction, no IB data
limitation, no indicator limitation, and no accidental regression were found.**

---

## 11. RECOMMENDED NEXT RESEARCH *(research only — no production changes)*

1. **Change nothing.** PF 1.38 (honest figure) at t < 1.34 over 29 trades is noise. Every session/hour/family split has n ≤ 12.
2. **Do not act on the "ETH beats RTH" reading.** It contradicts the controlled 2-month study (−$135.20, PF 0.78, n=26 incremental, with a byte-identical RTH control). Two events dressed as seven observations.
3. **Let the opened allowlist run.** It is a live A/B costing nothing (shadow, no orders). Revisit when the **ETH set reaches n ≈ 100** — at the current rate (~7 ETH signals per 16 days) that is roughly 8 months.
4. **Fix the measurement blind spot:** 1-minute bar history expires after ~10 days, so signals older than that can no longer be replayed. **Archive daily 1-min bars now** or all future audits will silently drop their oldest (and, as seen here, most-losing) signals — a real survivorship risk in the *tooling*, not the strategy.
5. **Confidence is a constant 0.70** across all 29 signals — decorative, not probabilistic. Build a calibration curve only after ≥100 outcomes; do not use it for filtering or sizing before then.
6. **Context:** six prior audits established MES 15m is a random walk on clean IB data (VR ≈0.93, Hurst ≈0.55, lag-1 autocorr ≈−0.017). PF 1.38 at t≈1.2 does not contradict the null of zero edge.

---

## Assumptions & limitations (stated)

- All P&L is **hypothetical** — shadow only, no orders, no capital at risk.
- Fill model: next 1-min bar open + 1 tick adverse each side. Overnight fills are likely worse in reality (volume 0.14× RTH) → **ETH results here are optimistic**.
- Time stop 120 min = engine's own `ft_max_hold_bars: 8`, applied uniformly to RTH and ETH so the two are comparable.
- Intrabar stop-vs-target ties resolved **stop-first** (conservative), applied equally everywhere.
- 3 of 29 actionable signals could not be replayed (1-min bars expired); their previously-measured values are folded into the "ALL 29" column.
- The audit window spans a **config change**; era-split reported, but the FULL_ETH era has only 2 signals and supports no conclusion.
- Exchange time reported in **US/Eastern** (matching the engine's internal ET logic); CT used only where labeled.

---

## Caveman verdict

Engine look at whole session. All 24 hour, 44–48 check each hour. Data clean — 600
bar per hour, only hole is CME halt 17:00 ET. Cadence 96.5% exact. No bug, no
missing candle, no sleeping scheduler.

**No session filter block anything now.** Old RTH gate dead (0 hit). Entry window
dead (0 hit). Night allowlist opened 08-05 — already give 2 pre-market signal that
old config would kill, both win.

Money: careful here. Replayed 26 trade show **+$409, PF 1.74**. But 3 known loser
from 07-22 fall outside bar window because IB only keep 10 day of 1-minute bone.
Put them back: **+$263, PF 1.38, WR 58.6%**. Second number is the true one. Even
that carry **t = 1.34** — under the bar. Still coin.

Night look better than day (PF 3.50 vs 1.51). **Do not believe.** Seven trade, and
four of them same setup in one move, two more same setup in another move. That is
two event wearing seven costume. Controlled two-month study with clean control say
opposite: night incremental **−$135, PF 0.78**. Trust study with control.

**Recommend: change nothing.** Let opened night run — cost nothing, no order, no
money. Judge at hundred. One thing to fix in *tooling* not strategy: save 1-minute
bar every day, else future audit keep losing its oldest trade — and today the
oldest three were all loser. Tool that forget loser make every audit look better
than truth.
