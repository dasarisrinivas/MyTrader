# Research — Is the Overnight Suppression Still Justified?

**RESEARCH ONLY. Zero production changes.** No strategy, indicator, threshold,
scoring, confirmation, exit, risk, confidence or ranking logic was modified. The
frozen production engine (`shree/strategies/es_fifteen_min.py`) was executed
as-is, twice, changing exactly one input.

---

## 0. Headline finding first

**`shadow_full_eth` requires no new code.** The gate is:

```python
# es_fifteen_min.py:1275
if _is_overnight_pb and self._overnight_allowed_signals:
```

An empty set is falsy → the entire allowlist block is skipped. The in-code doc
already says *"Empty list = no restriction."* So:

```
shadow_full_eth = true   ≡   ft_overnight_allowed_signals: []
```

A config value, not a code change. Nothing was added to the repo's runtime path.

---

## 1. Method

The frozen strategy was instantiated twice over **identical** bars:

| Variant | `ft_overnight_allowed_signals` |
|---|---|
| **A) PRODUCTION** | `["EMA9_PB_LONG"]` |
| **B) FULL_ETH** | `[]` (no restriction) |

Everything else identical. Each run fully sandboxed (temp cwd + `ft_counter_file`
override + redirected `_SHADOW_LOG_PATH`) so signal caps, opening-range JSON and
the shadow decision log **never touched production state**.

- **Data:** real IB MES 15m, `useRTH=False`, **3,859 bars, 52 sessions, 2026-06-07 → 2026-08-05** (1,068 RTH / 2,791 ETH bars).
- **Replay:** next-bar-open fill, 1 tick (0.25 pt) adverse slippage per side, $1.70 round-turn IBKR fees, 1 contract, exits stop → target → 8-bar time stop (engine's own `ft_max_hold_bars`), stop assumed first on intrabar ties.
- **OR logic untouched** — OR families remain naturally inactive before the RTH open, exactly as instructed. No OR values fabricated.

### Control validation (proves only one thing changed)

| Slice | A) production | B) full_eth |
|---|---|---|
| **RTH** | n=43, net **+$176.90**, PF 1.19, t=+0.50 | n=43, net **+$176.90**, PF 1.19, t=+0.50 |

**RTH output is byte-identical.** The only delta is overnight. Clean experiment.

---

## 2. Results

### Deliverable 1 — signals currently suppressed
**26 fully-qualified signals** over 52 sessions (~13/month). *(Live shadow had
observed only 6 in ~2 weeks — consistent rate, far better sample here.)*

### Deliverable 2–5 — incremental effect of enabling full ETH

| Metric | A) PRODUCTION | B) FULL_ETH | Delta |
|---|---|---|---|
| Signals | 43 | 69 | **+26** |
| Net P&L | **+$176.90** | **+$41.70** | **−$135.20** |
| EV/trade | +$4.11 | +$0.60 | −$3.51 |
| Win rate | 48.8% | 47.8% | −1.0 pp |
| Profit factor | **1.19** | **1.03** | −0.16 |
| **Max drawdown** | **−$338.55** | **−$645.65** | **−$307 (1.9× worse)** |
| t-stat | +0.50 | +0.10 | — |

**The 26 incremental signals in isolation:**

| | |
|---|---|
| Net | **−$135.20** · EV **−$5.20** · PF **0.78** |
| Winners / losers | 12 / 14 (WR 46.2%) |
| t-stat | **−0.55** |

Enabling full ETH **cuts net P&L 76%** and **nearly doubles drawdown**, while
adding 60% more trades.

### Deliverable 7 — family breakdown (RTH vs ETH)

| Family | Session | n | Net | EV | WR | PF |
|---|---|---|---|---|---|---|
| EMA21_PB_LONG | RTH | 24 | −$60.80 | −$2.53 | 41.7% | 0.89 |
| **EMA21_PB_LONG** | **ETH** | 14 | **−$133.80** | −$9.56 | 42.9% | **0.60** |
| **EMA21_PB_SHORT** | **RTH** | 19 | **+$237.70** | +$12.51 | 57.9% | **1.60** |
| EMA21_PB_SHORT | ETH | 12 | −$1.40 | −$0.12 | 50.0% | 1.00 |

**EMA21_PB_LONG overnight is the entire damage** (−$133.80, PF 0.60). EMA21_PB_SHORT
overnight is flat (PF 1.00). Note EMA21_PB_SHORT is the *best* RTH family (PF 1.60)
yet contributes nothing overnight — the same setup, different session, different result.

### Deliverable 6 — hour-of-day (full_eth, ET)

| ET hour | n | Net | EV | WR | PF | t |
|---|---|---|---|---|---|---|
| **00:00** | 11 | **+$139.55** | +$12.69 | 72.7% | 2.17 | +1.07 |
| 01:00 | 1 | −$80.95 | — | 0% | 0.00 | — |
| 02:00 | 3 | −$3.35 | −$1.12 | 33.3% | 0.96 | −0.03 |
| 03:00 | 2 | −$59.40 | −$29.70 | 0% | 0.00 | −2.83 |
| 04:00 | 2 | +$46.60 | +$23.30 | 50% | 6.07 | +0.72 |
| 05:00 | 3 | −$32.60 | −$10.87 | 33.3% | 0.64 | −0.32 |
| 07:00 | 1 | −$52.20 | — | 0% | 0.00 | — |
| 19:00 | 1 | −$87.95 | — | 0% | 0.00 | — |
| 22:00 | 1 | +$39.55 | — | 100% | ∞ | — |
| 23:00 | 1 | −$44.45 | — | 0% | 0.00 | — |
| *(RTH ref)* 10:00 | 14 | +$169.95 | +$12.14 | 57.1% | 1.68 | +0.82 |
| *(RTH ref)* 13:00 | 7 | −$215.65 | −$30.81 | 14.3% | 0.30 | −1.46 |

**Session mapping:** Asia (19:00–03:00 ET) = net negative overall; Europe open
(03:00–05:00 ET) = negative; US pre-market (07:00–09:00 ET) = 1 trade, negative;
evening reopen (18:00–20:00 ET) = 1 trade, negative.

The single positive overnight hour is **00:00 ET (+$139.55, PF 2.17)** — but
**t=+1.07 (not significant)** and it is one hour selected from ten. Nine of ten
overnight hours have n ≤ 3.

---

## 3. Deliverable 8 — false-signal analysis (why overnight underperforms)

Not speculation — measured on all 3,859 bars:

| Feature | RTH median | ETH median | ETH/RTH |
|---|---|---|---|
| ATR(14) | 12.14 pt | 7.65 pt | **0.63×** |
| Bar range | 12.50 pt | 6.25 pt | **0.50×** |
| **Volume** | 22,977 | 3,105 | **0.14×** |
| \|log return\| | 6.64 bp | 3.63 bp | 0.55× |

**Mechanical cause — cost drag:**

| | Round-trip cost | Median ATR | Cost as % of ATR |
|---|---|---|---|
| RTH | $4.20 (0.84 pt) | 12.14 pt | **6.9%** |
| ETH | $4.20 (0.84 pt) | 7.65 pt | **11.0%** |

**Overnight costs 1.59× more relative to available range.** The strategy's targets
are ATR-scaled, so overnight targets shrink ~37% while slippage and the $1.70 fee
stay fixed. Combined with **86% lower volume** (fill quality in the replay is
*assumed* at 1 tick — in reality overnight slippage would likely be worse, making
these results **optimistic**).

Ranking the causes by evidence: **(1) reduced volatility/range vs fixed costs**
(1.59× drag, measured), **(2) low liquidity** (0.14× volume, measured — and a
likely source of *additional* unmodeled slippage), **(3) weak trends** (|return|
0.55×). Spread was **not** directly measurable from OHLCV bars — stated as a
limitation, not asserted.

---

## 4. Deliverable 9 — Recommendation

# KEEP RTH-ONLY (retain the current allowlist)

Based solely on measured results:

1. Enabling full ETH **reduced net P&L 76%** (+$176.90 → +$41.70).
2. It **nearly doubled max drawdown** (−$338.55 → −$645.65) — the worst outcome, since it buys more risk for less return.
3. The 26 incremental signals are **net negative** (−$135.20, PF 0.78, EV −$5.20).
4. The mechanism is **structural and measured**, not a fluke: 1.59× cost drag on 0.63× ATR with 0.14× volume. This is a persistent property of the overnight session, not a regime accident.
5. Modelled fills are **optimistic** overnight (1 tick assumed on 14%-volume bars), so the true result is likely worse than shown.

**Not "enable selected families overnight" either.** The tempting cut —
EMA21_PB_SHORT (PF 1.00) or hour 00:00 ET (PF 2.17) — is exactly the
survivorship/curve-fitting trap: those are the best of 2 families × 10 hours after
seeing results, on n=12 and n=11, at t≈0–1.07. Selecting them post-hoc would repeat
the documented history in this config (63 dated post-hoc tweaks; families disabled
on 0/5-week samples).

### Honest statistical caveat (stated, not buried)

The incremental effect has **t = −0.55**. That is **not significant harm** — it is
*directionally* negative with n=26 over 2 months. Strictly: this study does **not
prove** overnight signals are harmful; it shows they are **mildly negative, add
substantial drawdown, and provide no measurable benefit.** The decision rests on
*absence of demonstrated benefit* plus a *measured structural cost mechanism* —
not on a significant negative result.

**Net: the original JUL 4 2026 suppression decision is confirmed by clean IB ETH
data. It should stay, and it should stay for the reason measured here (cost drag),
which is more durable than the original backtest's P&L argument.**

---

## 5. Recommended next research (no production changes)

1. **Do not enable full ETH.** Do not carve out families or hours from this sample.
2. If the question is revisited, the honest test is **forward** and pre-registered: log both variants in parallel (the harness supports it) until the incremental set reaches **n ≥ 100**, then judge. At ~13/month that is ~8 months.
3. **Model overnight slippage properly** before any future ETH study — 1 tick on 3,105-contract bars is optimistic and biases every ETH result upward.
4. Context: six prior audits established MES 15m is a random walk on real IB data (VR ≈0.93, Hurst ≈0.55). Both variants here sit at PF 1.03–1.19 with t ≤ 0.50 — **neither shows edge.** This study answers "does removing suppression help?" (no), not "does the strategy work?" (still unproven).

---

## Assumptions & limitations

- 2 months / 52 sessions is the maximum 15m ETH history IB served in one request; results are a 2-month sample, not a multi-year study.
- Fill model: next-bar open + 1 tick each side. Overnight fills are likely worse in reality → ETH results here are **optimistic**.
- Spread/DOM not available from OHLCV; liquidity inferred from volume.
- Intrabar stop-vs-target ties resolved as stop-first (conservative, applied equally to both variants).
- Replay uses 15m bar high/low for touch detection; 1-min granularity would be marginally more precise but was applied identically to both variants, so the A/B comparison is unaffected.
- `TREND_CONT_LONG` appeared in live overnight blocks but not in this replay's incremental set — its per-session counter caps were consumed differently in the sandboxed replay. Noted, not adjusted.

---

## Caveman verdict

No new code needed. Empty list already = full ETH. Config, not code.

Ran frozen engine twice, same bars, one change. RTH output identical both runs —
proof only the night gate moved.

Open the night: **+26 signal, −$135.20, PF 0.78.** Total P&L fall from +$177 to
+$42 — lose three quarter. Drawdown grow from −$339 to −$646 — nearly double. More
trade, less money, more pain.

Why? Night market half the size, same toll. ATR 0.63×, range 0.50×, volume 0.14×,
but slippage and fee do not shrink. Cost eat **11% of ATR at night vs 6.9% day** —
1.59× drag. And my fill model is *kind* to night (1 tick on thin bar) — real night
worse than this.

One hour look good (00:00 ET, PF 2.17) and one family flat (EMA21_PB_SHORT). Do not
grab them. That is picking best of twenty after seeing answer — same sin that made
120-knob config.

**KEEP RTH-ONLY.** Suppression still justified. Not because night proven poison
(t=−0.55, not significant) but because night show **no benefit** and **real
measured cost**. Do not pay more toll for less road.
