# Backtest Analysis Report: ES Scoring Strategy
**Date:** February 6, 2026  
**Period:** Feb 1, 2025 – Jan 30, 2026 (RTH only)  
**Symbol:** ES (E-mini S&P 500)  
**Strategy:** `MesOneMinuteScoringStrategy` (1m bars, scoring-based entry)  
**Data:** 398K bars from IB (`ES_1m_combined.parquet`)

---

## 1. Executive Summary

**The strategy is unprofitable and not viable for live trading.** After fixing three critical bugs discovered during analysis, the 1-year backtest produces:

| Metric | Value |
|--------|-------|
| **Total P&L** | **-$5,209** (-10.42%) |
| Sharpe Ratio | -2.44 |
| Profit Factor | 0.85 |
| Win Rate | 34.1% |
| Max Drawdown | -10.78% ($5,392) |
| Total Trades | 1,552 |
| Avg Win / Avg Loss | $52.23 / -$31.73 (ratio 1.65) |
| Green Days | 93 / 256 (36.3%) |
| Green Weeks | 10 / 52 (19.2%) |
| Max Consecutive Losses | 15 |

**Breakeven win rate at 1.65 W/L ratio = 37.8%. Actual = 34.1%. The strategy consistently falls short by ~4 percentage points.**

---

## 2. Bugs Found & Fixed

### Bug 1 (CRITICAL): Direction Inversion — 20.7% of Trades Inverted
**File:** `shree/strategies/scoring_entry.py`, lines 669-679  
**Root cause:** The HTF (15-minute) direction override ran AFTER scoring components were computed. When 1m EMAs said SHORT but 15m regime said UPTREND, the code flipped the direction to LONG — but the score components (EMA_STACK_DOWN, EMA_SLOPE_DOWN, BELOW_VWAP) had already been computed for the SHORT direction. This created incoherent signals: SHORT-quality components used to justify LONG trades. 334 out of 1,614 trades (20.7%) were direction-inverted.

**Fix:** Replaced the direction override with a -30 point HTF_CONFLICT penalty. When 1m and 15m timeframes disagree, the trade is filtered out by the score threshold instead of being force-flipped. Post-fix: 0% direction-inverted trades.

### Bug 2 (SIGNIFICANT): `datetime.now()` in Backtest
**File:** `shree/strategies/scoring_integration.py`, line 171  
**Root cause:** The lunch/evening session threshold adjustment used `datetime.now()` (wall clock) instead of the backtest's simulated timestamp. Running the backtest at 10 PM would apply evening thresholds (full=65, half=50) to ALL trades regardless of their simulated time, incorrectly filtering out valid RTH signals.

**Fix:** Use the `current_time` parameter already passed through the evaluation chain.

### Bug 3 (MODERATE): Hardcoded Parameter Overrides
**File:** `backtest/run.py`, lines 490-491  
**Root cause:** `stop_atr_multiplier` was hardcoded to 3.0 and `take_profit_multiple` to 1.0, clobbering the config file values (1.5 and 2.0 respectively). This meant the backtest always ran with 1:1 R:R and very wide 3x ATR stops, regardless of config.

**Fix:** Only apply defaults if the config doesn't specify values.

---

## 3. Performance Deep Dive

### 3.1 Hourly Breakdown (CST)
```
Hour   PnL         Trades  Avg PnL   Win Rate
09:00  -$1,460.65    406   -$3.60    35.7%    ← Opening volatility, worst hour
10:00  +$  100.65    494   +$0.20    36.4%    ← ONLY profitable hour
11:00  -$1,334.80    277   -$4.82    34.3%    ← Pre-lunch chop
12:00  -$   34.90      1   -$34.90    0.0%    ← Lunch (mostly filtered)
13:00  -$  844.70    153   -$5.52    32.0%    ← Post-lunch fade
14:00  -$1,242.75    210   -$5.92    27.1%    ← Worst avg PnL
15:00  -$   15.15     11   -$1.38    27.3%    ← End of day
```

**Only 10:00 AM is profitable.** All other hours lose money. The 2:00-3:00 PM window is particularly poor (27.1% win rate).

### 3.2 Score Band Analysis
```
Score    PnL         Trades  Avg PnL   Win Rate
[45,50)  +$ 174.85    211   +$0.83    38.9%    ← Only profitable band!
[50,55)  -$ 930.40    171   -$5.44    33.3%
[55,60)  -$  81.85    144   -$0.57    37.5%
[60,65)  -$ 525.45    183   -$2.87    37.2%
[65,70)  -$ 418.90    211   -$1.99    34.1%
[70,80)  -$1,313.35   329   -$3.99    30.1%
[80,100) -$1,737.20   303   -$5.73    32.0%    ← HIGHEST scores = worst PnL
```

**The scoring system is inversely correlated with profitability.** The highest-scoring trades (80+) have the worst results (-$5.73 avg, 32.0% WR). Only the lowest band (45-50, half-size) is profitable. This is a fundamental indictment: the scoring system does not identify good trades.

### 3.3 ATR Regime Analysis
```
ATR      PnL         Trades  Avg PnL   Win Rate
<2       -$1,212.90    321   -$3.78    34.0%    ← Low vol, death by cuts
2-4      -$2,595.30    772   -$3.36    33.7%    ← Most trades, most losses
4-6      -$1,443.90    336   -$4.30    33.3%
6-8      +$  636.65     79   +$8.06    41.8%    ← ONLY profitable band
8+       -$  216.85     44   -$4.93    34.1%
```

**The strategy only works in elevated volatility (ATR 6-8 pts).** 88% of trades occur in ATR < 6 where the strategy is consistently unprofitable. The 1.5x ATR stops in low-vol environments are still too wide relative to signal quality.

### 3.4 Exit Analysis
- **Stop losses:** 1,013 trades (65.3%), total loss -$32,391
- **Take profits:** 506 trades (32.6%), total gain +$27,182
- **Other:** 33 trades (2.1%), gain +$377

The 2:1 take-profit is being hit far less often than the stop. For 2:1 R:R to work, you need ~33% win rate minimum, and ~38% to overcome commissions/slippage. The strategy sits in the danger zone at 34.1%.

---

## 4. Streak & Risk Analysis

| Metric | Value |
|--------|-------|
| Max consecutive losses | 15 |
| Streaks of 5+ losses | 57 |
| Streaks of 8+ losses | 16 |
| Streaks of 10+ losses | 8 |
| Max drawdown duration | 897 days* |
| Worst day | -$232.40 |
| Worst week | -$561.25 |

*The drawdown is essentially permanent — the strategy never recovers.

With $50K capital and these loss streaks, a 15-trade losing streak at avg -$32/trade = -$480. Not catastrophic individually, but the strategy never generates enough winning streaks to recover.

---

## 5. Fundamental Problems

### 5.1 The Scoring System Has No Predictive Edge
The most damning finding is that **higher scores predict worse outcomes**. Trades scored 80+ (maximum confidence) have the worst PnL and lowest win rate. This means the features being scored — EMA stack alignment, EMA slope, VWAP position, ADX strength, candle body ratio — do not reliably predict short-term ES price direction on a 1-minute timeframe.

### 5.2 1-Minute Noise Dominates Signal
ES 1-minute bars are extremely noisy. The EMA(9)/EMA(21) crossover that determines direction flips constantly. The scoring system compounds this by layering multiple noisy indicators on top of a noisy direction signal. More indicators ≠ better signal; they just increase false confidence.

### 5.3 No True Mean Reversion or Structural Edge
The strategy is a trend-following approach on 1-minute data, which is inherently mean-reverting. It's buying breakouts in a timeframe that overwhelmingly reverts. The 1.5x ATR stop gets hit before the 3.0x ATR target in the majority of cases because 1-minute trends simply don't extend that far.

### 5.4 Transaction Costs Matter
At 1,552 trades over 256 days (6.1/day), slippage and commission consume significant edge. Even with only $2.40/trade commission plus 1 tick slippage: 1,552 × ($2.40 + $12.50) = ~$23K in friction. Any marginal edge is obliterated.

---

## 6. What Would Actually Fix This

1. **Abandon 1-minute trend following** — The timeframe is too noisy. Either move to 15m/30m for trend following, or switch to a mean-reversion approach on 1m.

2. **Dramatically reduce trade frequency** — 6 trades/day with 34% win rate is a guaranteed loss. 1-2 high-conviction trades based on structural levels (prior day high/low, VWAP, opening range) would be more viable.

3. **Rebuild the scoring system from outcomes, not intuition** — The current scoring weights were set manually. Use the 1,552 trades of data to actually measure which features predict profitable exits, and weight accordingly.

4. **Add a market microstructure edge** — Order flow, bid/ask imbalance, volume delta. Pure price-indicator approaches on 1m ES are well-arbitraged by HFT firms.

5. **Fix the time-of-day filter** — Only the 10 AM hour is profitable. Consider trading only 9:45-11:00 and perhaps the 13:00-13:30 window (after lunch, before afternoon fade).

---

## 7. Files Modified

| File | Change |
|------|--------|
| `shree/strategies/scoring_entry.py` | HTF direction override → HTF_CONFLICT -30pt penalty |
| `shree/strategies/scoring_integration.py` | `datetime.now()` → `current_time` parameter |
| `backtest/run.py` | Hardcoded overrides → respect config file values |

---

## 8. Verdict

**NOT VIABLE FOR LIVE TRADING.** Three significant bugs were found and fixed, improving results from -$6,385 to -$5,209, but the strategy remains fundamentally unprofitable. The scoring system is inversely correlated with trade quality — its highest-confidence signals produce its worst trades. No amount of parameter tuning can fix this; the approach needs architectural redesign.
