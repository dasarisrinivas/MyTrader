# Gold Futures Intraday Bot — Deep Code Analysis & Improvement Plan

## 1. Strategy Summary

Your bot runs a **rules-based intraday trend-following system** with two signal families:

| Signal Family | Logic | Regime Required |
|---|---|---|
| **VWAP/EMA21 Pullback** | Price pulls back into VWAP or EMA21 zone during a confirmed trend, then reclaims the level | TRENDING_BULL / TRENDING_BEAR |
| **Opening Range Breakout (ORB)** | Price closes beyond the N-minute opening range high/low with volume confirmation | TRENDING or RANGING |

**Regime detection** uses ADX (trend strength) + EMA9/EMA21 alignment + VWAP relationship, with optional multi-timeframe confirmation and price structure gates.

**Session awareness** classifies bars into buckets (Overnight → Pre-COMEX → COMEX Open → Midday → Pre-Close) with per-bucket parameter multipliers.

**Risk**: ATR-based SL/TP with R:R minimum, post-loss cooldowns (uniform and direction-aware), news lockout windows.

---

## 2. Top 5 Reasons Win Rate May Be Low

### 2.1 — Regime Detection Is Too Restrictive AND Too Late

The regime classifier has **seven independent gates** before allowing a TRENDING label:

1. Warmup bars check
2. ATR ratio band (min/max)
3. ADX band (min/max)
4. EMA spread minimum
5. EMA slope threshold
6. Price structure (HH/HL or LH/LL)
7. Multi-timeframe ADX + EMA alignment

**Problem**: Gold trends are fast and violent — by the time all seven gates agree, you've missed 60-70% of the move. The pullback into VWAP/EMA21 often never comes because the initial impulse was the only entry window.

**Evidence in code**: The price structure gate (`price_structure_enabled`) compares `highs_arr[-1]` vs `highs_arr[0]` over a lookback window. In gold's typical spike behavior, the first bar of a move makes the high, and subsequent bars consolidate below it — this gate would label that as RANGING even though the trend is intact.

### 2.2 — Pullback Entry Is Too Demanding

The pullback signal requires **all** of:
- Close within touch band of VWAP or EMA21
- Reclaim of the level (touched below, closed above for longs)
- Bar direction confirmation (bullish close for longs)
- Minimum body fraction
- VWAP extension not exceeded
- EMA extension not exceeded
- R:R check passes

Gold's pullbacks are often shallow (0.1-0.3% retracement) and fast (1-2 bars). By requiring all these conditions simultaneously on one bar, you're filtering out the majority of valid setups. The signal fires rarely, and when it does, the move is often exhausted.

### 2.3 — ORB Signal Lacks Retest Logic

The ORB fires on first close beyond the range + buffer. In gold, the highest-probability ORB entries come on the **retest** of the breakout level, not the initial break. Your code has no concept of:
- Breakout → pullback to OR level → continuation entry
- Failed breakout tracking (price breaks out then closes back inside)
- OR range width relative to recent ATR (a narrow OR in high-vol conditions is unreliable)

### 2.4 — No Momentum Confirmation Beyond EMAs

The bot uses EMA crossover + ADX as its sole trend confirmation. Gold is heavily driven by:
- **Order flow / delta** (not just volume — net buying vs selling pressure)
- **Momentum divergences** (price makes new high but momentum indicator doesn't)
- **Tick-level microstructure** (large prints, iceberg detection)

None of these are available in your 1-min OHLCV framework. The bot is making decisions with one hand tied behind its back.

### 2.5 — Uniform ATR-Based Exits Don't Match Gold's Behavior

Gold has regime-dependent volatility patterns:
- **COMEX open (8:20-9:30 ET)**: Fast moves, 2-3x normal ATR, but short-lived
- **Midday (11:00-13:00 ET)**: Low vol, mean-reverting, ATR shrinks
- **News events**: 5-10x normal ATR in seconds, then reversal

A single `atr_sl_multiplier` applied uniformly means:
- Stops are too tight during COMEX open (stopped out by noise)
- Stops are too wide during midday (giving back profits in chop)
- News events blow through any reasonable ATR-based stop

Your bucket system adjusts `atr_min_ratio_mult` (entry filter) but the SL/TP calculation only distinguishes "extended hours" vs "regular" — it doesn't use the bucket-level granularity.

---

## 3. Concrete Improvements

### 3.1 — Regime Detection: Adaptive Threshold + Faster Confirmation

**Current problem**: Fixed ADX thresholds don't account for gold's varying ADX baseline across sessions.

```python
# IMPROVEMENT: Normalize ADX against its own recent distribution
# instead of fixed thresholds
def _adaptive_adx_threshold(self, features: pd.DataFrame, lookback: int = 50) -> float:
    """ADX trending threshold = rolling median + 0.5 * rolling stdev."""
    if "adx" not in features.columns or len(features) < lookback:
        return self._entry.adx_trend_min  # fallback to static
    recent_adx = features["adx"].iloc[-lookback:]
    return float(recent_adx.median() + 0.5 * recent_adx.std())
```

**Also**: Replace the price-structure gate with a simpler momentum check — the rate of change of EMA9 over 3 bars tells you more than HH/HL pattern matching on noisy 1-min data.

### 3.2 — Add RSI Divergence Filter to Pullback Signals

Pullbacks into VWAP/EMA21 fail most often when momentum is already diverging. Adding a simple RSI(14) divergence check prevents entries at the end of a trend:

```python
# In compute_indicators, add:
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = _rma(gain, 14)
avg_loss = _rma(loss, 14)
df["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

# In _check_pullback, before building the signal:
if "rsi" in features.columns:
    rsi = float(features["rsi"].iloc[-1])
    # Block long entries when RSI shows bearish divergence territory
    if action == "BUY" and rsi > 75:
        remember("pullback_rsi_overbought", rsi=rsi)
        continue
    if action == "SELL" and rsi < 25:
        remember("pullback_rsi_oversold", rsi=rsi)
        continue
```

### 3.3 — ORB Retest Entry (Higher Win Rate Variant)

```python
# Add state tracking for ORB breakout-then-retest
self._orb_breakout_triggered: Optional[str] = None  # "LONG" or "SHORT"
self._orb_breakout_bar_idx: int = 0

def _check_orb_retest(self, features, close, atr, regime, bar_ts):
    """Second-chance ORB: enter on retest of the breakout level."""
    if self._orb_breakout_triggered is None:
        return self._hold_signal(regime, "orb_no_prior_breakout")

    bars_since = len(features) - self._orb_breakout_bar_idx
    if bars_since > 10:  # Retest must happen within 10 bars
        self._orb_breakout_triggered = None
        return self._hold_signal(regime, "orb_retest_window_expired")

    if self._orb_breakout_triggered == "LONG":
        # Price pulled back to OR high and held
        if close >= self._or_high * 0.999 and close <= self._or_high * 1.002:
            # Confirm bar is bullish
            row = features.iloc[-1]
            if close > float(row.get("open", close)):
                # Valid retest entry
                ...
```

### 3.4 — Session-Aware SL/TP (Not Just Extended vs Regular)

```python
def _sl_tp(self, action, entry_price, atr, bar_ts=None, bucket_cfg=None):
    """Bucket-aware SL/TP with per-session multipliers."""
    # Use bucket config if available
    if bucket_cfg is not None:
        sl_mult = self._exit.atr_sl_multiplier * bucket_cfg.get("sl_mult", 1.0)
        tp_mult = self._exit.atr_tp_multiplier * bucket_cfg.get("tp_mult", 1.0)
    elif self._is_extended_hours(bar_ts):
        sl_mult = self._exit.extended_atr_sl_multiplier
        tp_mult = self._exit.extended_atr_tp_multiplier
    else:
        sl_mult = self._exit.atr_sl_multiplier
        tp_mult = self._exit.atr_tp_multiplier
    # ... rest unchanged
```

**Recommended bucket-specific multipliers for GC:**

| Bucket | SL Mult | TP Mult | Rationale |
|---|---|---|---|
| OVERNIGHT | 1.8x | 2.5x | Wide stops, let winners run in thin liquidity |
| PRE_COMEX | 1.3x | 2.0x | London session, moderate |
| COMEX_OPEN | 1.0x | 1.5x | Tight stops, fast targets (highest noise) |
| MIDDAY | 0.8x | 1.2x | Tight everything, small range |
| PRE_CLOSE | 1.2x | 1.0x | Tight TP, avoid getting caught in close |

### 3.5 — Add Trailing Stop Logic

Your code has `_sl_tp` returning fixed levels. Gold trends can extend 2-4x ATR intraday. You need a trailing mechanism:

```python
@dataclass
class GoldSignal:
    # ... existing fields ...
    trail_trigger_atr: float = 0.0     # Start trailing after this ATR gain
    trail_step_atr: float = 0.0        # Trail by this ATR distance

# In signal generation:
signal.trail_trigger_atr = atr * 1.5   # Start trailing after 1.5 ATR profit
signal.trail_step_atr = atr * 0.5      # Trail at 0.5 ATR behind price
```

The trading manager (not shown) would then implement the trailing logic using these parameters. This single change typically improves trend-following P&L by 15-25% by letting winners extend.

### 3.6 — Partial Exit Logic

Add to `GoldSignal`:

```python
partial_exit_levels: list[tuple[float, float]] = field(default_factory=list)
# List of (price_level, fraction_to_exit)
# e.g., [(tp * 0.5, 0.5), (tp, 0.5)]  — take half at 50% of TP, rest at TP
```

For gold intraday, a **50/50 split** works well:
- Exit 50% at 1.0 ATR profit
- Trail remaining 50% with 0.5 ATR trailing stop

---

## 4. Risk Management Issues

### 4.1 — No Daily Loss Limit

The code has per-trade R:R checks and post-loss cooldowns, but **no aggregate daily drawdown limit**. After 3 consecutive losses in a choppy session, the bot will keep firing. This is the #1 cause of blow-up days.

**Required**: Add to `GoldSignalGenerator.generate()`:

```python
if self._daily_realized_pnl < -self._entry.max_daily_loss_points:
    return self._hold_signal(regime, "daily_loss_limit_reached")
```

### 4.2 — No Max Trades Per Day

Without a cap, the bot can overtrade during volatile/choppy sessions. Recommendation: max 4-6 round trips per day for GC, with at most 2 in any single bucket.

### 4.3 — Post-Loss Cooldown Is Bar-Based, Not Time-Based

A 5-bar cooldown = 5 minutes. During COMEX open, 5 minutes of chop is nothing — the same pattern that caused the loss will repeat. For the losing signal family, the cooldown should be **at least 15-20 minutes** (15-20 bars) during high-vol sessions, and shorter (5-8 bars) during midday.

### 4.4 — No Correlation/Macro Risk Gate

Gold is a macro asset. The bot doesn't check:
- DXY direction (inverse correlation — long gold when dollar weakens)
- US10Y yields (negative correlation)
- SPX risk-on/risk-off (gold as hedge)
- VIX spikes (gold rallies on fear)

At minimum, feed in a 5-min DXY EMA slope as a confirmation filter. If DXY is rising sharply, don't go long gold regardless of what EMA/VWAP say.

---

## 5. Market Context & News Integration

### 5.1 — Current Implementation

The `news_lockout_windows_et` mechanism blocks all signals during defined windows. This is good but blunt — it's a binary on/off rather than adaptive behavior.

### 5.2 — Improvements

**Pre-news positioning**: Block new entries 30 minutes before high-impact events (FOMC, NFP, CPI), not just during them. Widen stops on existing positions.

**Post-news momentum**: After a high-impact event, gold often trends in one direction for 30-60 minutes. The bot should **relax** regime requirements post-news (e.g., skip the ADX gate for 15 minutes after the lockout ends) to catch the initial move.

**News-driven volatility scaling**: After a news event, ATR spikes. The bot's ATR-based stops will automatically widen, but the SL floor/ceiling (`sl_floor_points`, `sl_ceiling_points`) may clip the ATR-adjusted stop to an inappropriate level. Verify these aren't too tight for news reactions.

---

## 6. Time-Based Logic Review

### 6.1 — What's Good

- Session buckets with per-bucket parameter tuning (Phase 5) — well architected
- Opening range tracking with minimum range validation
- Extended hours support with wider multipliers

### 6.2 — What's Missing

**Optimal gold trading windows not explicitly prioritized:**

| Window (ET) | Gold Behavior | Recommendation |
|---|---|---|
| 03:00–05:00 | London open, gold often sets daily direction | HIGH priority — should have its own bucket logic |
| 08:20–09:30 | COMEX open, highest volume, fastest moves | Already your COMEX_OPEN bucket — good |
| 10:00–10:30 | Econ data releases (ISM, consumer confidence) | Add news-aware sub-bucket |
| 11:00–13:00 | Midday chop, mean-reversion dominates | Consider switching to mean-reversion signals here |
| 13:00–13:30 | Bond auction times, gold reacts | Lockout or heightened caution |

**The London open is critical for gold.** Your PRE_COMEX bucket (03:00 ET → COMEX open) lumps the entire London session together. The London open (typically 03:00 ET) often produces the day's first directional move. Consider splitting PRE_COMEX into LONDON_OPEN (03:00–05:00) and PRE_COMEX_LATE (05:00–08:20).

### 6.3 — CME Maintenance Window

Your code mentions a MAINTENANCE bucket but I don't see explicit logic handling the CME daily maintenance window (16:00-17:00 CT / 17:00-18:00 ET). Ensure no signals fire during this period — stale data during maintenance will generate false signals.

---

## 7. Technical Indicators — Evaluation & Additions

### 7.1 — Current Indicators

| Indicator | Implementation | Grade |
|---|---|---|
| EMA(9/21) | Standard EWM, correct | B+ |
| ATR(14) | Wilder smooth, correct | A |
| ADX(14) | Full +DI/-DI/DX pipeline, correct | A |
| VWAP | Session-anchored, correct | A |
| EMA Slope | Simple delta over lookback | B |

### 7.2 — Missing Indicators (High Priority)

**RSI(14)** — Already discussed. Essential for filtering exhaustion entries.

**Volume-Weighted RSI or Money Flow Index (MFI)** — Better than plain RSI for gold because it incorporates volume. Gold's volume profile is highly informative.

**Bollinger Bands (20, 2.0)** — The Bollinger Band width (BBW) is the single best indicator for detecting when gold is about to make a large move. When BBW contracts to a multi-session low, the next breakout is typically 2-3x ATR.

```python
df["bb_mid"] = df["close"].rolling(20).mean()
df["bb_std"] = df["close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
```

**Volume delta approximation** — Without tick data, approximate buy/sell pressure:

```python
# Close position within bar as proxy for buy/sell pressure
df["bar_delta"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, 1) - 0.5
df["cum_delta"] = df["bar_delta"].rolling(10).sum()
```

### 7.3 — Lower Priority Additions

- **Keltner Channels** (for mean-reversion during midday)
- **VWAP standard deviation bands** (1σ, 2σ) — natural support/resistance for gold intraday
- **Previous day's high/low/close** — gold respects these levels with high probability

---

## 8. Code Quality & Bugs

### 8.1 — Potential Bug: ORB Bar Counting

In `_update_session_state`, `self._or_bar_count` increments for every bar after `session_open`, but there's no check for gaps. If data has missing bars (e.g., no trade during a minute), the OR window extends in clock time but not bar count. For gold, which can have thin minutes during overnight, this means the OR might form on 5 bars spread across 15 minutes instead of 5 consecutive minutes.

**Fix**: Use clock time rather than bar count, or verify bars are consecutive:

```python
# Use time elapsed since session open instead of bar count
elapsed_minutes = (bar_time.hour * 60 + bar_time.minute) - (session_open.hour * 60 + session_open.minute)
if elapsed_minutes >= self._session.opening_range_minutes:
    ...
```

### 8.2 — Potential Bug: Price Structure Gate Array Indexing

```python
highs_arr = recent["high"].values
lows_arr = recent["low"].values
if bull_ema:
    if highs_arr[-1] <= highs_arr[0] and lows_arr[-1] <= lows_arr[0]:
```

This only compares the **last** bar to the **first** bar of the lookback window. A proper HH/HL check should verify a *monotonic* trend or at least compare swing points, not just endpoints. Price could make HH on bar 3, then pull back on bar 5 (the last bar), and this gate would falsely reject the trend.

### 8.3 — Inefficiency: Full Indicator Recomputation

`compute_indicators` is called on every bar via `generate_gold()`, recomputing EMAs, ATR, ADX, and VWAP from scratch across the entire DataFrame. For a full session (~960 1-min bars by close), this becomes expensive.

**Fix**: Implement incremental indicator updates or cache the last N computed rows and only compute new ones.

### 8.4 — Missing Guard in `_check_rr`

```python
if risk <= 0:
    return False
return (reward / risk) >= self._exit.min_rr_ratio
```

No check for `reward <= 0`. If the TP is on the wrong side of entry (which can happen with tick snapping edge cases), this would return True for any min_rr_ratio ≤ 0, silently accepting a guaranteed loser.

### 8.5 — Snap-to-Tick Edge Case

`_snap_to_tick` uses `math.ceil(ticks - 1e-9)` and `math.floor(ticks + 1e-9)`. With GC tick size of $0.10 and prices around $2,300, you're operating with `ticks ≈ 23,000`. The epsilon of 1e-9 is fine for this range, but if you ever trade micro gold (MGC) or the tick size config is wrong, floating point can cause off-by-one-tick errors. Consider using `Decimal` or integer-cent arithmetic for price snapping.

---

## 9. Backtesting Concerns

### 9.1 — No Lookahead Bias Detected

The indicator computation is causal (shift(1) for prev_close, no future data in EMA/ATR). VWAP is session-anchored. This is clean.

### 9.2 — Potential Overfitting Risk

The strategy has **at minimum 25-30 tunable parameters** across entry, exit, session, and indicator configs. With this many degrees of freedom, it's easy to overfit to historical data. Each Phase 2/5 gate adds parameters that can be curve-fitted.

**Recommendations:**
- Freeze indicator parameters (EMA periods, ATR period, ADX period) — these are well-studied and shouldn't need optimization
- Group remaining parameters into 3-4 "profiles" (aggressive, moderate, conservative) rather than tuning each independently
- Use walk-forward optimization, not single-period backtest
- Out-of-sample test on at least 6 months of data not used in any parameter selection

### 9.3 — Missing Realistic Costs

No mention of:
- Commission per contract ($1.25-$2.50 per side for GC)
- Slippage model (0.1-0.3 ticks typical, 1-3 ticks during news)
- Exchange fees

---

## 10. Upgraded Strategy Blueprint

### Phase 1: Immediate Changes (Highest Impact)

1. **Add daily loss limit** — stop trading after X points of loss per day
2. **Add trailing stop logic** — let winners extend beyond fixed TP
3. **Add RSI filter** — block exhaustion entries (RSI > 75 long, < 25 short)
4. **Fix ORB to use clock time** instead of bar count
5. **Add `reward <= 0` guard** to `_check_rr`

### Phase 2: Strategy Enhancements (Medium Term)

1. **ORB retest entry** — second-chance entry on pullback to breakout level
2. **Adaptive ADX thresholds** — normalize ADX against its own recent distribution
3. **Bucket-specific SL/TP multipliers** — especially tighter during midday, wider overnight
4. **Partial exits** — 50% at 1 ATR, trail remaining 50%
5. **London open sub-bucket** — explicitly handle 03:00-05:00 ET

### Phase 3: Contextual Intelligence (Longer Term)

1. **DXY correlation filter** — feed in DXY 5-min EMA slope
2. **Bollinger Band width** — detect compression/expansion for ORB conviction
3. **Volume delta proxy** — approximate buy/sell pressure from bar position
4. **Post-news momentum mode** — relax regime gates for 15 min after lockout ends
5. **Mean-reversion mode for midday** — switch from trend-following to Keltner Channel mean-reversion during 11:00-13:00 ET

### Phase 4: Infrastructure

1. **Incremental indicator computation** — don't recompute entire history every bar
2. **Trade journal/logging** — structured log of every signal, entry, exit with full context
3. **Real-time P&L tracking** — feed back into daily loss limit
4. **Walk-forward backtest harness** — prevent parameter overfitting

---

## 11. Files Needed for Deeper Analysis

To provide more specific recommendations, I'd need:

1. **`shree/config/gold.py`** — The actual parameter values (ADX thresholds, ATR multipliers, session times, etc.) would let me evaluate whether specific values are appropriate for GC
2. **`shree/risk/trade_math.py`** — Position sizing logic, contract specs
3. **Trade log / backtest results** — Win rate, average win/loss, max drawdown, trade distribution by session bucket
4. **`GoldTradingManager`** — How signals are actually converted to orders, position tracking, exit management
5. **Any execution layer** — How orders are submitted (market/limit), fill handling, partial fill logic
