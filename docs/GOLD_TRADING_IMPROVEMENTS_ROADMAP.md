# Gold Trading Improvements Roadmap

## Purpose

This roadmap turns the current Gold bot review into an ordered implementation plan.
It focuses on improving **entry quality**, **anti-chase behavior**, **regime accuracy**, and **safe rollout** without overcomplicating the strategy too early.

Primary code paths:
- `shree/strategies/gold/strategy.py`
- `shree/strategies/gold/signals.py`
- `shree/strategies/gold/regime.py`
- `shree/execution/gold/risk.py`
- `shree/execution/gold/manager.py`
- `shree/config/gold.py`

---

## Current strengths

The current Gold strategy already has a solid base:
- clean separation between regime, signals, risk, and execution
- causal indicator computation
- ATR-based SL/TP model
- basic MTF support
- session gating and overnight-specific exit widening
- daily loss limits, cooldowns, and hard contract caps
- decent focused test coverage for signals, risk, rollover, extended hours, and strategy wiring

The next gains are most likely to come from **better filtering**, not more indicators.

---

## Guiding principles

1. **Prioritize entry quality over signal quantity**
2. **Use the trade journal and runtime logs to validate every threshold change**
3. **Prefer small reversible changes with focused tests**
4. **Roll out one signal-family improvement at a time**
5. **Keep RTH and overnight behavior intentionally distinct**

---

## Improvement themes

### 1. Better pullback confirmation
Current pullback logic is too permissive. It mostly accepts:
- correct regime
- close near VWAP or EMA21
- close on the correct side of the level

Needed improvement:
- require actual rejection / recovery behavior
- use candle body quality and wick behavior
- optionally require break of previous bar high/low or bar color confirmation

### 2. Stronger ORB filtering
Current ORB is based on close crossing OR high/low.

Needed improvement:
- breakout threshold beyond the OR line
- volume confirmation
- no-chase protection
- limit late/overextended breakout entries

### 3. Stronger regime classification
Current regime logic is mostly:
- ADX in range
- EMA9 vs EMA21 ordering
- price vs VWAP

Needed improvement:
- EMA slope requirement
- minimum EMA spread
- optionally short-term market structure confirmation

### 4. Explicit anti-chase logic
Add protection when price is too stretched from:
- VWAP
- EMA21
- OR line

### 5. Smarter exit and cooldown behavior
Later improvements should make:
- time stops progress-aware
- cooldowns context-aware instead of purely bar-count based

---

## Phase 1 — High-value, low-risk entry quality upgrades

### Goal
Reduce weak entries without changing overall architecture.

### Scope
#### A. Pullback rejection confirmation
Files:
- `shree/strategies/gold/signals.py`
- `shree/config/gold.py`

Add configurable checks such as:
- long: bar low tags/pierces level zone, then closes back above level
- short: mirrored version
- minimum candle body fraction
- optional bullish/bearish close confirmation
- optional prior-bar break confirmation

Suggested config additions:
- `pullback_reclaim_required: true`
- `pullback_min_body_fraction: 0.35`
- `pullback_confirm_break_prior_bar: false`
- `pullback_max_wick_fraction: 0.65`

#### B. ORB breakout threshold + no-chase filter
Files:
- `shree/strategies/gold/signals.py`
- `shree/config/gold.py`

Add:
- minimum breakout distance beyond `or_high` / `or_low`
- max breakout candle size vs ATR
- max extension from OR after breakout

Suggested config additions:
- `orb_breakout_min_atr_fraction: 0.10`
- `orb_max_breakout_candle_atr: 1.25`
- `orb_max_extension_atr: 0.75`

#### C. Volume confirmation for ORB
Files:
- `shree/strategies/gold/signals.py`
- `shree/config/gold.py`

Add:
- compare breakout bar volume to rolling median / mean

Suggested config additions:
- `orb_volume_lookback_bars: 20`
- `orb_volume_min_multiple: 1.20`

### Required tests
- new pullback rejection tests in `tests/gold/test_gold_signals.py`
- ORB threshold and no-chase tests in `tests/gold/test_gold_signals.py`
- volume confirmation tests in `tests/gold/test_gold_signals.py`

### Validation
- compare count of signals before/after over recent sample sessions
- verify fewer low-quality breakouts and weak pullback entries in logs

### Success criteria
- fewer entries immediately after stretched candles
- fewer signals where price only barely crosses a level
- no regression in existing gold signal and strategy tests

---

## Phase 2 — Regime quality improvements

### Goal
Improve market state labeling so entries only fire in stronger contexts.

### Scope
#### A. EMA slope and spread filters
Files:
- `shree/strategies/gold/regime.py`
- `shree/config/gold.py`

Add:
- minimum EMA spread between `ema9` and `ema21`
- slope lookback for `ema9` and optionally `ema21`

Suggested config additions:
- `ema_spread_min_ratio: 0.00015`
- `ema_slope_lookback_bars: 3`
- `ema_slope_required: true`

#### B. Optional price-structure confirmation
Files:
- `shree/strategies/gold/regime.py`

Examples:
- bullish: recent higher low / higher high structure
- bearish: recent lower high / lower low structure

Keep this optional and configurable.

#### C. Regime subtype refinement
Optional later addition:
- `TRENDING_BULL_EXTENDED`
- `TRENDING_BEAR_EXTENDED`
- `TRENDING_BULL_WEAK`
- `TRENDING_BEAR_WEAK`

This should only be done if Phase 1 and 2A prove useful and manageable.

### Required tests
- extend `tests/gold/test_gold_strategy.py`
- extend `tests/gold/test_gold_mtf.py`
- add regime-specific cases for weak vs strong trend conditions

### Validation
- inspect examples previously labeled trending but visually flat
- confirm they reclassify to `RANGING` or weaker states

### Success criteria
- fewer trend labels during sideways drift
- better alignment between regime and actual continuation behavior

---

## Phase 3 — Anti-chase and stretched-market protection

### Goal
Block entries taken too far from fair value or after overextended moves.

### Scope
Files:
- `shree/strategies/gold/signals.py`
- `shree/config/gold.py`

Add signal-type-aware extension guards:
- max distance from VWAP in ATRs
- max distance from EMA21 in ATRs
- stricter threshold for ORB than pullbacks
- optional overnight-specific stricter values

Suggested config additions:
- `pullback_max_vwap_extension_atr: 1.0`
- `pullback_max_ema_extension_atr: 0.8`
- `orb_max_vwap_extension_atr: 1.5`
- `extended_hours_extension_strictness_mult: 0.8`

### Required tests
- add anti-chase tests in `tests/gold/test_gold_signals.py`
- add extended-hours anti-chase tests in `tests/gold/test_gold_extended_hours.py`

### Validation
- inspect losing entries that occurred after already-large directional bars
- ensure they are blocked after change

### Success criteria
- fewer entries at local exhaustion points
- improved average excursion after entry

---

## Phase 4 — Smarter time-stop and trade management

### Goal
Exit dead trades earlier without interfering with valid runners.

### Scope
#### A. Progress-aware time stop
Files:
- `shree/execution/gold/manager.py`
- `shree/config/gold.py`

Replace flat timeout-only logic with staged checks such as:
- after N bars, require at least `0.25R` unrealized progress
- after M bars, require break-even or better
- otherwise exit

Suggested config additions:
- `time_stop_stage_1_bars: 20`
- `time_stop_stage_1_min_progress_r: 0.25`
- `time_stop_stage_2_bars: 40`
- `time_stop_stage_2_min_progress_r: 0.0`

#### B. Direction-aware cooldowns
Files:
- `shree/strategies/gold/signals.py`

Instead of blocking all entries equally after loss:
- block same signal family first
- optionally block same direction only
- allow opposite-side valid regimes if needed

### Required tests
- add manager/logic tests for staged time-stop behavior
- add cooldown granularity tests

### Validation
- compare average hold time on losers before/after
- verify no increase in premature exits on winners

### Success criteria
- dead trades exit earlier
- same-pattern revenge entries reduce

---

## Phase 5 — Session-aware specialization

### Goal
Tune signal behavior to Gold’s different intraday personalities.

### Scope
Files:
- `shree/config/gold.py`
- `shree/strategies/gold/strategy.py`
- `shree/strategies/gold/signals.py`

Introduce named session buckets:
- Asia / overnight
- London / pre-COMEX
- COMEX open
- midday
- pre-close

Potential behavior changes:
- ORB active only in COMEX open window
- pullbacks allowed in London and COMEX
- stricter volume and ATR thresholds overnight
- reduced aggressiveness in midday chop

### Required tests
- session routing tests
- signal enable/disable tests per session bucket

### Validation
- journal analysis by hour bucket
- compare PnL contribution by session segment

### Success criteria
- fewer trades in historically weak time windows
- better signal-family/session fit

---

## Phase 6 — Data-driven calibration and score refinement

### Goal
Use real trade outcomes to tune thresholds instead of relying on intuition alone.

### Scope
Use:
- `data/gold_journal/`
- `gold_trades` / `gold_daily_summary` in journal DB
- runtime logs in `logs/gold_trading.log`

Recommended analysis slices:
- signal family vs P&L
- hour-of-day vs P&L
- ATR bucket vs outcomes
- ADX bucket vs outcomes
- OR width vs ORB outcome
- stretched-entry distance vs win rate
- time-to-target and time-to-stop

Possible follow-up changes:
- stronger confidence model
- per-signal-family thresholds
- session-specific overrides

### Success criteria
- threshold changes justified by measured outcomes
- fewer intuition-only tweaks

---

## Testing roadmap

### Immediate test additions
- `tests/gold/test_gold_signals.py`
  - pullback rejection
  - breakout threshold
  - volume confirmation
  - anti-chase blocks
- `tests/gold/test_gold_strategy.py`
  - stronger regime gating behavior
- `tests/gold/test_gold_extended_hours.py`
  - overnight-specific stricter filters

### Later test additions
- manager/time-stop behavior tests
- session-bucket routing tests
- DB/journal-driven analysis helpers if scripts are added

---

## Rollout plan

### Step 1
Implement only Phase 1A + 1B:
- pullback rejection confirmation
- ORB breakout threshold/no-chase

### Step 2
Run focused tests and compare signal counts.

### Step 3
Paper-run and inspect:
- idle diagnostics
- blocked reasons
- entry signal quality in `logs/gold_trading.log`

### Step 4
If stable, implement Phase 1C and Phase 2A.

### Step 5
Only then move into time-stop and session specialization.

---

## Recommended next action

If only one improvement batch is implemented next, do this:

1. **Pullback rejection confirmation**
2. **ORB breakout threshold + volume/no-chase filters**

That combination should improve entry quality materially without requiring architectural changes.

---

## Definition of done for the roadmap

A phase is complete only when all are true:
- code change implemented
- focused tests added and passing
- runtime logs show clear block/allow reasons
- signal count and trade behavior reviewed from paper/sim output
- no unrelated MES behavior changed
