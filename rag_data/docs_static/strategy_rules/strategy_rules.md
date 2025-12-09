# Trading Strategy Rules

## Core Trading Philosophy

Our trading approach combines three layers of intelligence:
1. **Rule Engine:** Hard filters and signal generation
2. **RAG Retrieval:** Historical context and pattern matching
3. **LLM Decision:** Final judgment with reasoning

---

## Layer 1: Rule Engine Filters

### Hard Filters (Blockers)
These filters MUST pass before any trade is considered:

1. **ATR Filter**
   - Minimum ATR: 0.3 points
   - Maximum ATR: 5.0 points
   - Purpose: Avoid dead or extremely volatile markets

2. **Cooldown Filter**
   - Minimum time between trades: 15 minutes
   - Purpose: Prevent overtrading after wins or revenge trading after losses

3. **Market Hours Filter**
   - Only trade during US market hours (9:30 AM - 4:00 PM ET)
   - Avoid first 15 minutes (opening volatility)
   - Avoid last 5 minutes (closing chaos)

4. **Volatility Regime Filter**
   - Block trades when ATR > 2x 20-day average
   - Warn when ATR < 0.5x average

### Signal Generation Rules

**BUY Signal Requirements:**
- Price > EMA9 > EMA20 (bullish trend alignment)
- MACD histogram > 0 or rising
- RSI < 70 (not overbought)
- Near support (PDL) or breakout above resistance

**SELL Signal Requirements:**
- Price < EMA9 < EMA20 (bearish trend alignment)
- MACD histogram < 0 or falling
- RSI > 30 (not oversold)
- Near resistance (PDH) or breakdown below support

---

## Layer 2: RAG Context Guidelines

### What RAG Retrieves
1. Similar historical trades (same setup, trend, volatility)
2. Relevant strategy documentation
3. Recent market summaries
4. Mistake notes from similar setups

### How RAG Influences Decisions
- **High win rate in similar setups (>60%):** Increases confidence
- **Low win rate in similar setups (<40%):** Decreases confidence or blocks
- **Relevant mistake notes found:** Extra caution applied

---

## Layer 3: LLM Decision Criteria

### Confidence Thresholds
- **80-100%:** Strong trade, full position size
- **60-79%:** Moderate confidence, consider reduced size
- **40-59%:** Weak signal, likely HOLD
- **0-39%:** Do not trade

### Position Sizing by Confidence
- 80%+ confidence: 100% position (1.0x)
- 70-79%: 75% position (0.75x)
- 60-69%: 50% position (0.5x)
- <60%: No trade

---

## Risk Management Rules

### Per-Trade Risk
- Maximum risk per trade: 1% of account
- Maximum daily loss: 3% of account
- Emergency stop: 5% daily loss = stop trading

### Stop Loss Placement
- Default: 1.5 × ATR from entry
- Near support/resistance: Just beyond level
- Never move stop further from entry

### Take Profit Strategy
- Primary target: 2 × ATR (1.33:1 R:R)
- Scale out: 50% at 1.5 ATR, rest at 2.5 ATR
- Trailing stop after 1 ATR profit

---

## Trade Entry Rules

### DO Enter When:
1. Rule engine generates signal (score > 40)
2. Trend aligned with trade direction
3. LLM confidence > 60%
4. No similar recent losing trades in RAG

### DO NOT Enter When:
1. Any hard filter is blocking
2. Price in middle of range (no edge)
3. Against clear trend
4. Multiple warning filters triggered
5. After 3 consecutive losses (emotional reset needed)

---

## Exit Rules

### Take Profit Exit
- Exit when price hits take profit level
- Log as TP_HIT

### Stop Loss Exit
- Exit when price hits stop loss level
- Log as SL_HIT
- Trigger cooldown period

### Trailing Stop Exit
- After 1 ATR profit, trail stop to breakeven
- After 1.5 ATR profit, trail at 0.5 ATR
- Log as TRAILING

### Manual Exit
- Exit if market conditions change dramatically
- Exit if news event occurs
- Log as MANUAL with reason

---

## Pattern-Specific Rules

### Breakout Trades
- Wait for close above/below level (not just touch)
- Volume should confirm (above average)
- Initial stop below/above breakout level
- Target: Next major level or 2x ATR

### Pullback Trades
- Wait for price to pull back to EMA or level
- Enter on reversal candle
- Stop below pullback low
- Target: Recent high/low or 1.5x ATR

### Reversal Trades
- Require divergence (price vs RSI/MACD)
- Must be at major support/resistance
- Use tighter stops (1 ATR)
- Scale into position

---

## Mistake Prevention Checklist

Before every trade, verify:
- [ ] Signal strength > 40?
- [ ] LLM confidence > 60%?
- [ ] All hard filters passed?
- [ ] Not trading against trend?
- [ ] Not near opposite level (buying near resistance)?
- [ ] Not in cooldown period?
- [ ] Daily loss limit not reached?
- [ ] Position size within limits?

---
*This document is static reference material for the RAG system.*
