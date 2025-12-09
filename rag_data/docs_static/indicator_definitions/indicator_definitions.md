# Trading Indicator Definitions

## Trend Indicators

### EMA (Exponential Moving Average)
Gives more weight to recent prices than simple moving average.

**EMA9 (Fast):**
- Period: 9 bars
- Use: Short-term trend direction
- Signal: Price above = bullish, below = bearish

**EMA20 (Medium):**
- Period: 20 bars
- Use: Intermediate trend direction
- Signal: Key support/resistance level

**EMA50 (Slow):**
- Period: 50 bars
- Use: Major trend direction
- Signal: Trend filter - don't short above, don't long below

**EMA Stack:**
- Bullish: Price > EMA9 > EMA20 > EMA50
- Bearish: Price < EMA9 < EMA20 < EMA50
- Chop: EMAs tangled/crossing

---

## Momentum Indicators

### RSI (Relative Strength Index)
Measures speed and magnitude of price movements.

**Settings:**
- Period: 14 bars
- Range: 0-100

**Interpretation:**
- Overbought: RSI > 70 (potential reversal down)
- Oversold: RSI < 30 (potential reversal up)
- Neutral: RSI 30-70
- Bullish momentum: RSI > 50 and rising
- Bearish momentum: RSI < 50 and falling

**Divergence:**
- Bullish: Price makes lower low, RSI makes higher low
- Bearish: Price makes higher high, RSI makes lower high

---

### MACD (Moving Average Convergence Divergence)
Shows relationship between two moving averages.

**Components:**
- MACD Line: EMA12 - EMA26
- Signal Line: EMA9 of MACD Line
- Histogram: MACD Line - Signal Line

**Interpretation:**
- Bullish: MACD above signal line, histogram positive
- Bearish: MACD below signal line, histogram negative
- Momentum increasing: Histogram bars growing
- Momentum decreasing: Histogram bars shrinking

**Crossovers:**
- Bullish cross: MACD crosses above signal line
- Bearish cross: MACD crosses below signal line

---

## Volatility Indicators

### ATR (Average True Range)
Measures market volatility using the greatest of:
- Current High - Current Low
- |Current High - Previous Close|
- |Current Low - Previous Close|

**Settings:**
- Period: 14 bars

**Use Cases:**
- Stop loss placement: Entry ± (1.5 × ATR)
- Position sizing: Smaller positions when ATR high
- Trade filtering: Skip trades in extreme ATR

**Interpretation:**
- High ATR: Market volatile, use wider stops
- Low ATR: Market quiet, may break out soon
- Rising ATR: Volatility increasing
- Falling ATR: Volatility decreasing

---

## Volume Indicators

### Volume Ratio
Current volume compared to average volume.

**Calculation:**
```
Volume Ratio = Current Volume / 20-bar Average Volume
```

**Interpretation:**
- Ratio > 1.5: High volume (breakout more valid)
- Ratio 0.7-1.5: Normal volume
- Ratio < 0.7: Low volume (less conviction)

---

## Support/Resistance Levels

### PDH/PDL (Previous Day High/Low)
Key reference levels from yesterday's session.

**PDH (Previous Day High):**
- Strong resistance level
- Breakout above = bullish
- Rejection = potential short

**PDL (Previous Day Low):**
- Strong support level
- Breakdown below = bearish
- Bounce = potential long

---

### Pivot Points
Calculated support/resistance levels.

**Formula:**
```
Pivot (P) = (PDH + PDL + PDC) / 3
R1 = (2 × P) - PDL
S1 = (2 × P) - PDH
R2 = P + (PDH - PDL)
S2 = P - (PDH - PDL)
```

**Use:**
- Price above pivot = bullish bias
- Price below pivot = bearish bias
- R1/R2 = resistance targets
- S1/S2 = support targets

---

## Combined Signal Scoring

### Signal Strength Calculation
```
Score = (Trend × 30) + (Momentum × 25) + (Level × 25) + (Volume × 20)

Where:
- Trend: 1 if aligned, 0.5 if neutral, 0 if against
- Momentum: RSI + MACD alignment (0-1)
- Level: Proximity to S/R (0-1)
- Volume: Volume confirmation (0-1)
```

### Minimum Thresholds
- Score < 30: No trade
- Score 30-50: Weak signal (reduce size)
- Score 50-70: Moderate signal (normal size)
- Score > 70: Strong signal (full size)

---
*This document is static reference material for the RAG system.*
