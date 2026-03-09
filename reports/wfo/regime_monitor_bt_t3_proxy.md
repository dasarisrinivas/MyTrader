# Regime Performance Monitor — T4

**Source:** backtest_MES_2025-02-01_2026-01-31_20260309_142310_trades.csv  
**Total trades:** 49  
**ATR threshold (low/high-vol boundary):** 13.0  
**Rolling window:** last 20 trades

---

## Overall Performance

| Metric | Value |
|--------|-------|
| Total P&L | $+83.65 |
| Win rate | 44.9% |
| Profit factor | 1.11 |
| Expectancy | $+1.71/trade |
| Rolling 20-trade PF | 0.64 |

---

## Per-Regime Breakdown

### Low-Vol (ATR < 13.0)
- Trades: 49
- Total P&L: $+83.65
- Win rate: 44.9%
- Profit factor: 1.11
- Expectancy: $+1.71/trade

### High-Vol (ATR ≥ 13.0)
- Trades: 0
- Total P&L: $+0.00
- Win rate: 0.0%
- Profit factor: 0.00
- Expectancy: $+0.00/trade

---

## Monthly P&L

| Month | Trades | P&L | Profit Factor |
|-------|--------|-----|---------------|
| 2025-04 | 1 | $-5 | 0.00 |
| 2025-06 | 5 | $-40 | 0.64 |
| 2025-07 | 12 | $+299 | 5.81 |
| 2025-08 | 21 | $-27 | 0.92 |
| 2025-09 | 10 | $-144 | 0.44 |

---

## Rolling Profit Factor (last 60 evaluations)

```
  ▆▅▅▅▅▅█▇▆▆▆▅▅▄▄▃▃▂▂▂▃▃▂▂▂▂▂▁▁▁
  low=0.64  high=2.25  current=0.64
```

---

## Size Decision

**⚠️  REDUCE to 50%  (rolling PF=0.64 < 0.8)**

_Rules_:
- PF ≥ 1.2: full size
- 0.8 ≤ PF < 1.2: full size, monitor
- 0.6 ≤ PF < 0.8: 50% size
- PF < 0.6: 25% size + halt for review

---

> **T4 Finding:** With ~53 trades/year, rolling performance monitoring is more
> reliable than IS/OOS grid search (WFO). The grid search systematically
> selects aggressive parameters (ADX=15, tight band) that over-trade in OOS.
> Fixed params from `bt_t3_proximity.yaml` outperform any WFO output
> by +$947 (-$844 WFO vs +$103 baseline).