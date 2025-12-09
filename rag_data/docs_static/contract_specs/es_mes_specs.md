# E-mini S&P 500 Contract Specifications

## ES - E-mini S&P 500 Futures

### Basic Specifications
| Attribute | Value |
|-----------|-------|
| Symbol | ES |
| Exchange | CME Globex |
| Contract Size | $50 × S&P 500 Index |
| Tick Size | 0.25 index points |
| Tick Value | $12.50 |
| Point Value | $50.00 |

### Margin Requirements
- **Initial Margin:** ~$15,400 (varies by broker/volatility)
- **Maintenance Margin:** ~$14,000
- **Day Trade Margin:** ~$500-1,000 (broker dependent)

### Trading Hours
- **Globex:** Sunday 6:00 PM - Friday 5:00 PM ET
- **Daily Maintenance:** 5:00 PM - 6:00 PM ET (no trading)
- **RTH:** 9:30 AM - 4:00 PM ET

### Contract Months
- March (H), June (M), September (U), December (Z)
- Front month is most liquid

---

## MES - Micro E-mini S&P 500 Futures

### Basic Specifications
| Attribute | Value |
|-----------|-------|
| Symbol | MES |
| Exchange | CME Globex |
| Contract Size | $5 × S&P 500 Index |
| Tick Size | 0.25 index points |
| Tick Value | $1.25 |
| Point Value | $5.00 |

### Margin Requirements
- **Initial Margin:** ~$1,540 (1/10th of ES)
- **Maintenance Margin:** ~$1,400
- **Day Trade Margin:** ~$50-100 (broker dependent)

### Key Differences from ES
- 1/10th the size of ES
- Same price movement
- Lower margin requirements
- Ideal for smaller accounts
- More contracts for same exposure

---

## Risk Calculations

### Position Sizing Formula
```
Position Size = (Account Risk %) / (Stop Loss Points × Point Value)

Example:
- Account: $10,000
- Risk: 1% = $100
- Stop Loss: 2 points
- For MES: $100 / (2 × $5) = 10 contracts
- For ES: $100 / (2 × $50) = 1 contract
```

### P&L Calculations
```
P&L = (Exit Price - Entry Price) × Contracts × Point Value

Example (ES):
- Entry: 5900.00
- Exit: 5902.50
- Contracts: 1
- P&L = 2.5 × 1 × $50 = $125

Example (MES):
- Entry: 5900.00
- Exit: 5902.50
- Contracts: 10
- P&L = 2.5 × 10 × $5 = $125
```

### ATR-Based Stop Calculation
```
Stop Loss = Entry Price - (ATR × Multiplier)

Example:
- Entry: 5900.00
- ATR: 8.5 points
- Multiplier: 1.5
- Stop Loss = 5900 - (8.5 × 1.5) = 5887.25
```

---

## Rollover Information

### Quarterly Rollover
- Occurs 8 days before contract expiration
- Volume shifts to next contract
- Rollover dates: March, June, September, December

### Rollover Impact
- Slight price difference between contracts
- Spread may widen during rollover
- Best to roll before expiration week

---

## Liquidity Considerations

### Most Liquid Times
1. US Market Open (9:30-10:00 AM ET)
2. US Close (3:30-4:00 PM ET)
3. European Overlap (8:00-11:30 AM ET)

### Low Liquidity Times
1. Asian Session (7 PM - 3 AM ET)
2. US Lunch (12:00-2:00 PM ET)
3. Holidays

---
*This document is static reference material for the RAG system.*
