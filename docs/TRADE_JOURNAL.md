# ShreeBot Trade Journal — Phase 1+2 Observation Period

> **Purpose:** Structured record of daily trading activity, blocked signals, near-misses,
> and hypothetical outcomes. Used to drive Phase 3+4 optimization decisions.
>
> **Period:** Mar 3, 2026 → Mar 17, 2026 (2-week Phase 1+2 review window)
>
> **Decision Gates:**
> - **Mar 10:** Touch-band near-misses ≥ 10? → Implement Fix #6 (ATR touch band)
> - **Mar 17:** 2-week review: trades/day < 2? → Start Fix #3 (day-type classifier)

---

## Weekly Summary

### Week 1: Mar 3–7, 2026

| Metric | Value | Notes |
|---|---|---|
| **Total Trades** | 1 | OR_BREAK_LONG on Mar 3 |
| **Wins / Losses** | 0W / 1L | |
| **P&L** | **-$39.80** | -$35 loss + $4.80 commission |
| **Signals Generated** | ~10 | Strategy produced BUY/SELL signals |
| **CHOP Blocks** | 8+ | All pullback signals killed |
| **Touch-Band Near-Misses (D)** | 20+ (Mar 5 alone) | Fix #6 candidate |
| **OR Gap-Through Misses (E)** | 2+ | Price gapped through OR, no clean cross |
| **F Diagnostic Bug** | Confirmed | F_short failures hidden by F_long diagnostic |

**Key Insight:** CHOP guard is now the #1 trade suppressor, replacing VX (which was fixed
in Phase 1+2). The hybrid RAG pipeline labels trend=CHOP even when ADX is 36–40 (clearly
trending), causing the Block-All Guard to kill valid setups.

---

## Daily Entries

---

### Mar 3 (Monday) — First Trade with Phase 1+2 Optimizations

**Market:** V-shaped recovery. Opened ~6768, sold off to 6718, rallied to 6845, faded to 6817.
**VX:** 20.95 (neutral range 16–22)

#### Actual Trades

| Time | Signal | Direction | Entry | SL | TP | Outcome | P&L |
|---|---|---|---|---|---|---|---|
| 13:45 | OR_BREAK_LONG (B) | BUY | 6846.75 | 6839.75 | 6853.75 | 🔴 SL hit @ 13:52 | -$39.80 |

**Confidence chain:** base 0.70 → sentiment +0.319 → VX neutral 0.0 → hybrid uncertain -0.05 → **final 0.720**

**Analysis:**
- OR High was 6841.75; entry at 6846.75 was 5 pts above (chasing the breakout)
- 6pt fixed SL with ATR=12.9 → only 0.47 ATR of room
- Hybrid flagged CHOP/UNCERTAIN — was a warning sign
- Price continued falling after stop-out (6845 → 6834 → 6817), so wider stop would NOT have saved this trade
- Post-trade: price never recovered to TP level

#### Blocked Signals
None — the only signal that fired was the OR breakout (exempt from CHOP guard).

#### Earlier Blocked Signal (13:30)
- **EMA9_PB_LONG (C)** at 13:30: ADX=30, conf=0.732 → **CHOP blocked** ✅ Correct block (price fell after)

**Lesson:** The CHOP guard correctly blocked the pullback but let the breakout through. The breakout failed due to late entry + tight stop + fading momentum.

---

### Mar 4 (Tuesday) — CHOP Guard Dominates

**Market:** Strong rally all day. OR Low=6780, OR High=6797.5. Price: 6780 → 6889 (+109 pts).
**VX:** 22.10 (elevated range 22–28, pullbacks get -0.05 VX adjustment)

#### Actual Trades
None.

#### Blocked Signals

| Time | Signal | ADX | Conf (after overlays) | Would-Have-Been |
|---|---|---|---|---|
| 08:15 | EMA9_PB_LONG (C) | 24 | 0.603 | 🟢 **TP hit** — price rallied 40+ pts from ~6840 |
| 08:30 | EMA21_PB_LONG (A) | 22 | 0.600 | 🟢 **TP hit** — same rally |
| 09:00 | EMA21_PB_LONG (A) | 21 | 0.600 | 🟢 **TP hit** — same rally |
| 14:45 | EMA21_PB_LONG (A) | 19 | 0.650 | ❓ Late in day, marginal |

**All blocked by CHOP Block-All Guard.** Hybrid said trend=CHOP on every cycle.

**Analysis:**
- Market was clearly trending (low-ADX grind-up, +109 pts on the day)
- The first 3 blocked signals at ~6840 area would have been easy 8pt TP winners
- CHOP guard cost an estimated **+$40 to +$120** in missed profits
- Hybrid RAG pipeline's trend classification lagged behind actual price action

**Lesson:** CHOP guard is too aggressive on quiet trend days. ADX was 19–24 (below typical
"trending" thresholds) but the price action was unmistakably directional.

---

### Mar 5 (Wednesday) — Waterfall Selloff, D Near-Misses

**Market:** Sharp selloff. Opened ~6870, dropped to 6784, recovered to 6839 in evening.
**ADX:** 14–22 all day (low despite strong move — ADX lags on fast moves)

#### Actual Trades
None.

#### Signal D (Short Pullback) Near-Misses — **20 bars**

Price never bounced back to EMA21 — the selloff was too steep. Signal D requires
`bar_high >= EMA21 * (1 - touch_pct)` and it failed on 20 consecutive bars:

| Time | Bar High | Touch Needed | Gap |
|---|---|---|---|
| 09:30 | 6830.2 | 6846.3 | -16.1 pts |
| 09:45 | 6835.2 | 6844.3 | -9.1 pts |
| 10:45 | 6827.5 | 6835.3 | -7.8 pts |
| **11:00** | **6833.0** | **6834.0** | **-1.0 pt** ← closest miss |
| 11:15 | 6827.0 | 6832.3 | -5.3 pts |
| 11:30 | 6809.2 | 6829.1 | -19.9 pts |
| 12:00 | 6786.0 | 6820.3 | -34.3 pts |

**Fix #6 (ATR touch band)** would have caught the 11:00 near-miss and several others.

#### Signal E (OR Breakdown) — Gap-Through Problem

OR Low = 6862. Price gapped through on the 09:30 bar (prev=6861.5, close=6823.8).
Since prev_close (6861.5) was already below OR Low (6862), E saw no "fresh cross."
The breakdown was missed because it happened too fast.

#### Signal F Diagnostic Bug Confirmed

F diagnostic only reports F_long stack failure — F_short failures are invisible in logs.
Actual F_short likely failed due to ADX < 25 (ADX was 14–22 all day).

**Lesson:** Fast-move blind spot — when trends are strong and swift, pullback signals
can't fire (no pullback) and F can't fire (ADX lagging). ATR touch band + E gap-through
logic would have captured trades here.

---

### Mar 6 (Thursday) — Short Signals Fired, CHOP Blocked Again

**Market:** Continued weakness. Opened ~6730, bounced to 6770 area, choppy.
**VX:** elevated

#### Actual Trades
None.

#### Blocked Signals

| Time | Signal | ADX | Conf | Entry | SL | TP | Hypothetical Outcome |
|---|---|---|---|---|---|---|---|
| 10:30 | EMA21_PB_SHORT (D) | **40** | 0.670 | 6762.50 | 6768.50 | 6754.50 | 🔴 **SL hit** — price rose to 6769.75 next bar |
| 11:00 | EMA21_PB_SHORT (D) | **36** | 0.670 | 6766.00 | 6772.00 | 6758.00 | ⏳ Pending — price at 6766 |

**Analysis:**
- ADX was 36–40 — clearly trending, yet hybrid still says trend=CHOP
- Signal #1 would have lost due to tight 6pt stop (ATR=12.5, stop = 0.48 ATR)
- Direction was correct (price returned to 6762–6766 range) but stop too tight
- CHOP guard accidentally saved us from signal #1's SL hit

**Lesson:** Even when CHOP guard is wrong about the trend, the 6pt fixed stop on D signals
is independently problematic. Two issues compounding: (1) CHOP misclassification blocking
valid trends, (2) tight stops that get hunted even when direction is right.

---

## Running Tallies (for Decision Gates)

### Touch-Band Near-Misses (for Mar 10 Gate: ≥10 → Implement Fix #6)

| Date | Signal | Count | Closest Miss |
|---|---|---|---|
| Mar 5 | D (short PB) | **20** | 1.0 pt at 11:00 |
| **Total** | | **20+** | **GATE MET ✅** |

### CHOP Guard Blocks (for Mar 17 Gate: trades/day < 2)

| Date | Blocks | Would-Have-Won | Would-Have-Lost |
|---|---|---|---|
| Mar 3 | 1 | 0 | 1 (price fell after) |
| Mar 4 | 4 | 3 | 0-1 |
| Mar 5 | 0 | — | — |
| Mar 6 | 2 | 0 | 1 (tight stop) |
| **Total** | **7** | **3** | **2** |

### Trades per Day (target: 2–3+)

| Date | Trades | Signals Generated | Blocked |
|---|---|---|---|
| Mar 3 | 1 | 2 | 1 |
| Mar 4 | 0 | 4 | 4 |
| Mar 5 | 0 | 0 | 0 |
| Mar 6 | 0 | 2 | 2 |
| **Avg** | **0.25/day** | | |

---

## Emerging Patterns & Recommended Actions

### 1. CHOP Guard Over-Blocking (CRITICAL)
- **Evidence:** 7 blocks in 4 days, 3 would have been winners
- **Root Cause:** Hybrid RAG pipeline labels trend=CHOP even with ADX 36–40
- **Options:**
  - (a) Enable CHOP exception (`ENABLE_CHOP_EXCEPTION=1`) — allows ADX≥25 + bias-aligned through
  - (b) Reduce CHOP guard scope — only block when ADX < 20 (truly choppy)
  - (c) Reclassify: trust strategy's ADX over hybrid's trend label
- **Priority:** HIGH — this is the #1 frequency killer

### 2. Fixed 6pt Stop Too Tight on A/B/D/E (HIGH)
- **Evidence:** Mar 3 OR breakout (-7pts, 0.47 ATR), Mar 6 D signal (SL at 0.48 ATR)
- **Direction was right** in Mar 6 D signal but stop hunted by noise
- **Fix:** Make A/B/D/E ATR-adaptive like C/F already are (min 6pt, max 20pt, 1.0x ATR)
- **Priority:** HIGH — even when signals fire, tight stops reduce win rate

### 3. Touch-Band Near-Misses (Mar 10 Gate: ALREADY MET)
- **Evidence:** 20+ near-misses on Mar 5 alone
- **Fix:** Fix #6 (ATR touch band) — allow pullback within ATR*0.15 of EMA21
- **Priority:** READY TO IMPLEMENT

### 4. OR Gap-Through (MEDIUM)
- **Evidence:** Mar 5 E signal missed because price gapped through OR Low
- **Fix:** Add "already below" continuation logic for E (if prev_close recently crossed)
- **Priority:** MEDIUM

### 5. F_short Diagnostic Blind Spot (LOW)
- **Evidence:** F diagnostic only shows F_long failure, F_short reasons invisible
- **Fix:** Add F_short diagnostic path in es_fifteen_min.py
- **Priority:** LOW (diagnostic only, doesn't affect trading)
