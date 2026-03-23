# Mar 23 2026 — Overnight Short Loss Forensic

## Trade examined

Latest losing live trade:

- **Entry time:** 2026-03-23 03:15 CST
- **Side:** SELL
- **Signal type:** `EMA21_PB_SHORT`
- **Fill:** `6512.75`
- **Stop:** `6524.50`
- **Target:** `6495.00`
- **Exit time:** 2026-03-23 03:54 CST
- **Exit fill:** `6524.75`
- **Result:** `-$60.00`

This was a valid overnight short that reversed upward and hit the stop.

## Why the bot took it

The setup passed all normal entry checks.

### Strategy signal

- `EMA21_PB_SHORT`
- ADX: `31`
- RSI: `44`
- MACD histogram: `-0.79`
- ATR: `9.1`
- Overnight-adjusted stop/target in effect

### Overlay context

- Combined sentiment: `-0.16`
	- technically bearish, but almost entirely **VIX-derived** rather than social-flow confirmation
	- Stocktwits: `+0.00 (60)`
	- Reddit: `+0.00 (0 posts found)`
	- Twitter: `+0.00 (0 samples)`
	- VIX: `-0.40`
	- practical takeaway: at `03:15` CST, the sentiment layer added very little real directional conviction beyond volatility regime context
- VX: `25.75`
	- pullback penalty applied: `-0.05`
- Hybrid pipeline: `SELL (61%)`
- Hybrid agreed with strategy
- Final confidence: `0.803`

### Risk / order flow

- `RiskGate pass`
- Margin acceptable
- Stop distance acceptable
- Order filled normally
- No contract-selection or session bug involved in the execution path

## Why it lost

The loss was caused by **post-entry reversal**, not a broken gate or order path.

Sequence:

1. Bot entered short at `6512.75`
2. Trade initially worked: by `03:30`, price reached `6509.50` and floating P&L was about `+$15.63`
3. Reversal then accelerated quickly: by `03:45`, price had bounced to `6517.25` and floating P&L had flipped to about `-$23.12`
4. Protective stop was working correctly at `6524.50`
5. Stop was hit at `03:54` for `6524.75`
6. Realized result was approximately `-12` points / `-$60`

This matters because it was **not** a trade that failed instantly. It behaved correctly at first, then reversed roughly `15+` points from the profitable peak in about `24` minutes. That profile fits the overnight-thin-liquidity / sharp-reversal narrative better than a slow grind against the position.

### Structural context near the stop

- Mar 23 Opening Range high: `6534.75`
- Mar 23 Opening Range low: `6523.25`
- Trade stop: `6524.50`

So the stop sat only `1.25` points above `OR_L`.

The hybrid breakdown logged `NO_LEVEL` because PDH and PDL were not nearby, but the Opening Range boundary was effectively sitting right at the stop. Any mean-reversion push back up to test that boundary was enough to stop the trade out. That structural proximity was not explicitly surfaced as a risk factor in the live decision path.

Conclusion: **valid trade, valid loss**.

## Comparison trade — winning short

Compared against the most recent strong winner:

- **Entry time:** 2026-03-20 11:15 CST
- **Side:** SELL
- **Signal type:** `EMA21_PB_SHORT`
- **Fill:** `6608.00`
- **Exit:** `6594.50`
- **PnL:** `+$67.50`

### Winner context

- ADX: `31`
- RSI: `41`
- MACD histogram: `-0.15`
- ATR: `9.9`
- Hybrid pipeline: `SELL (96%)`
- Final confidence: `0.800`
- Session: RTH / stronger daytime context

## Key comparison takeaways

At first glance, the loser does **not** look obviously weak on raw numbers:

- Losing trade confidence was actually slightly higher (`0.803` vs `0.800`)
- Loser had more bearish MACD than the winner
- Loser had nominally bearish sentiment support, but it was mostly a **VIX proxy**, not active social confirmation

So the raw signal metrics alone do **not** explain the difference.

### Biggest practical differences

#### 1. Session quality

The loser was **overnight**.

That means:

- thinner liquidity
- more reversal risk
- weaker continuation reliability
- more noise around otherwise valid pullback entries

The winner occurred in a stronger daytime environment.

#### 2. Hybrid certainty

- Losing trade: hybrid `SELL (61%)`
- Winning trade: hybrid `SELL (96%)`

That is the cleanest model-quality difference.

Both trades received hybrid agreement boosts, but the winner had **far stronger hybrid conviction**.

#### 3. Trade follow-through

- Loser: valid short, briefly profitable, then sharp overnight reversal into stop
- Winner: clean downside continuation into target

## Interpretive conclusion

This overnight loser was:

- technically valid
- operationally valid
- not a contract bug
- not a session bug
- not a risk-gate failure

But it was still a **moderate-conviction overnight short**, with sentiment driven mainly by VIX and a stop sitting near the Opening Range low. That combination is materially weaker than a strong RTH short with near-max hybrid confirmation.

## Suggested next steps

### Option 1 — Overnight hybrid threshold for A/D shorts

Require stronger hybrid confirmation for overnight `EMA21_PB_SHORT` / `EMA21_PB_LONG` style entries.

Example direction:

- if session is overnight
- and signal family is A / D style pullback
- and hybrid confidence is below a stronger cut such as `0.75`
- then either block or dampen further

This remains the most actionable refinement from this review.

### Option 2 — Overnight short quality filter

Add an additional quality rule for overnight shorts when the setup is valid but conviction is not elite.

Potential filter inputs:

- hybrid confidence band
- sentiment magnitude
- VX regime
- proximity to structural levels

### Option 3 — Review more overnight losers before hard-coding

Before changing rules, compare several examples:

- Mar 23 03:15 loser
- Mar 20 06:45 loser
- Mar 20 07:15 loser
- Mar 20 11:15 winner

Goal:

- confirm whether **overnight + moderate hybrid confidence** is the repeatable failure pattern

## Bottom line

This trade lost because a valid overnight short **worked briefly**, then reversed upward sharply after entry.

The clearest difference versus the winning comparison trade is not the core setup family, but the **context quality**:

- **overnight + moderate hybrid conviction** lost
- **RTH + very strong hybrid conviction** won

That reinforces this as a **market-context quality problem** more than a pure signal-quality failure.

Most promising refinement: **tighten overnight pullback entries when hybrid confidence is only moderate, especially below roughly 75% hybrid confidence**.
