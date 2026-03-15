Now I have everything I need. Here is my full professional assessment.

---

### 1. Strategy Summary (Trader Perspective)

This is a **systematic short-volatility, theta-harvesting strategy** — specifically, a weekly credit spread program on SPY using defined-risk structures (bull put spreads, bear call spreads, and occasional iron condors).

**What it's doing:** Selling short-dated (5–7 DTE) SPY options at approximately the 25-delta strike, collecting premium, and banking on SPY *not* moving far enough to breach that strike before expiration. The structural edge is pure **theta decay** — the options are expected to expire worthless or be bought back at 50% of the credit received.

**What edge it claims:**
- Time decay (theta) on short-dated options accelerates into expiration
- A ~75% probability-of-profit starting point (selling ~0.25 delta = ~25% chance of expiring ITM)
- Regime-aware direction switching (puts in uptrends, calls in downtrends)
- 11-filter entry gating to avoid high-risk environments (FOMC, VIX spikes, large gaps)

**Honest trader read:** This is essentially the *"selling weekly premium on SPY"* strategy that tens of thousands of retail traders run in some form. The architecture around it — 11 filters, guards, PDT tracking, regime detection — is meaningfully more sophisticated than the average retail approach. But the core trade is vanilla: short 0.25-delta vertical spreads with defined risk.

---

### 2. Strengths / Pros

**Defined-risk structure is the right choice for this account size.** Selling naked options on a $5k account is account-blowing behavior. A $10-wide spread caps the worst case at $920 per trade. That is sound and non-negotiable for anyone operating at this scale. The builder clearly learned from a prior naked-option version (the 2020 crash comparison bears this out — -10.1% vs -23%).

**The entry filter stack is genuinely thoughtful.** Most retail bots have zero or one filter. This one gates on: event risk (FOMC, CPI, NFP), VIX floor *and* ceiling, IV rank relative to 52-week range, VIX spike detection, large-gap detection, trend direction, market regime (SMA200), support/resistance proximity, skew asymmetry, and PDT compliance. Each of these has real market logic behind it — especially the IV rank filter (don't sell cheap premium), the VIX spike guard (don't sell into accelerating volatility), and the skew filter (inverted skew = institutional hedging = something is wrong).

**Thursday force-close is non-negotiable discipline.** Holding weekly credit spreads through Friday expiration is the single most dangerous thing a retail options seller can do. Gamma explodes. A 1% move in SPY on Friday can take a spread from worthless to max loss in 30 minutes. Forcing closure at 3:45 PM Thursday is exactly correct. This alone separates this bot from most amateur implementations.

**The Theta/Delta ratio filter (|θ|/|Δ| ≥ 0.08) is a meaningful quality filter.** Most retail systems just target delta and call it done. Requiring that decay-per-unit-of-directional-exposure meet a minimum threshold ensures you're being paid adequately for the risk. This is subtle and correct.

**The 50% profit target is statistically appropriate.** Taking spreads off at 50% of max profit dramatically improves realized win rate and capital efficiency — you free up margin faster to redeploy, and you avoid the dangerous gamma risk that builds up in the final 2 DTE. This is standard best practice among professional premium sellers.

**Crash scenario stress tests were actually run and disclosed.** The builder did not hide the 2018 and 2022 results. The -43.4% and -39.1% figures are painful but honest. Most retail backtests conveniently exclude crash periods or don't test them at all.

**PDT compliance logic is correctly implemented.** A $5k account running into PDT violations would get locked out entirely. Capping at 3 round-trips in a rolling 5-day window is the right legal constraint. Most bots I've seen ignore this entirely.

---

### 3. Weaknesses / Risks

**The 83.6% backtest win rate is the biggest red flag in this document.** Selling 0.25-delta options has a theoretical probability of profit of approximately 75%. Getting 83.6% in backtests implies the system is either (a) successfully filtering out many losing environments via the 11 filters, (b) benefiting from the 50% profit target — since trades closed early count as wins — or (c) overfitted to the specific 1-year test window. Likely all three. The concern is that 67 trades is an extremely thin sample from which to assert 83.6% reliability. That is not statistical significance — that is one year of favorable data.

**The 1-year baseline is almost certainly a bull/low-vol regime.** Without knowing exactly which 12-month period was used for the "baseline," if it was 2023 or any similar low-VIX trending year, the backtest is essentially showing you what premium selling looks like in its *ideal* environment. The crash tests directly contradict the baseline confidence: -43.4% drawdown in 2018 Q4, -39.1% in 2022. A $5,000 account losing 43% is a $2,150 loss — from a bot that supposedly returned +27.81% per year. That asymmetry is severe.

**The slippage model is dangerously optimistic.** "1–2% of premium" on credit spreads is not realistic. At $0.80 net credit, 1% is less than a penny. In real execution, the bid/ask spread alone on the short leg of a SPY weekly option at 0.25 delta is typically $0.05–$0.15 wide. The long hedge leg adds another $0.03–$0.08. Realistic slippage on a combo order is far more likely $0.05–$0.12 round-trip, not $0.01. On a $0.80 credit, $0.10 in slippage is 12.5% of your take — not 1–2%. This inflates every backtest trade's profitability meaningfully.

**Black-Scholes with VIX as IV proxy is not how options actually behave.** The backtest's options simulator uses Black-Scholes to reconstruct option prices from historical VIX levels. This ignores: term structure (VIX is a 30-day measure, not a 7-day measure), IV skew across strikes (put IV ≠ call IV ≠ VIX/√52), the fact that short-dated weekly options price at significantly different implied volatility than the VIX implies, and bid/ask friction. The model systematically misprices short-dated OTM options — which is exactly what this strategy trades. The backtest Greeks and credit values are approximations, not market reality.

**The 2× loss stop creates a systematic risk problem.** Selling a spread for $0.80 and stopping out at $1.60 means you're risking $160 to make $40 (the 50% profit target = closing at $0.40, profit = $0.40 × 100 = $40 net of commissions). That is a **1:4 reward-to-risk ratio** on winners vs losers. To be profitable at that ratio, you need a win rate above 80% — which is why the 83.6% backtest figure is so load-bearing. If live win rate drops to 70% (still excellent by most standards), this strategy likely loses money. The math is uncomfortably dependent on the win rate being genuinely that high in live markets.

**Regime dependency is severe and disclosed but underweighted.** The 2022 bear market result (-39.1%) is not an edge case — it's an entire year of the dominant market regime being wrong for this strategy. The SMA200 filter switches to call spreads in a downtrend, but a grinding bear market with periodic violent rallies (classic 2022) repeatedly triggers stop-outs on short call spreads. The strategy has no answer for the scenario where every regime filter is correctly set and the market still systematically works against the position.

**Correlation risk with the MES bot is explicitly admitted and underweighted.** The README notes both bots are "short volatility" and "are not hedges." Running two short-vol strategies simultaneously from a $5k account (presumably total capital, not per-bot) means a VIX spike event hits *both* positions at once. The combined drawdown potential is not the sum of the individual risks — it's amplified because both strategies fail in the same tail scenario.

**5-minute monitoring creates real gap risk.** Selling weekly options on SPY with 5-minute monitoring loops is not adequate during volatile markets. SPY can move 0.5–1% in a single candle during a Fed press conference or a geopolitical shock. A position that's fine at :00 can be at max-loss territory by :05 with no opportunity to intervene. The Thursday upgrade to 60-second monitoring is appropriate but only solves part of the problem.

---

### 4. Backtest Quality Review

**Sample size: Inadequate.** 67 trades over one year is statistically insufficient to distinguish edge from luck. A 95% confidence interval on an 83.6% win rate from 67 samples spans roughly 73%–91%. The lower bound of 73% is below the break-even threshold for a 1:4 reward/risk ratio. You cannot confirm the edge exists from this data alone. A robust backtest for this strategy needs at minimum 5 years, ideally 10+, spanning multiple volatility regimes.

**The Sharpe of 1.28 is plausible but not meaningful in isolation.** For a strategy that takes 67 trades a year, a single bad month can devastate the Sharpe ratio. More importantly, Sharpe ratios on option-selling strategies are structurally misleading — they look excellent during calm markets and then print a massive left-tail event that the ratio never warned you about. The crash tests are doing more useful analytical work than the Sharpe ratio is.

**The -13.0% max drawdown in the baseline is unrealistically low.** Given that the same strategy produced -46.2% and -40.1% drawdowns in stress tests, a -13% baseline tells you the test window avoided regime stress entirely. This is not a realistic drawdown estimate for live trading.

**Transaction costs: Insufficiently modeled.** At $2.60 commission per spread (per the README), commissions are captured. But the slippage model (1–2% of premium) is, as noted above, unrealistically favorable. A more honest model would use $0.05–$0.10 per spread in slippage, which would reduce the average net P&L per trade from $20.75 meaningfully — possibly by 15–25%.

**The win rate of 83.6% is not independently validated.** There is no out-of-sample test, walk-forward analysis, or cross-validation. The entire 1-year period is in-sample. Even if the filters were not manually tweaked to this period, the fact that 11 filters are operating simultaneously creates substantial overfitting risk simply through combinatorial selection — each additional filter has a chance of being inadvertently tuned to the test window.

**2020 COVID result of -10.1% is the most credible number in the document.** The spread structure legitimately does what it claims — the max loss cap is real, and it shows. This result is believable and speaks well of the defined-risk approach.

---

### 5. Options-Specific Concerns

**Delta targeting at 0.25 is reasonable but mechanically applied.** In practice, delta varies significantly intraday. A strike that is 0.25 delta at 9:35 AM may be 0.30 delta by 11:00 AM if SPY moves. The bot scans once at entry and doesn't re-evaluate. This is acceptable given the weekly timeframe, but it means some entries will have materially different actual delta than intended.

**VIX as IV proxy completely ignores the volatility term structure.** SPY 7-DTE options frequently trade at a significant premium *or* discount to the 30-day VIX, depending on term structure. In contango environments (normal), short-dated IV is often 15–20% lower than VIX. In backwardation (crisis), short-dated IV explodes well above VIX. Using VIX/√52 to estimate weekly IV can be off by 30%+ in either direction. The backtest strike selection may be systematically choosing strikes that are actually much closer to or further from the money than modeled.

**$10-wide spreads on a $560+ SPY are notably narrow.** A $10 spread on a $560 stock is less than 2% of the underlying. At 5–7 DTE with 0.25-delta positioning, the long leg at $10 lower is providing relatively limited gamma protection in a fast-moving market. A meaningful intraday SPY move can rapidly close the gap between short and long strikes, particularly on Thursdays.

**Theta/Delta filter (≥ 0.08) is a proxy measure.** The ratio doesn't account for vega exposure — which matters enormously for premium sellers. A spread can have an excellent theta/delta ratio and simultaneously have terrible vega exposure, meaning an IV expansion will hurt the position far more than theta is earning. There is no vega filter in the entry system.

**No IV crush exploitation logic.** Selling premium before anticipated IV crush events (immediately after earnings or major data releases) is a known, distinct edge. This strategy *avoids* those dates (FOMC, CPI, NFP blocked), which reduces risk but also avoids one of the most reliable premium-selling opportunities available. This is a conservative choice that limits upside.

**Assignment risk mitigation relies entirely on bot reliability.** The Thursday 3:45 PM force-close is the entire defense against assignment risk. If the bot crashes on Thursday morning (explicitly acknowledged as a risk), and the user doesn't respond to a Telegram alert quickly, the position can expire ITM. Assignment risk on ETF options is not theoretical — it happens, and the cleanup (short stock positions, margin calls) can be painful on a $5k account.

**No adjustment logic.** When a spread moves against the position, the only response is to stop out. There is no mechanism to roll the spread (close current week, reopen next week at better strikes), convert to a wider structure, or add a hedge. Professional premium sellers spend significant time on adjustment logic — it is often the difference between a 65% and an 80% realized win rate.

---

### 6. Real Market Execution Issues

**The BAG/combo order fill rate on IBKR for SPY options is not 100%.** IBKR's combo orders are frequently rejected by market makers or only partially filled, particularly in fast markets. The limit-then-retry-$0.01 logic is the right approach, but at 3 retries before cancellation, the bot will skip a meaningful percentage of otherwise valid trade setups. In a low-VIX environment where the credit barely clears $0.60, there is virtually no room to improve price — you're essentially accepting mid or slightly worse.

**Atomic fill of both legs is not guaranteed despite being a BAG order.** IBKR routes combo orders atomically *in theory*, but there are documented cases where partial fills occur, particularly during volatile conditions or when one leg has low volume. If the bot receives a partial fill and the position isn't properly tracked, leg risk is introduced — you could end up holding a naked short option, which on a $5k account is catastrophic.

**5-minute polling is inadequate for active risk management.** In professional short-option operations, monitoring is continuous or near-continuous, not polled. The gap between monitoring cycles creates real execution risk: a 2× loss stop condition might be triggered, remain undetected for up to 5 minutes, and by the time the close order is submitted the spread may be worth 3× or 4× the original credit. This is not a coding problem — it's a fundamental architectural limitation of polling-based risk management for short options.

**IBKR connection reliability in live trading is non-trivial.** The exponential backoff reconnect logic is mentioned but not detailed. IBKR TWS has known stability issues with API connections, particularly after market hours and around system restarts. A bot that loses connection during an active position and takes 30+ minutes to reconnect during a fast market move can suffer full max loss with no intervention.

**Slippage on Thursday close orders will be systematically worse than modeled.** The bot force-closes all positions at 3:45 PM Thursday — which means it is *always a motivated seller*. Market makers know that weekly options sellers are closing positions Thursday afternoon. The bid/ask spread on options near expiration widens significantly, and the bot's limit orders may not fill at mid. Persistent Thursday closing slippage is a real drag that the backtest does not model.

**PDT limits cap the learning period severely.** At 3 round-trips per 5-day window, this strategy takes roughly 67 trades per year. Building statistical confidence in live performance (say, 100 trades) takes approximately 18 months. During that entire period, the trader is operating in a regime where they cannot definitively know if the live strategy matches the backtest.

---

### 7. Improvements / Next Steps

**Add a vega filter to the entry system.** No vega budget means you're flying blind on volatility exposure. A simple rule: skip entry if the spread's net vega implies more than $X loss per 1-point VIX increase. Given that the short leg is negative vega (IV expansion hurts), this is a real and unmanaged risk.

**Replace VIX-as-IV-proxy with actual option chain IV.** When scanning the live chain, the bot already has access to real bid/ask prices and Greeks via IBKR. Use the actual implied volatility from the option chain rather than VIX/√52. This would make strike selection, theta/delta calculations, and risk estimates significantly more accurate in both the backtest simulation and live execution.

**Build rolling adjustment logic.** Define a trigger (e.g., short leg delta > 0.40, not yet at stop) that rolls the spread forward by one week rather than simply stopping out. This would reduce realized loss frequency dramatically and is standard practice for professional premium sellers. A roll captures more premium, moves the strike further OTM, and gives the position more time to recover.

**Implement real-time monitoring via IBKR's event-driven API (callbacks), not polling.** Replace the 5-minute polling loop with price callbacks and position-delta callbacks from ib_insync. This allows immediate reaction to stop conditions rather than accepting up to 5-minute exposure gaps. On Thursdays especially, continuous monitoring is the only responsible approach.

**Increase the backtest scope to 10+ years, multiple walk-forward windows.** Run the strategy on 2013–2024 data and validate out-of-sample using walk-forward testing (e.g., train on 2013–2017, test on 2018–2019, retrain on 2013–2019, test on 2020, etc.). The results will almost certainly be more sobering and more realistic than the 1-year baseline.

**Model slippage realistically.** Use actual bid/ask width data from historical option chains (available via databases like CBOE datashop, OptionMetrics, or even manually sampled IBKR data) rather than 1–2% of premium. The strategy's edge depends on the net credit being sufficiently above the floor — accurate slippage modeling will show whether the floor of $0.60 net credit is actually achievable after realistic fills.

**Add a SPX/0DTE volatility early-warning layer.** SPX same-day expiration (0DTE) order flow increasingly drives intraday SPY moves. A simple check on 0DTE call/put ratio or unusual SPX options activity before entry could serve as a last-line filter for days when gamma exposure from the broader market is unusually high and a sharp move is more likely.

**Decouple the MES and SPY bots with a portfolio-level risk limit.** Both bots should share a combined daily loss limit, not just individual limits. A $5k account with two short-volatility strategies hitting simultaneous drawdowns has no meaningful buffer. Consider adding a simple long VIX call hedge (buy one 30-delta VIX call per month) as a portfolio-level tail-risk offset.

**Track live Greeks in the position monitor, not just price-based stops.** The delta stop at 0.50 is good, but also tracking the spread's net gamma (how fast delta is changing) and net vega (IV sensitivity) in real-time would allow earlier exits when the risk profile deteriorates rapidly, before the 0.50 delta threshold is breached.

---

### 8. Final Verdict

**Verdict: Potentially Tradable With Improvements — but currently not ready for live capital.**

Here is the honest breakdown:

The bones of this strategy are sound. Selling defined-risk weekly credit spreads on SPY with disciplined entry filters, a Thursday hard-close rule, and a 50% profit target is a legitimate approach practiced by professional traders. The architecture shows real thinking — the 11-filter entry gate, the IV rank requirement, the regime switching, the PDT compliance layer, and the crash period stress tests all reflect someone who has read seriously about options trading and thought carefully about the failure modes.

But the backtest is built on a foundation that cannot support the confidence it implies. One year, 67 trades, VIX-as-IV-proxy, 1–2% slippage, no out-of-sample validation, and no adjustment logic. The 83.6% win rate and +27.81% return are not credible as forward estimates — they are the result of one favorable year, optimistic slippage assumptions, and the mathematical compression of the 50% profit target making losses look rarer than they actually are on a per-dollar basis.

The crash tests tell the real story: this strategy can lose 40%+ of a small account in a single bad quarter. On a $5k account, that is $2,000 gone. The response to that is not "the spread cap saved us" — the response is that a 40% drawdown at $5k means you now have $3,000, your margin is constrained, and your PDT-limited trade frequency makes recovery slow and painful.

The risk/reward math (50% profit target, 2× loss stop = roughly 1:4 payoff ratio) requires a realized win rate of 80%+ to be profitable after costs. That is a high bar that has not been demonstrated beyond one favorable year. Before running this live, the minimum credible baseline would be 5 years of backtesting with realistic slippage, walk-forward validation, and at least 50 live paper-trading fills to verify execution assumptions."