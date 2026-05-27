# Why no trades fire (MES + SPY) — root cause (2026-05-26)

## Verdict

The strategies are **not** broken and they **are** generating signals. The shared
**trading_manager is vetoing ~100% of them**, so nothing reaches the broker. The
binding fault is a single static rule that is incompatible with how these strategies
are built.

## Evidence

**MES — signals generated, then 100% vetoed.** `live_trading.log` shows the strategy
emitting BUY signals (EMA21_PB_LONG, EMA9_PB_LONG, TREND_CONT_LONG, PROX_LONG). The
trading_manager then rejects every one. Last 400 manager decisions for `es_fifteen_min`:
**400 REJECT, 0 APPROVE**, all with the same reason:

> `R:R 1.33 < required 2.0 (static). Mandate requires asymmetric payoff.`

**SPY — same shared manager.** `spy_options.log`: the signal engine emits straddles,
which then hit `🛡️ TRADING MANAGER VETO (SPY): LONG_STRADDLE — skip`. Between vetoes
it logs "no signals this cycle." Same gate, same outcome.

## The fundamental logic error

The trading_manager enforces a **static R:R ≥ 2.0 floor** (`config.py:72`,
`min_rr_ratio = TM_MIN_RR default 2.0`; `rules.py` Q5 check). But `es_fifteen_min` is a
**high-win-rate, low-R:R** strategy — its take-profit is ~1.25–1.35× its stop, so it
*structurally* produces ~1.3:1 setups (visible in every signal: SL 9.4 / TP 11.8,
SL 6.0 / TP 8.0, …). A 2.0 floor can therefore **never** approve a normal signal from
this strategy. The two are incompatible by construction.

This is the wrong gate for this strategy. At its validated ~55% win rate, a 1.33:1
setup has **positive** expectancy (0.55×1.33 − 0.45×1 ≈ +0.15R). Demanding a 2.0
"asymmetric payoff" is a rule for a *low*-win-rate trend system, not a high-WR one.

## Why it's stuck (cold-start deadlock)

There's an adaptive path that can relax the floor below 2.0 — but only with
accumulated winning-trade history. The `shadow_expectancy` record shows it computing
off a stale prior (`wr_prior 0.4091, n=22`) and still deferring to `static_floor 2.0`,
`would_pass: False`. Because every trade is vetoed, no history accrues, so the floor
never relaxes → **permanent deadlock**. This is exactly the cold-start deadlock in
`docs/rag_authority_review.md`, now manifesting as zero live trades.

## Important: this gate was never in the validated backtest

The 26-month validation (+$2,142, 53–59% WR) ran the strategy through `RiskManager`
only — it **never applied the trading_manager's R:R ≥ 2.0 gate**. So live is strictly
more restrictive than the configuration that was proven profitable, to the point of
taking zero trades.

## Recommended fix

Align the manager's R:R floor with the strategy's validated design:

- **Set `TM_MIN_RR=1.2`** (env var read at `config.py:72`; currently unset → defaulting
  to 2.0). The strategy emits 1.25–1.4, so a 1.2 floor admits the validated signals
  while still rejecting genuinely poor payoffs. One-line, reversible, no code change.
- Also fix the **stale expectancy prior**: the shadow uses `wr_prior 0.4091 (n=22)`,
  but the 26-month book is ~55% WR. Updating that prior makes the adaptive path's
  `required_rr` ≈ 1.0 — consistent with the strategy — so it won't re-block once the
  static floor is lowered.

## One caveat to weigh first

`trading_manager_state.json` shows `health_status: SUSPECT`, last 5 outcomes all LOSS,
"WR=20% over 30 trades, red_days=4." That recent *live* record (pre-deadlock) is poor
and conflicts with the 26-month backtest — likely small-sample variance or trades from
before the shorts-off / ADX-12 changes were deployed. The fix unblocks trading back to
the validated config; given the SUSPECT flag, it's reasonable to watch the first days
closely (or paper-confirm) rather than size up immediately.

## SUSPECT-health investigation (the 20%-WR record)

Pulled the last 30 real live executions (orders.db `executions`; ignored the orders
table's $100k+ sums — backtest/paper pollution). Findings:

- **Span: 2026-03-20 → 2026-05-07.** Live fills *stopped May 7* — well before the
  May-25 config change and the R:R deadlock. So this record is **100% old-config**
  (shorts on, ADX 18, no opening/Friday blocks, London/OR signals live).
- **Shorts: 17 trades, 18% WR, −$380** — exactly the drag the validated change disables.
- **Longs: 13 trades, 23% WR, −$283** — old-config longs (ADX-18 floor, no blocks),
  small sample, in a choppy Mar–May tape (EMAs flat, 7510–7550).
- **The validated config (shorts-off, ADX-12, blocks) has taken ZERO live trades** —
  the R:R deadlock blocked it from the moment it was deployed.

**What this means:** the SUSPECT health is *not* a verdict on the validated config —
it's the inferior old config (plus the now-removed shorts) trading a choppy month. It
should not, by itself, stop us from unblocking the validated config.

**But one yellow flag:** the old-config *longs* also underperformed the backtest here
(23% WR live vs ~55–59% in-sample). Some of that is config (old longs ≠ new longs) and
small-sample chop, but it's a reminder that backtest→live edge transfer is unproven for
this book. Treat the first live trades of the new config as confirmation, not a
foregone conclusion.

## Recommendation (revised)

1. **Fix the deadlock** so the validated config can actually trade: `TM_MIN_RR=1.2`
   (+ refresh the stale expectancy prior to the 26-month ~55% WR).
2. **Deploy carefully** — minimal size (or a short paper window) for the first ~10–20
   trades to confirm the longs-only book performs in line with backtest *before* sizing
   up. The old-config live record argues for verifying transfer, not blind trust.

## Edge-transfer deep-dive (the 26 live longs)

Dissected the 26 true live long entries (Mar–May, old config):

- **Slippage averaged −0.73 pts — *favorable* for a long** (filled cheaper than
  intended). Execution is clean, so the single biggest backtest→live risk is **ruled
  out**: the backtest's fills hold up live.
- **By session:** RTH 15 trades / 27% WR / −$222; **OVERNIGHT 11 trades / 9% WR /
  −$365.** Overnight was the dominant drag.
- avg R:R 1.16, with 7 trades at ~1.0 (the old fixed-stop floor — worse than the new
  config's ~1.33 structural stops).

**The reconciling fact:** in the validated 26-month backtest, the new config takes only
**8 overnight longs total** (75% WR, +$351) — overnight is negligible. The old live
setup took 11 overnight longs in *7 weeks*. And `bot.log` confirms the **current live
bot already skips overnight** ("15m non-RTH bar (OVERNIGHT): skipping strategy,
exit-check only"). So the worst live drag is **already structurally eliminated** in the
current setup.

| Validated 26mo longs (ADX-12) | n | WR | PnL |
|---|---|---|---|
| All | 392 | 53% | +$2,142 |
| **RTH (the live-relevant book)** | **384** | **53%** | **+$1,791** |
| Overnight | 8 | 75% | +$351 |

## Edge-transfer verdict

Reasonably encouraging, though not provable until the new config trades live:
- ✅ Fills are clean/favorable — no execution tax; backtest PnL should transfer.
- ✅ The catastrophic live bucket (overnight longs) is already disabled (RTH-only live).
- ✅ The old-config RTH losers ran at ~1.0 R:R with no opening/Friday blocks; the new
  config improves R:R to ~1.33 and removes the opening-30min whipsaw + Friday TREND_CONT
  — i.e. the new RTH longs are a *better* book than the ones that lost.
- ⚠️ Still unproven on live data; the validated edge is thin per-trade (~+$4.7/RTH
  trade), so confirm the first ~10–20 RTH longs track ~53% WR before sizing up.

## Fix APPLIED (2026-05-27) + verification

Changed (both reversible config edits — no strategy/code logic touched):
- `deploy/launchd/com.shree.trading-manager.plist` → added `TM_MIN_RR=1.2` and
  `TM_SOFT_PAUSE_MIN_RR=1.3` to EnvironmentVariables (the active launcher).
- `start_trading_manager.sh` → same two env exports (covers the nohup launch path).

**Verified by replaying 300 real rejected MES signals through `rules.evaluate()`**
with the current live state (posture NORMAL, cold-start recent=[]):

| Floor | APPROVE | REJECT | MODIFY |
|---|---|---|---|
| OLD min_rr=2.0 | 0 | 300 | 0 |
| **NEW min_rr=1.2** | **259** | **40** | **1** |

The deadlock breaks (0% → 86% approval). The 40 remaining rejects are *legitimate*,
not the deadlock: 28 are Q1/Q2 regime/strategy-fit mismatches, ~11 are per-trade risk
> $90 cap (a 20-pt stop × $5 = $100 on the $4,499 account). So the manager keeps its
real risk controls — it's not "approve everything."

Standard strategy backtest re-run after the edits: ADX-12 chunk = 152 trades / +$255.2,
**identical** to before — confirms the default-off MOM_BRK signal and the engine fix did
not regress the validated book.

**ACTION REQUIRED (user):** the manager must be **restarted** to pick up the new env —
reload the launchd agent (re-copy the plist to `~/Library/LaunchAgents/` if that's where
it's loaded from, then `launchctl unload`/`load`), or restart via your manager scripts.
After restart, `logs/trading_manager.log` should start showing APPROVE decisions instead
of "R:R < required 2.0".

## Win/loss of the now-approved signals

Asked whether the unblocked signals would win or lose. I tried to simulate each
approved signal directly (first-touch SL/TP on 1m data), but **validated that sim
against the engine first — and it failed**: on the *same 392 validated trades*, the
engine gives 53.1% WR / +$2,142 while the standalone first-touch sim gives 34.2% /
−$4,794 (15% sign agreement). The ad-hoc sim doesn't reproduce the engine's fill/exit
logic, so its number is discarded. Also, the manager-log signal sample is a polluted
multi-run slice, not a clean population.

**Trustworthy answer (from the engine, which is the faithful backtest):** the validated
config's trades — i.e. the trades that would actually be taken — are **392 trades,
53.1% WR, +$2,142** over 26 months (winners). The manager at `min_rr=1.2` approves ~86%
of generated signals, rejecting only regime-mismatches and over-$90-risk trades (the
marginal ones), so the approved book stays this winning profile. Net: **if taken, the
book is a net winner (~53% WR)** on backtest — with the standing caveat that live
transfer is unproven, so confirm on the first ~10–20 RTH longs.

## Summary

Not a strategy bug and not a data bug — a **manager/strategy R:R mismatch**: a static
2.0 R:R mandate vetoed a strategy designed for ~1.3:1, and the cold-start deadlock kept
it locked. The poor recent live record is **old-config + overnight + shorts**, none of
which the validated RTH longs-only book trades. Fills transfer cleanly. Fix applied
(`TM_MIN_RR=1.2`), verified (0%→86% manager approval), no backtest regression. Restart
the manager to activate; deploy at minimal size and confirm the first ~10–20 RTH longs
before sizing up.
