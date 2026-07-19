# Portfolio Governance — deterministic rules (2026-07-19)

The SPY options portfolio is self-governing. `scripts/portfolio_governor.py`
runs every **Saturday 09:00 CT** (launchd `com.shree.spy-weekly-governor`),
computes per-family evidence, applies the rules below, writes
`data/family_tiers.json` (which the executor reads — governor > config
override > code default), and sends the weekly Telegram report. **No LLM, no
opinion, no manual edits.** Changing a tier by hand means editing config
overrides — the governor will still re-derive its own verdict next weekend.

## Evidence tiers (never mixed)
- **LIVE** — real executor fills with recorded P&L. The only dollars.
- **SHADOW-dispatched** — sent signals, simulated exits (`blocked_gate` NULL).
- **SHADOW-blocked** — gate-killed signals, simulated exits (counterfactual).

## Tier ladder and size caps
```
experimental (0 contracts, shadow only)
  → pilot (1)  → probation (2)  → production (3)  → core (risk-budget)
```

## Promotion rules (one step per week, evidence must clear)
| Step | Rule |
|---|---|
| experimental → pilot | shadow pooled decided **n≥30** AND **Wilson-lo > 40%** (breakeven for 1R/1.5R bracket) |
| pilot → probation | **≥5 live fills** AND live expectancy **> $0** |
| probation → production | **≥15 live fills** AND live **PF ≥ 1.15** AND live Wilson-lo **≥ 40%** |
| production → core | **≥40 live fills** AND **PF ≥ 1.30** AND \|max DD\| ≤ 6× avg win |

## Demotion rules (checked before promotions; immediate at review)
| Rule | Action |
|---|---|
| D1: live rolling-10 P&L sum < 0 (n≥6) | down **1 tier** |
| D2: live PF < 0.8 (n≥10) | down to **pilot** |
| D3: shadow pooled Wilson-lo < 30% (n≥30) | to **experimental** |

**Intra-week protection is separate and faster:** the executor's rolling
auto-kill (last ≥4 trades of a pilot family sum < 0 → family refuses entries)
and the TM/bracket/daily-loss stack act immediately, not on weekends.

## Adaptive risk scale (within tier; never a fixed allocation)
Start 1.0 → **−0.25** if rolling-10 EV < half of all-time EV (decay) →
**−0.25** if |max DD| > 3× avg win → floor **0.5**. Core tier may reach 1.25.
Executor multiplies position size by the scale (min 1 contract).

## Edge-decay detection (report alerts; suggest, rules act)
- rolling-10 EV < 50% of all-time EV → "decay" (also cuts risk scale)
- confidence calibration BROKEN: high-conf WR < low-conf WR (both n≥10)
- live WR trails shadow WR by >15 pts (execution quality suspect)
- D1–D3 above are the enforced versions of decay.

## Regime guards (family-specific, config)
- `PC_AFTERNOON_FLOW`: VIX ≤ 25 (edge proven 15.9–22.0 only), 14:00–14:55 ET.
- `VWAP_REVERSION`: shadow-only; window 10:30–13:30 ET hard stop.

## What remains manual (by design)
1. Reviewing the weekly Telegram report (read-only).
2. Approving a **core**-tier promotion in config (capital-scale decision).
3. IB Gateway login and account-level actions.
4. Backfilling the 4 pre-fix trades with missing exit P&L from IB statements.
