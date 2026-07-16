# SPY Options Exit Engine v2 — Design (2026-07-14)

Scope: **exit logic only.** Entry, confidence engine, sizing, strike selection, risk limits, and order routing are untouched. The IB-resident bracket stop-loss remains the ultimate hard protection and is never weakened.

## 0. Current state and why it churns

`manager._check_exit_conditions()` runs every poll (~75 s) with **7 independent one-shot triggers; any single `reasons.append(...)` → `executor.close_position(key, reasons[0])` → 100% close**:

| # | Trigger | Defect |
|---|---|---|
| 5 | Profit "target" +0.5% SPY | Full-closes winners at ~+0.4R; the 1.5R bracket TP never gets a chance |
| 1 | Adverse ≥0.5% SPY | One-poll, no confirmation |
| 2 | Regime flip | **Single-poll** classifier output; flips back next bar |
| 3 | Adverse ≥1.0% | Legitimate (catastrophic) |
| 4 | Time stop 45/90 min | Legitimate ceiling (theta-motivated) |
| 6 | VWAP reversion | **Single-poll band demotion** — ABOVE_1SD→INSIDE_1SD once = exit. Killed the Jul-13 749P 106 s after placement. Audit: this label = 0% WR for puts in TREND_DOWN |
| 7 | IV-adjusted premium stop | Redundant with the IB bracket SL, delta-approximated |

No grace period, no hysteresis, no partials, first-reason-wins labeling (poisons analytics).

---

## 1. Architecture

New module `shree/spy_options/exit_engine.py`. The manager becomes a thin adapter; the executor gains two order-management primitives. Nothing upstream changes.

```
manager._poll (every ~75s)
  └─ _check_exit_conditions()          [ADAPTER — builds inputs, applies decision]
       ├─ snap  = ExitSnapshot.from_live(...)   # pure data: bars, bands, regime, greeks…
       ├─ state = self._exit_states[key]        # persistent per-position counters/stage
       ├─ decision = ExitEngine.evaluate(state, snap)   # PURE FUNCTION
       └─ apply:
            HOLD            → nothing (log score at DEBUG)
            TIGHTEN_STOP_BE → executor.move_stop(key, be_price)      [NEW primitive]
            PARTIAL_EXIT    → executor.partial_close(key, fraction)  [NEW primitive]
            TRAIL_STOP      → executor.move_stop(key, trail_price)
            FULL_EXIT       → executor.close_position(key, reason)   [existing]

Hard protections that live OUTSIDE this engine (unchanged, higher priority):
  • IB bracket stop-loss (resident at exchange — works even if bot dies)
  • executor 15:50 0DTE flatten, 14:00 EOD flatten script
  • TM kill-switches
```

**Components**

| Class | Role |
|---|---|
| `ExitSnapshot` | Frozen per-evaluation inputs: last N **closed** 5m bars (forming-bar fix guarantees this), VWAP + band position, EMA9/21 + slopes, RSI5m, ATR14, regime label, tape score, unrealized R (premium, vs structural stop), delta now vs entry, DTE, VIX, minutes held, minutes to close |
| `PositionExitState` | Persistent: stage, entry snapshot (band/regime/conf/tier/delta), confirm counters (`vwap_adverse_closes`, `regime_flip_evals`, `pending_evals`), HWM R, partials taken |
| `ExitConfidenceScorer` | Binary factors × weights → 0–100 + per-factor breakdown (explainability) |
| `ExitPolicy` | Priority ladder (§8) + adaptive threshold (§9) |
| `ExitEngine.evaluate()` | Pure, deterministic: `(state, snap) → ExitDecision(action, score, factors, reason)` |

Determinism: factors read **closed bars only** + current poll scalars; no wall-clock randomness; same `(state, snap)` → same decision. Every decision logs the factor table.

---

## 2. Decision flow (per evaluation)

```
evaluate(state, snap):
│
├─ [P2] CATASTROPHIC?  adverse SPY ≥1.0% ∨ premium ≤ −(iv_stop+10)% ∨ VIX +15% intraday
│         └─ yes → FULL_EXIT("catastrophic")            (grace does NOT protect)
├─ [P3] CEILING?  held ≥ max_hold ∨ 0DTE ≥ flatten time
│         └─ yes → FULL_EXIT("max hold")
├─ GRACE?  held < grace_min → HOLD (score computed & logged, not acted on)
├─ [P5] PROFIT LADDER  (checked before soft exits — winners managed, not dumped)
│         ├─ R ≥ +1.0 ∧ no partial yet → PARTIAL_EXIT(50%) + TIGHTEN_STOP_BE → stage BE_LOCK
│         └─ stage BE_LOCK ∧ R_hwm − R ≥ trail_giveback → TRAIL_STOP(lock 50% of HWM gain)
├─ [P4] EXIT CONFIDENCE
│         ├─ score = Σ(active factor weights), cap 100
│         ├─ thr   = adaptive_threshold(state, snap)
│         ├─ score ≥ thr+15                    → FULL_EXIT (one-shot override)
│         ├─ score ≥ thr:
│         │     stage==EXIT_PENDING (2nd consecutive) → FULL_EXIT
│         │     else → stage=EXIT_PENDING, HOLD        (confirmation eval)
│         └─ score < thr−10 ∧ stage==EXIT_PENDING → de-escalate to MANAGE (hysteresis)
└─ HOLD
```

## 3. Exit state machine

```
            ┌────────┐  held ≥ grace_min   ┌────────┐  R ≥ +1R (partial+BE)  ┌─────────┐
 entry ───► │ GRACE  │ ───────────────────►│ MANAGE │ ──────────────────────►│ BE_LOCK │
            └────────┘                     └────────┘                        └─────────┘
   hard exits only        score ≥ thr ▲│ score < thr−10          trail / bracket manage
   (P1/P2/P3)                         │▼ (de-escalate)
                                ┌──────────────┐   2nd consecutive ≥ thr,
                                │ EXIT_PENDING │   or one-shot ≥ thr+15     ┌────────┐
                                └──────────────┘ ──────────────────────────►│ EXITED │
                                                                            └────────┘
 Any state: P1 IB bracket SL / P2 catastrophic / P3 ceiling → EXITED
```

## 4. Exit Confidence factors (binary × weight, cap 100)

All "closes" = **closed 5m bars**, all thesis-relative (mirrored for calls/puts — fixes the put asymmetry):

| Factor | W | Activation |
|---|---|---|
| `vwap_full_reversion` | 25 | Close **crossed VWAP itself** to the opposite side, **2 consecutive closes** |
| `vwap_band_decay` | 10 | Entry band lost (e.g. ABOVE_1SD→INSIDE_1SD) for **≥3 consecutive closes** AND EMA9 slope against thesis — a single-bar band demotion contributes **zero** |
| `ema9_cross` | 15 | 2 consecutive closes beyond EMA9 against thesis |
| `ema21_break` | 20 | 1 close beyond EMA21 against thesis (structure break) |
| `regime_flip_confirmed` | 20 | Opposite regime on **≥3 consecutive evaluations** AND EMA9 slope reversed — one-poll flips contribute zero |
| `momentum_stall` | 10 | No new favorable extreme in last 4 bars AND last 2 closes adverse |
| `rsi_reversal` | 10 | RSI5m crossed 50 against thesis (entry side was ≥60/≤40) |
| `tape_flip` | 10 | RealFlow tape sign against thesis, 2 consecutive polls (skip if feed unavailable) |
| `atr_adverse_expansion` | 15 | One adverse closed bar with range > 1.5 × ATR14 |
| `delta_decay` | 10 | \|Δ_now\| < 0.7 × \|Δ_entry\| (thesis not paying; theta winning) |

Pullback tolerance for TREND_CONTINUATION is emergent: an EMA9 retest scores at most 15+10=25 — far below any threshold. Exit requires **stacked, confirmed** evidence (e.g. VWAP cross + EMA21 break + regime = 65).

## 5. Adaptive threshold (§9 requirement)

```
thr = 55  (base)
    + 10 if entry tier EXTREME or entry conf ≥ 0.85     # quality earns room
    +  5 if trend structure intact (regime == entry regime AND EMA9 slope with thesis)
    +  5 if unrealized R ≥ +0.5                          # winners get room
    − 10 if DTE == 0 and past 14:00 ET                   # theta urgency
    −  5 if VIX ≥ 24                                     # tail risk
    −  5 if held ≥ 0.75 × max_hold                       # graceful pre-ceiling
clamp [40, 80]
```

## 6. Pseudocode (core)

```python
def evaluate(state: PositionExitState, snap: ExitSnapshot) -> ExitDecision:
    if catastrophic(snap):                      # P2 — grace does not shield
        return ExitDecision(FULL_EXIT, 100, [], "catastrophic")
    if snap.minutes_held >= snap.max_hold_min or past_flatten(snap):   # P3
        return ExitDecision(FULL_EXIT, 100, [], "max_hold")

    update_counters(state, snap)                # consecutive-close / regime counters

    if snap.minutes_held < cfg.grace_min(snap.dte):                    # grace
        return ExitDecision(HOLD, score_of(state, snap), factors, "grace")

    r = snap.unrealized_r
    state.hwm_r = max(state.hwm_r, r)
    if r >= cfg.partial_at_r and not state.partial_taken:              # P5
        state.partial_taken = True; state.stage = BE_LOCK
        return ExitDecision(PARTIAL_EXIT, 0, [], "profit_ladder_1R",
                            fraction=cfg.partial_fraction, new_stop=snap.entry_premium)
    if state.stage == BE_LOCK and state.hwm_r - r >= cfg.trail_giveback_r:
        return ExitDecision(TRAIL_STOP, 0, [], "trail",
                            new_stop=premium_at_r(snap, state.hwm_r * 0.5))

    score, factors = score_of(state, snap)                             # P4
    thr = adaptive_threshold(state, snap)
    if score >= thr + cfg.one_shot_margin:
        return ExitDecision(FULL_EXIT, score, factors, "exit_confidence_strong")
    if score >= thr:
        if state.stage == EXIT_PENDING:
            return ExitDecision(FULL_EXIT, score, factors, "exit_confidence_confirmed")
        state.stage = EXIT_PENDING
        return ExitDecision(HOLD, score, factors, "pending_confirmation")
    if state.stage == EXIT_PENDING and score < thr - cfg.deescalate_margin:
        state.stage = MANAGE                                           # hysteresis
    return ExitDecision(HOLD, score, factors, "hold")
```

## 7. Configuration (suggested initial values)

```python
@dataclass
class ExitEngineConfig:
    enabled: bool = True
    shadow_mode: bool = True          # log decisions, don't act (rollout, §10)
    grace_min_0dte: int = 6           # ≥1 full closed 5m bar + slack
    grace_min_swing: int = 10
    base_threshold: int = 55
    one_shot_margin: int = 15         # thr+15 → exit without 2nd confirmation
    deescalate_margin: int = 10       # < thr−10 → leave EXIT_PENDING
    partial_at_r: float = 1.0
    partial_fraction: float = 0.5
    trail_giveback_r: float = 0.5     # in BE_LOCK, give back ≤0.5R from HWM
    vwap_band_decay_closes: int = 3
    vwap_cross_closes: int = 2
    regime_confirm_evals: int = 3
    # weights as table §4; max_hold stays 45/90 (existing, calibrated)
```

The +0.5% SPY "profit target" trigger is **deleted** — profit-taking is the ladder (partial at +1R, BE, trail); the IB bracket TP (1.5R) handles the runner.

## 8. Priority ladder (hard-coded order)

1. IB bracket stop-loss (exchange-resident — outside engine)
2. Catastrophic (adverse ≥1.0% SPY / premium ≤ −(iv_stop+10)% / VIX spike)
3. Time ceiling (45/90 min) + 0DTE 15:50 flatten + EOD
4. Exit Confidence ≥ adaptive threshold, confirmed
5. Profit protection (partial / BE / trail)
6. Weak-thesis factors — **feed the score only, never exit alone**
7. Informational (Telegram context only)

## 9. Executor primitives (order management only — routing untouched)

- `partial_close(key, fraction)`: market-SELL `round(qty×fraction)`; reduce bracket children `totalQuantity` via ib_insync modify (same orderId re-place).
- `move_stop(key, new_stop)`: modify SL child `auxPrice`/`lmtPrice` (same orderId). Never widens: `new_stop = max(new_stop, current_stop)` for longs.
- Both idempotent + logged with the factor table.

## 10. Unit tests (pytest, pure-function — no IB)

```
test_grace_blocks_vwap_reversion_at_2min          # Jul-13 749P scenario → HOLD
test_hard_stop_active_during_grace                # catastrophic at 1min → FULL_EXIT
test_single_band_demotion_scores_zero             # ABOVE_1SD→INSIDE once → factor inactive
test_band_decay_needs_3_closes_and_ema_slope
test_regime_flip_single_eval_no_exit
test_regime_flip_3_evals_plus_ema_confirms
test_score_at_threshold_requires_second_eval      # EXIT_PENDING → confirmed
test_score_thr_plus_15_exits_one_shot
test_deescalation_returns_to_manage               # hysteresis
test_partial_at_1R_moves_stop_to_BE_once          # idempotent
test_trail_locks_half_hwm_gain
test_time_ceiling_beats_low_score                 # priority ordering
test_put_call_factor_mirror_symmetry              # same scenario mirrored → same score
test_determinism_same_inputs_same_output
test_0dte_afternoon_threshold_drops_10
```

## 11. Backtest methodology

1. **Entries held constant** — replay the exact historical entry set (219-signal DB + the rules_v2 replay harness from the filter-ablation study). Exits are the *only* independent variable.
2. **Two simulators on identical entries**: (a) current 7-trigger one-shot engine (faithful port, incl. trigger order); (b) ExitEngine v2.
3. Option P&L via the delta/BS approximation from the greeks-strike backtest (flat ATM IV per DTE — stated limitation); same-bar stop/target ambiguity resolved as stop (conservative).
4. **Walk-forward**: calibrate weights/threshold on days 1–40, validate untouched on days 41–60. Report validation only.
5. **Significance**: 2,000× bootstrap CI on per-trade expectancy delta; permutation test on PF delta.
6. **Premature-exit definition (measurable)**: non-hard-stop exit where price subsequently reaches the original 1.5R target within 60 min. Also churn = exits < 10 min after entry.
7. **Acceptance to leave shadow mode**: premature rate −50 %, expectancy delta CI excludes ≤0, max DD ≤ 1.2× baseline.
8. **Rollout**: deploy `shadow_mode=True` — engine logs its decision alongside the live one-trigger engine each poll; after ≥2 weeks compare shadowed vs actual outcomes on real fills, then flip.

## 12. Before/after metrics (report table)

| Metric | Baseline (7-trigger) | ExitEngine v2 |
|---|---|---|
| Avg hold time (min) | | |
| Premature exit rate (§11.6) | | |
| Churn rate (exit <10 min) | | |
| Win rate | | |
| Profit factor | | |
| Avg R multiple | | |
| % reaching full 1.5R target | | |
| Max drawdown | | |
| Exit-reason distribution (labels now additive, not first-wins) | | |
