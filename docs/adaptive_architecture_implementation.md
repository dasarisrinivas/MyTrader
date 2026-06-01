# Adaptive Architecture — Implementation Journal

Tracks phase-by-phase implementation of `docs/adaptive_architecture_redesign.md`.
Each phase: code changes, rationale, affected modules, expected behavior, validation.

Legend: ✅ done · 🔄 in progress · ⏳ pending · ⛔ blocked

---

## Phase 1 — R:R Neutralization  ✅ IMPLEMENTED & VALIDATED (2026-05-31)

**Validation result (backtest, year-2 2025-03-15→2026-05-26, warm = year-1 learning loaded):**

| Metric | Legacy warm | Cold baseline | Phase 1 warm | Verdict |
|---|---:|---:|---:|---|
| P&L | +$110 | +$867 | **+$900.90** | ✅ ≈ cold (+$34) |
| Trades | 60 | 135 | **134** | ✅ restored |
| Win Rate | 50.0% | 55.6% | **56.0%** | ✅ |
| Profit Factor | 1.08 | 1.29 | **1.30** | ✅ |
| Sharpe | −0.89 | 1.69 | **1.80** | ✅ |
| Max DD | −0.80% | −1.32% | **−1.32%** | ✅ |

Pass criteria met: Phase 1 warm ≈ cold AND ≫ legacy; trade count restored (134≈135);
drawdown unchanged. Adaptive R:R starvation eliminated; path divergence resolved. The
small outperformance vs cold (+$34) indicates the remaining adaptive logic (RR-lowering on
strong buckets; auto_suppress on catastrophic ones) is now net-mildly-positive rather than
destructive — intended behavior. **Phase 1 complete.**

---

### Phase 1 — implementation detail (below)


**Goal:** the adaptive learning layer must never raise the effective R:R floor above the
static base (`cfg.min_rr_ratio = 1.2`). Preserve all risk controls and strategy logic.
Remove adaptive trade starvation.

### Root cause recap
Strategy R:R band = 1.24–1.36 (median 1.255). Base floor 1.2 passes all. The adaptive
layer raised the floor to 1.33–1.5 on flagged buckets → rejected 67/70 lost cold trades
(R:R<1.33) → forked the path-dependent state machine. The adaptive R:R lever is the only
live destructive control variable.

### Code changes

| File | Function / field | Change |
|---|---|---|
| `shree/trading_manager/config.py` | `_envb()` (new) | bool env helper |
| `shree/trading_manager/config.py` | `ManagerConfig.adaptive_modifies_rr` (new) | flag, default **False** (fix ON); env `TM_ADAPTIVE_MODIFIES_RR=1` restores legacy |
| `shree/trading_manager/rules.py` | `evaluate()` | when `not cfg.adaptive_modifies_rr`: `effective_min_rr = min(adaptive.min_rr_required, cfg.min_rr_ratio)` — adaptive may lower but never raise the floor |

**Scope guard:** the cap applies ONLY to the adaptive/learning R:R lever. The
deterministic soft-pause risk control (`soft_pause_min_rr`, applied later via
`max(effective_min_rr, cfg.soft_pause_min_rr)` during a 2–4 loss streak) is **unchanged**.
`auto_suppress` (binary REJECT) is **unchanged** — it is Phase 2 scope and fired 0× in the
walk-forward, so it is not Phase-1-critical.

### Before / after architecture

```
BEFORE (legacy / TM_ADAPTIVE_MODIFIES_RR=1):
  bucket stats ──► derive_thresholds ──► min_rr_required (1.33–1.5)
                                              │
  signal (R:R≈1.25) ──► Q5_rr_ratio ◄─────────┘   1.25 < 1.5 ⇒ REJECT ⇒ path fork

AFTER (default / fix ON):
  bucket stats ──► derive_thresholds ──► min_rr_required (1.33–1.5)
                                              │ cap at base
  signal (R:R≈1.25) ──► Q5_rr_ratio ◄── min(min_rr_required, 1.2)=1.2  ⇒ 1.25 ≥ 1.2 ⇒ PASS
  (soft_pause_min_rr risk control still raises floor to 1.5 during loss streaks)
```

### Verification (unit-level, done)
- `config.py`, `rules.py` compile.
- Default: adaptive floor 1.5 → capped to 1.2 → strategy R:R 1.252 **passes** (was rejected).
- Legacy (`TM_ADAPTIVE_MODIFIES_RR=1`): floor 1.5 applies → 1.252 **rejected** (bug reproduced for A/B).

### Risk assessment
- **Live trading:** LOW. Only relaxes an over-tight learning gate back to the static base
  R:R (1.2). All deterministic risk controls (per-trade cap, daily loss, drawdown halt,
  consec-loss cooldown, soft-pause RR/conf, LOCKED/PROBATION) are untouched. Worst case:
  more trades at base-R:R quality — which is the profitable cold baseline.
- **Backtest parity:** maintained — same code path; flag-gated.
- **Reversibility:** full — `TM_ADAPTIVE_MODIFIES_RR=1` restores prior behavior.
- **Regression surface:** ✅ RESOLVED — SPY uses a **separate `evaluate_spy()`** function
  (rules.py:569). The modified `evaluate()` (rules.py:92) is the MES-only path. No SPY
  impact. Change is fully MES-scoped.

### Expected impact
- **Trade count:** 60 → ~135 (restores the cold universe; adaptive no longer rejects R:R).
- **PF / WR / DD:** expected ≈ cold baseline (PF 1.29, WR ~56%, DD ~−1.3%) — Phase 1 only
  removes the starvation; conviction weighting (Phases 2–3) is what should exceed cold.

### Validation plan (run on capable host — sandbox can't load the 2-yr file)
1. **Unit (done):** flag logic, compile.
2. **Backtest A/B (year 2, 2025-03-15→2026-05-26):**
   - `B` (legacy): `TM_ADAPTIVE_MODIFIES_RR=1 BT_WITH_MANAGER=1 BT_KEEP_DB=1` (warm) → expect ≈ $110 / 60 trades.
   - `A` (Phase 1): default (flag off) warm → expect ≈ cold ($867 / ~135 trades).
   - **Pass criterion:** A ≈ cold and A ≫ B; trade count ≈ 135.
3. **SPY regression:** confirm SPY-options manager decisions are unchanged (diff
   `manager_decisions.jsonl` for spy_signal entries before/after).
4. **Walk-forward:** `scripts/walk_forward_mes.sh` warm run should now ≈ cold (no RR starvation).

### Open items / contradictions found
- None blocking. Note: `BT_ADAPTIVE_RR_CAP` env (ablation toggle) is retained beneath the
  new flag for backtest experimentation; harmless (it can only further-lower the floor).

---

## Phase 2 — Conviction-Based Sizing  ✅ IMPLEMENTED (2026-05-31, awaiting backtest validation)

**Goal:** transform adaptation from rejection/RR into a bounded exposure scalar
(`size_multiplier`) applied AFTER signal generation. Trade universe unchanged.
Default OFF behind `cfg.adaptive_sizing_enabled` (`TM_ADAPTIVE_SIZING`).

### Quantization decision: **Option A (preferred)** — chosen & justified
Backtest uses a **continuous P&L-weight model** (`realized_pnl × size_mult`); live uses
**deterministic integer rounding** (`max(1, round(qty × mult))`, capped at
`max_position_size`). Minor divergence accepted and documented: the backtest's continuous
weight is the economic-truth model for validation; live integer rounding is the executable
reality. This preserves the backtest **universe identity** (entries/exits unchanged — the
weight never alters quantity in backtest, so no path divergence), which is the Phase-2
acceptance requirement. (Option B fractional-risk parity deferred; not needed to prove
correctness.)

### Code changes
| File | Change |
|---|---|
| `learning.py` | `AdaptiveThresholds` += `size_multiplier`, `catastrophic_flag`, `size_tier`. New `_bucket_size_tier()` mapping + `SIZE_MIN/MAX/CATASTROPHIC` consts. `derive_thresholds` populates them (always evaluates catastrophic). |
| `config.py` | `adaptive_sizing_enabled` flag (`TM_ADAPTIVE_SIZING`, default False). |
| `rules.py` | `Decision` += `size_multiplier`/`size_tier`/`catastrophic_flag`. `evaluate()`: always compute+record sizing (`conviction_size` check); `auto_suppress`→REJECT retained ONLY in legacy (sizing disabled); approve/modify carry the scalars. |
| `backtest/engine.py` | order metadata records `size_multiplier`/`size_tier`/`catastrophic_flag`. `realized_pnl` stays RAW (qty=1). |
| `live_trading_manager.py` | `_place_hybrid_order` applies `adaptive_size_multiplier` in the existing qty chain (bounded [0.50,1.25]; 0.0→skip); inert until manager attaches it. |
| `scripts/phase2_validation.py` (new) | dual RAW vs WEIGHTED report + catastrophic audit + sizing distribution + universe-integrity/path-divergence vs Phase 1. |

### Sizing spec (implemented, deterministic)
```
tier mapping (n≥MIN_SAMPLE=5):  WR≥60%&exp>0→EXCELLENT 1.15 · 53–60→GOOD 1.05 ·
  47–53/no-data→NEUTRAL 1.00 · 40–47→WEAK 0.85 · <40→POOR 0.70
catastrophic (ALWAYS evaluated): n≥30 AND WR<10% AND exp≤−1R(avg_loser) → 0.00 (skip)
clamp [0.50,1.25]; catastrophic 0.0 is the only sub-0.50 value & only deletion
```
Verified: EXCELLENT/GOOD/NEUTRAL/WEAK/POOR map correctly; catastrophic fires only at
~0-win/30+ (true total-loss); 1-win/33 → POOR 0.70× (graceful, still trades).

### Catastrophic instrumentation
`catastrophic_flag` is ALWAYS set on `AdaptiveThresholds` and `Decision`, and recorded in
the `conviction_size` check on EVERY evaluation (passed=False only when catastrophic).
Never silently skipped. The validation report counts triggers + their P&L.

### Risk review
- Per-trade risk cap binds on the sized position (max mult 1.25 ≤ existing 2% cap path).
- size_mult ∈ [0.50,1.25] (catastrophic 0.0); live additionally capped by `max_position_size`.
- No new existence gate except catastrophic (near-never). No change to entry timing, signal
  selection, R:R/SL/TP. Default OFF → zero behavior change until enabled & validated.
- MES/SPY isolation preserved (`evaluate()` only; `evaluate_spy()` untouched).

### Validation plan (run on capable host)
```
# 1. Phase-2 backtest (warm, year 2), sizing ON:
TM_ADAPTIVE_SIZING=1 BT_WITH_MANAGER=1 BT_KEEP_DB=1 BT_LEARNING_DB=/tmp/p2.db \
  python3 -m backtest.run --symbol MES --start 2025-03-15 --end 2026-05-26 \
  --bar 1m --bar2 15m --session rth --data-source file --data-file data/ib/ES_1m_multiyr.parquet
# 2. Dual-output + universe-integrity report (Phase2 CSV vs Phase1 CSV):
python3 scripts/phase2_validation.py reports/<phase2>_trades.csv reports/<phase1>_trades.csv
```
**Acceptance:** trade-count diff ≤1% · universe overlap >95% · zero path-divergence ·
weighted PF ≥ raw PF · catastrophic triggers logged (expect ~0).

### Status: implemented & unit-verified. Economic validation FAILED (weighted PF 1.24 <
raw PF 1.29; −$94.84). Sizing distribution showed bucket classification anti-predictive
(EXCELLENT 1.15× buckets −$418; POOR 0.70× buckets +$212).

### Phase 5 forward-predictiveness audit (2026-05-31) — measured, year1→year2:
- Corr(Y1 WR, Y2 P&L)=−0.332; Corr(Y1 Exp, Y2 Exp)=−0.470; Corr(Y1 Edge, Y2 Exp)=−0.574.
- Memory decay does NOT restore predictiveness: corr stays negative at 180/120/90/60/30d
  half-life (−0.37 → −0.27).
- Rolling walk-forward: 3mo train +0.138 mean (median +0.197); 6mo −0.018 (median −0.170);
  9mo +0.038 (median −0.151) — not stably predictive.
- Feature importance: Sample Count +0.557 (frequency, not quality); WR/Exp/Edge all REVERSE
  (−0.33/−0.37/−0.48).

### DECISION: CASE B — memory decay does not restore predictive power.
**Action: Keep Phase 1 only. Disable adaptive sizing permanently (default OFF, leave as-is).
Use bucket learning for TELEMETRY ONLY.** Phase 3 remains BLOCKED (and is not warranted —
no predictive quality signal exists to build a confidence channel on).

**PRODUCTION CANDIDATE = PHASE 1** (adaptive R:R neutralized; +$900.90 / 134 / PF 1.30).

---

## PRODUCTION PACKAGING (2026-05-31) — engineering only, no strategy change

Transition: strategy discovery → production engineering. Phase 1 behavior is
byte-identical (all new logic is additive/flagged-off).

**A+B. Config separation + feature flags** — `shree/config/production_modes.py` (NEW).
`ProductionConfig`: MODE ∈ {BASELINE(default), RESEARCH_ADX, RESEARCH_RTH,
EXPERIMENTAL}; flags `enable_adx_filter / enable_session_filter / enable_orb_filter
/ enable_adaptive_sizing / enable_bucket_learning` — **all default False**. Declarative
registry only; NOT wired into any decision path → zero behavior change. Verified:
`MES_MODE=BASELINE`, all flags False, `any_experimental=False`.

**C. Observability** — `shree/observability/entry_observability.py` (NEW). `log_entry()`
emits per-entry: setup, regime label, ADX, session, entry reason, confidence, and a
P&L attribution tag (`setup/regime/session`). Fully exception-isolated (never raises,
returns None) → cannot affect execution. Writes structured log + `logs/entry_observability.jsonl`.
Wired into `backtest/engine.py` entry (try/except, additive; does not touch trades/P&L).

**D. Safe experiment sandbox** — `scripts/experiment_sandbox.py` (NEW). Copy-on-read
post-hoc simulation of the rejected filters (ADX≥22, RTH-only, OR_BREAK removal,
combined) on a trades CSV; no import of execution/live modules, no production state
touched. Reproduces the deployment-rule check (COMBINED gate destroys 44% of profitable
trades → FAIL vs 30% limit). Existing env toggles (BT_NO_ADAPTIVE, BT_ADAPTIVE_RR_CAP,
TM_ADAPTIVE_SIZING, BT_*) + scripts (walk_forward_mes.sh, rr_ablation_mes.sh,
phase2_validation.py) remain the isolated experiment harness.

**Byte-identical guarantee:** no production *decision* file's logic changed. The
backtest engine edit is an additive, exception-isolated logging call placed after order
submission; trades/P&L are unaffected. To confirm, re-run the Phase-1 backtest and diff
the trades CSV against the prior run (expected: identical).

### Production lock state
- MODE=BASELINE; every experimental flag OFF; no gating; no adaptive logic.
- Files changed for packaging: 3 NEW (production_modes.py, entry_observability.py +
  __init__, experiment_sandbox.py) + 1 additive log call in backtest/engine.py.
- Deliverable status: config separation ✅ · feature flags ✅ · observability ✅ · sandbox ✅.

---
(superseded planning note below)
## Phase 2 — Conviction-Based Sizing  ⏳ pending (do not start until Phase 1 review passes)
## Phase 3 — Confidence Channel  ⏳ pending
## Phase 4 — Regime Sizing  ⏳ pending
## Phase 5 — Memory Decay  ⏳ pending
