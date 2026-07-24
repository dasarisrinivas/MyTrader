# ThetaData Flow Research — RESULT

**Verdict: FAIL → DO NOT PROCEED.** Stop options-flow research. Cancel the
ThetaData Standard subscription before it renews. Do not invent a replacement
hypothesis. The negative result is accepted.

**Date:** 2026-07-24. Referee: [THETADATA_PREREGISTRATION.md](THETADATA_PREREGISTRATION.md)
(locked 2026-07-23, before any data was pulled — untouched).

Production untouched throughout: TC, confidence, governor, allocation, risk,
exits, and all live behavior were never modified. Observation only.

---

## What was tested (real data, not simulated)

- **Source:** ThetaData Options Standard, live. Real OPRA `trade_quote` ticks
  (trade paired with prevailing NBBO), SPY 0DTE near-ATM (±2 strikes).
- **Volume:** 46 sessions, Apr 1–Jul 23 2026; ~300k–750k ticks/session;
  **98% aggressor-classified** (Lee-Ready on real NBBO), verified correct.
- **Snapshots:** 710 SIGNAL (flow in the 30-min window before every signal) +
  598 INTERVAL (unconditional, 13×/session) = 1,308 rows, all with real data.
- **Outcomes:** signal option P&L (`pnl_pct`, 426 with outcomes) and SPY forward
  return at +5/+15/+30 min (SPY 1-min bars from IB — Options Standard excludes
  ThetaData stock data, so the underlying came free from IB).
- **Battery:** quintile lift, permutation/shuffle, direction-confound, feature
  correlation, BH-FDR, and the pre-registered OOS split (train Apr–May / test
  Jun–Jul).

## Results

### Primary hypothesis H1 — delta-weighted aggressor flow (`dw_flow`)
| Test | Result |
|---|---|
| Signal-P&L lift | non-monotonic, top−bottom +0.0019, **shuffle p=0.87** |
| Confound | raw −0.026, partial −0.017 (≈0) |
| Fwd return +5/+15/+30 | **shuffle p=0.64 / 0.94 / 0.32** |
| OOS (fwd +15) | **train p=0.93, test p=1.0** |

`dw_flow` carries **no** forward information. (It IS orthogonal to price
features — max |corr| 0.21 — i.e. genuinely new data that simply predicts
nothing.)

### All flow measures
- **Signal-P&L, all 8 measures × {all signals, 0DTE July cluster}: BH-FDR none pass.**
- **Forward return, all horizons:** only `iv_weighted_side` flickered
  (BH-flagged at +5/+15). Destruction test: significant in-sample
  (train p=0.0025 / 0.0) but **dead out-of-sample** (test p=0.35 / 0.21),
  non-monotonic, effect **2.4–4 bps** — a textbook overfit / multiple-comparisons
  ghost, killed by the pre-registered OOS split. At +30min it fails BH outright.

### Calibration (is the test even capable of finding signal?)
The **existing price features also predict nothing** on the signal-P&L outcome
(shuffle p 0.08–0.67). So flow is not uniquely weak — the outcome is noise to
every feature, and flow adds nothing beyond price. This is consistent with the
six prior audits: the signal set has no forward edge from any feature examined.

## PASS / FAIL scorecard (all six required for PASS)

| # | Criterion | Result |
|---|---|---|
| 1 | Significant incremental information | **FAIL** (H1 p=0.64–0.94; only ghost survived in-sample) |
| 2 | Survives out-of-sample | **FAIL** (dw_flow p=0.93/1.0; iv_weighted_side died OOS) |
| 3 | Not explained by current indicators | (moot — orthogonal but no signal) |
| 4 | Improves the 657 shadow signals | **FAIL** (all measures flat on signal P&L) |
| 5 | Separates winners/losers | **FAIL** (all non-monotonic) |
| 6 | Economically meaningful after costs | **FAIL** (best 4 bps ≪ 0DTE spreads) |

**Any one fails → FAIL. Five of six fail outright.**

## What this proves — and does not

- **Proves:** aggressor-classified 0DTE near-ATM option flow, in a 30-min
  pre-context window, contains no information that predicts either the bot's
  signal outcomes or SPY's near-term forward return, over Apr–Jul 2026, that
  survives shuffle + out-of-sample + multiple-comparisons control.
- **Does not claim:** that all options flow everywhere is useless. Not tested:
  longer flow-accumulation windows (multi-hour/day), full-chain dealer-gamma
  reconstruction, real ISO-tagged sweeps (our sweep proxy over-flagged and was
  excluded), or non-SPY underlyings. Per protocol (Task 7) we do NOT now pursue
  these — that would be inventing a new hypothesis to rescue a null.

## Recommendation

1. **Cancel the ThetaData Standard subscription** before it renews (~$80/mo).
2. **Stop the options-flow research direction.** The one axis believed orthogonal
   to price has been tested with real OPRA data at the same rigor that retired
   every prior feature, and it is null.
3. **Do not invent another flow hypothesis.** Accept the negative result.
4. Value delivered: for ~$80 and one day, a permanent, evidence-backed NO on a
   direction that could otherwise have consumed months. That is the research
   working as designed.

## Cost

- ThetaData Options Standard: one month (~$80). No renewal. IB underlying: $0.
- Net: ~$80 for a definitive, pre-registered NO.
