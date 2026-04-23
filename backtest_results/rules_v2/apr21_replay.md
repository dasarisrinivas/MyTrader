# rules_v2 replay report

## Summary

- `allowed_continuation`: 1
- `blocked_by_orb_gate`: 5
- `blocked_by_pc_ratio`: 2
- `blocked_legacy`: 7
- `total_continuation`: 1
- `total_legacy`: 7

## Events

| time | kind | signal | dir | price | conf | allowed | rule | reason | regime |
|---|---|---|---|---|---|---|---|---|---|
| 2026-04-21T11:10-04:00 | LEGACY | ORB_BREAKOUT | P | 710.25 | 0.78 | ✗ | orb_gate | ORB time-gate: 11:10 ET outside [09:45, 11:00) — breakout is stale; regime has moved on | TRANSITION |
| 2026-04-21T12:35-04:00 | CONTINUATION | TREND_CONTINUATION | P | 706.75 | 0.82 | ✓ | pass | pass: new zone (leg#0, 0 entries prior) | TREND_DOWN |
| 2026-04-21T14:10-04:00 | LEGACY | PC_RATIO_EXTREME | P | 704.90 | 0.88 | ✗ | pc_ratio | PC_RATIO suppressed in TRANSITION regime | TRANSITION |
| 2026-04-21T14:10-04:00 | LEGACY | ORB_BREAKOUT | P | 704.90 | 0.82 | ✗ | orb_gate | ORB time-gate: 14:10 ET outside [09:45, 11:00) — breakout is stale; regime has moved on | TRANSITION |
| 2026-04-21T14:25-04:00 | LEGACY | ORB_BREAKOUT | P | 704.40 | 0.82 | ✗ | orb_gate | ORB time-gate: 14:25 ET outside [09:45, 11:00) — breakout is stale; regime has moved on | TRANSITION |
| 2026-04-21T14:35-04:00 | LEGACY | PC_RATIO_EXTREME | P | 704.10 | 0.87 | ✗ | pc_ratio | PC_RATIO suppressed in RANGE_BOUND regime | RANGE_BOUND |
| 2026-04-21T14:35-04:00 | LEGACY | ORB_BREAKOUT | P | 704.10 | 0.78 | ✗ | orb_gate | ORB time-gate: 14:35 ET outside [09:45, 11:00) — breakout is stale; regime has moved on | RANGE_BOUND |
| 2026-04-21T14:35-04:00 | LEGACY | ORB_BREAKOUT | P | 704.10 | 0.80 | ✗ | orb_gate | ORB time-gate: 14:35 ET outside [09:45, 11:00) — breakout is stale; regime has moved on | RANGE_BOUND |
