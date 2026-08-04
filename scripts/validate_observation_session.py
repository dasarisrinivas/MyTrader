"""End-of-day validation for the first observation session after D1-D4.

RESEARCH / REPORTING ONLY — reads logs and ledgers, changes nothing, cannot
place an order.

Answers the ten questions in the 2026-08-04 production-readiness decision with
evidence, and emits a PASS/FAIL/PENDING verdict per item plus a revert path for
anything that regressed.

    python3 scripts/validate_observation_session.py [YYYY-MM-DD]

Timestamps: spy_signals.sent_at and blocked_signals.jsonl `ts` are UTC;
spy_options.log is machine-local (CDT). Both are handled explicitly below.
"""
from __future__ import annotations

import collections
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = "/Users/svss/Documents/code/ShreeBot"
ET = ZoneInfo("America/New_York")

DAY = sys.argv[1] if len(sys.argv) > 1 else datetime.now().date().isoformat()

V2_LEDGER = os.path.join(ROOT, "logs", "v2_shadow_gate.jsonl")
BLOCKED = os.path.join(ROOT, "logs", "blocked_signals.jsonl")
BOTLOG = os.path.join(ROOT, "logs", "spy_options.log")
SIGDB = os.path.join(ROOT, "data", "spy_options_signals.db")

# Commit / revert map for regressions (2026-08-04 Phase 3 + Phase 4).
REVERT = {
    "D1": ("648b694", "shree/spy_options/rules_v2/config.py",
           "EntryGateConfig.confidence_floor_enabled",
           "set confidence_floor_enabled = True"),
    "D2": ("648b694", "shree/spy_options/manager.py", "SpyOptionsManager._poll",
           "remove the upstream v2_shadow_gate.record(_s) loop and restore the "
           "record(sig) call inside _dispatch_signals"),
    "D3": ("648b694", "shree/spy_options/signal_engine.py",
           "log_blocked_signal / _emission_stamp",
           "drop opportunity_key / emission_seq / is_first_emission fields"),
    "D4": ("648b694", "shree/spy_options/signal_engine.py",
           "log_blocked_signal / set_rejection_context",
           "remove the context merge and the manager set_rejection_context call"),
    "INFRA": ("c37b88e", "start_spy_options.sh + deploy/launchd/*.plist",
              "PYTHON_BIN / pgrep guard / AbandonProcessGroup",
              "git revert c37b88e"),
}

results: list[tuple[str, str, str]] = []      # (verdict, question, evidence)


def add(verdict: str, q: str, evidence: str) -> None:
    results.append((verdict, q, evidence))


def load_blocked() -> list:
    rows = []
    if not os.path.exists(BLOCKED):
        return rows
    with open(BLOCKED, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                rows.append({"__malformed__": ln})
                continue
            if str(d.get("ts", "")).startswith(DAY):
                rows.append(d)
    return rows


def load_v2() -> list:
    rows = []
    if not os.path.exists(V2_LEDGER):
        return rows
    with open(V2_LEDGER, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                rows.append({"__malformed__": ln})
                continue
            if str(d.get("ts_utc", "")).startswith(DAY):
                rows.append(d)
    return rows


def db_signals() -> list:
    if not os.path.exists(SIGDB):
        return []
    c = sqlite3.connect(SIGDB)
    c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute(
        "select * from spy_signals where substr(sent_at,1,10)=?", (DAY,))]


def logline_count(pattern: str) -> int:
    if not os.path.exists(BOTLOG):
        return 0
    n = 0
    with open(BOTLOG, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(DAY) and re.search(pattern, line):
                n += 1
    return n


def main() -> int:
    blocked = load_blocked()
    v2 = load_v2()
    sigs = db_signals()
    malformed_b = [r for r in blocked if "__malformed__" in r]
    malformed_v = [r for r in v2 if "__malformed__" in r]
    blocked = [r for r in blocked if "__malformed__" not in r]
    v2 = [r for r in v2 if "__malformed__" not in r]

    print("=" * 78)
    print(f"OBSERVATION-SESSION VALIDATION — {DAY}")
    print("first production session after defects D1-D4 + Phase 4 infra")
    print("=" * 78)
    engine_runs = logline_count(r"Signal Bot starting")
    print(f"engine starts today: {engine_runs}   "
          f"signals persisted: {len(sigs)}   blocked rows: {len(blocked)}   "
          f"v2 rows: {len(v2)}")
    if not sigs and not blocked:
        print("\nNO SESSION DATA — bot did not run or produced nothing. "
              "All checks PENDING.")
        return 2

    # 1 — V2 ledger created
    add("PASS" if os.path.exists(V2_LEDGER) else "FAIL",
        "1. Was v2_shadow_gate.jsonl created?",
        f"{V2_LEDGER} exists={os.path.exists(V2_LEDGER)}, rows today={len(v2)}")

    # 2 — V2 saw every qualified CALL_SWEEP
    cs_total = sum(1 for s in sigs if s["signal_type"] == "CALL_SWEEP")
    cs_blocked = sum(1 for b in blocked if b.get("signal_type") == "CALL_SWEEP")
    cs_seen = cs_total + cs_blocked
    v2_cs = len(v2)
    verdict = "PASS" if (cs_seen == 0 or v2_cs >= cs_seen) else "FAIL"
    if cs_seen == 0:
        verdict = "PENDING"
    add(verdict, "2. Did V2 evaluate every qualified CALL_SWEEP?",
        f"CALL_SWEEP observed={cs_seen} (db {cs_total} + blocked {cs_blocked}); "
        f"V2 ledger rows={v2_cs}. D2 requires V2 >= qualified count.")

    # 3 — D1 duplicate confidence gate gone
    conf_kills = [b for b in blocked
                  if b.get("gate") == "rules_v2:entry_gate"
                  and "confidence" in str(b.get("reason", "")).lower()]
    add("PASS" if not conf_kills else "FAIL",
        "3. Did D1 remove all duplicate confidence filtering?",
        f"rules_v2:entry_gate rejections mentioning confidence = {len(conf_kills)} "
        f"(was 91 on 2026-08-04). Any >0 means the floor is active again.")

    # 4 — D3 dedup markers
    marked = [b for b in blocked if "is_first_emission" in b]
    firsts = [b for b in marked if b.get("is_first_emission")]
    if not blocked:
        add("PENDING", "4. Did D3 eliminate duplicate shadow statistics?",
            "no blocked rows today")
    elif len(marked) < len(blocked):
        add("FAIL", "4. Did D3 eliminate duplicate shadow statistics?",
            f"{len(blocked)-len(marked)} of {len(blocked)} rows lack dedup markers")
    else:
        raw = len(blocked) / max(1, len(firsts))
        add("PASS", "4. Did D3 eliminate duplicate shadow statistics?",
            f"all {len(blocked)} rows marked; unique opportunities={len(firsts)}; "
            f"raw duplication {raw:.1f}x -> 1.0x after is_first_emission filter")

    # 5 — D4 expanded context
    need = ("regime_v2", "atr_ratio", "rsi_5m", "vwap", "delta", "iv_rank")
    missing = collections.Counter()
    for b in blocked:
        for f in need:
            if f not in b:
                missing[f] += 1
    if not blocked:
        add("PENDING", "5. Did D4 capture expanded rejection context?", "no rows")
    elif missing:
        add("FAIL", "5. Did D4 capture expanded rejection context?",
            f"missing fields across rows: {dict(missing)}")
    else:
        add("PASS", "5. Did D4 capture expanded rejection context?",
            f"all {len(blocked)} rows carry {list(need)}")

    # 6 — serialization / logging failures
    errs = logline_count(r"ERROR|CRITICAL|Traceback")
    drops = logline_count(r"NotifyQueue send failed|enqueue failed")
    bad = len(malformed_b) + len(malformed_v)
    add("PASS" if (bad == 0 and errs == 0) else "FAIL",
        "6. Were any serialization or logging failures observed?",
        f"malformed JSON lines={bad}; ERROR/CRITICAL/Traceback={errs}; "
        f"notify drops={drops}")

    # 7 — dispatcher / executor regressions
    disp = logline_count(r"Sending signal")
    execgate = logline_count(r"EXEC gate")
    execorder = logline_count(r"EXEC ORDER")
    pollerr = logline_count(r"Poll error")
    add("PASS" if pollerr == 0 else "FAIL",
        "7. Any dispatcher or executor regressions?",
        f"dispatched={disp}, executor gate evals={execgate}, orders={execorder}, "
        f"poll errors={pollerr}")

    # 8 — unexpected production trades
    add("REVIEW" if execorder else "PASS",
        "8. Unexpected production trades from the architectural fixes?",
        f"EXEC ORDER lines={execorder}. Policy allows <=2/day, 1 open, "
        f"CALL_SWEEP at pilot=1 contract. Any order must match that envelope.")

    # 9 — strategy behaviour change after D1
    fam_disp = collections.Counter(
        s["signal_type"] for s in sigs if s.get("blocked_gate") is None)
    fam_blk = collections.Counter(b.get("signal_type") for b in blocked)
    add("REVIEW", "9. Did any strategy behave differently after D1?",
        f"dispatched by family={dict(fam_disp) or 'none'}; "
        f"blocked by family={dict(fam_blk) or 'none'}. "
        f"Expect CALL_SWEEP/BULL_CALL_SPREAD/PC_RATIO_EXTREME to appear "
        f"downstream of entry_gate for the first time.")

    # 10 — further architectural change needed
    fails = [q for v, q, _ in results if v == "FAIL"]
    add("PASS" if not fails else "FAIL",
        "10. Further architectural changes needed before continuing?",
        "no failing checks" if not fails else f"{len(fails)} failing: {fails}")

    print()
    for v, q, ev in results:
        mark = {"PASS": "PASS   ", "FAIL": "FAIL   ",
                "PENDING": "PENDING", "REVIEW": "REVIEW "}[v]
        print(f"[{mark}] {q}")
        print(f"          {ev}")

    if fails:
        print("\n" + "=" * 78)
        print("REGRESSION — REVERT PATHS")
        print("=" * 78)
        for key, (commit, path, func, how) in REVERT.items():
            if any(key[-1] in q[:3] for q in fails) or key == "INFRA":
                print(f"  {key}: commit {commit}")
                print(f"      file    {path}")
                print(f"      symbol  {func}")
                print(f"      revert  {how}")
    print()
    counts = collections.Counter(v for v, _, _ in results)
    print(f"SUMMARY  {dict(counts)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
