#!/usr/bin/env python3
"""Portfolio Governor — deterministic weekend tier review (2026-07-19).

Self-governing portfolio: reads the three evidence tiers (LIVE fills /
SHADOW-dispatched / SHADOW-blocked, never mixed), applies the documented
promotion/demotion rules, writes data/family_tiers.json (the executor's
runtime tier source), and sends the weekly Telegram report. No LLM, no
opinion, no manual edits — docs/PORTFOLIO_GOVERNANCE.md is the contract.

Run: python3 scripts/portfolio_governor.py [--dry-run]
Scheduled: Saturdays 09:00 via com.shree.spy-weekly-governor (launchd).
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DB = os.path.join(ROOT, "data", "spy_options_signals.db")
TIER_FILE = os.path.join(ROOT, "data", "family_tiers.json")

TIERS = ["experimental", "pilot", "probation", "production", "core"]
BREAKEVEN_WR = 40.0          # % — bracket geometry 1R stop / 1.5R target

# Families the governor manages. Code-default tiers (first run seed).
SEED = {
    "TREND_CONTINUATION": "probation",
    "PC_AFTERNOON_FLOW": "pilot",
    "VWAP_REVERSION": "experimental",
    "ORB_BREAKOUT": "experimental",
    "PC_RATIO_EXTREME": "experimental",
    "CALL_SWEEP": "experimental",
    "PUT_SWEEP": "experimental",
}


def wilson_low(w: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    e = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return round((c - e) / d * 100, 1)


def family_stats(con, fam: str) -> dict:
    """All metrics for one family, per evidence tier."""
    s = {"family": fam}

    # LIVE — real fills with recorded P&L (chronological for rolling/DD).
    rows = con.execute(
        """SELECT fill_pnl_usd FROM spy_signals
           WHERE signal_type=? AND fill_entry_at IS NOT NULL
             AND fill_pnl_usd IS NOT NULL AND fill_exit_at IS NOT NULL
           ORDER BY fill_entry_at""", (fam,)).fetchall()
    pnls = [r[0] for r in rows]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    peak = run = maxdd = 0.0
    for p in pnls:
        run += p
        peak = max(peak, run)
        maxdd = min(maxdd, run - peak)
    s["live"] = {
        "n": len(pnls), "pnl": round(sum(pnls), 0),
        "wins": len(wins),
        "wr": round(100 * len(wins) / len(pnls), 0) if pnls else None,
        "pf": (round(sum(wins) / abs(sum(losses)), 2)
               if losses and sum(losses) != 0 else (99.0 if wins else None)),
        "expectancy": round(sum(pnls) / len(pnls), 1) if pnls else None,
        "rolling10_ev": round(sum(pnls[-10:]) / max(1, len(pnls[-10:])), 1) if pnls else None,
        "rolling10_sum": round(sum(pnls[-10:]), 0) if pnls else None,
        "max_dd": round(maxdd, 0),
        "wilson_lo": wilson_low(len(wins), len(pnls)),
        "avg_win": round(sum(wins) / len(wins), 0) if wins else 0,
    }

    # SHADOW — dispatched + blocked pooled for direction, split for report.
    def shadow(cond):
        r = con.execute(
            f"""SELECT COUNT(*), SUM(outcome='win'), SUM(outcome='loss'),
                       AVG(CASE WHEN outcome IN ('win','loss') THEN pnl_pct END)
                FROM spy_signals
                WHERE signal_type=? AND fill_entry_at IS NULL AND {cond}""",
            (fam,)).fetchone()
        n, w, l = r[0] or 0, r[1] or 0, r[2] or 0
        return {"n": n, "w": w, "l": l, "dec": w + l,
                "wr": round(100 * w / (w + l), 0) if (w + l) else None,
                "avg_pct": round((r[3] or 0) * 100, 2) if r[3] is not None else None,
                "wilson_lo": wilson_low(w, w + l)}
    s["shadow_d"] = shadow("blocked_gate IS NULL")
    s["shadow_b"] = shadow("blocked_gate IS NOT NULL")
    pooled_w = s["shadow_d"]["w"] + s["shadow_b"]["w"]
    pooled_dec = s["shadow_d"]["dec"] + s["shadow_b"]["dec"]
    s["shadow_pooled"] = {"dec": pooled_dec, "w": pooled_w,
                          "wilson_lo": wilson_low(pooled_w, pooled_dec)}

    # Regime split (shadow, decided) — report only.
    s["regimes"] = {
        r[0] or "?": f"{r[1]}W/{r[2]}L"
        for r in con.execute(
            """SELECT regime, SUM(outcome='win'), SUM(outcome='loss')
               FROM spy_signals WHERE signal_type=? AND outcome IN ('win','loss')
               GROUP BY regime""", (fam,))
    }

    # Confidence calibration (shadow decided): hi-conf WR vs lo-conf WR.
    r = con.execute(
        """SELECT
             SUM(confidence>=0.8 AND outcome='win'), SUM(confidence>=0.8 AND outcome IN ('win','loss')),
             SUM(confidence<0.8 AND outcome='win'),  SUM(confidence<0.8 AND outcome IN ('win','loss'))
           FROM spy_signals WHERE signal_type=?""", (fam,)).fetchone()
    hi_w, hi_n, lo_w, lo_n = (r[0] or 0), (r[1] or 0), (r[2] or 0), (r[3] or 0)
    s["calibration"] = {
        "hi_wr": round(100 * hi_w / hi_n, 0) if hi_n else None,
        "lo_wr": round(100 * lo_w / lo_n, 0) if lo_n else None,
        "broken": bool(hi_n >= 10 and lo_n >= 10
                       and (100 * hi_w / hi_n) < (100 * lo_w / lo_n)),
    }
    return s


def decide(cur_tier: str, s: dict) -> tuple:
    """Pure rule engine → (new_tier, risk_scale, reasons, alerts).

    Rules (docs/PORTFOLIO_GOVERNANCE.md is the authoritative copy):
      DEMOTION (checked first — safety over growth):
        D1 live rolling-10 EV < 0 with n≥6      → down 1 tier
        D2 live PF < 0.8 with n≥10              → down to pilot (if above)
        D3 shadow pooled Wilson-lo < 30 @ n≥30  → experimental
      PROMOTION (one step per week, evidence must clear):
        P1 experimental→pilot:   shadow pooled dec≥30 AND Wilson-lo > 40
        P2 pilot→probation:      live n≥5  AND live EV > 0
        P3 probation→production: live n≥15 AND live PF ≥ 1.15 AND Wilson-lo(live) ≥ 40
        P4 production→core:      live n≥40 AND live PF ≥ 1.30 AND max_dd > -2×avg_win...
                                 (|max_dd| ≤ 2×avg_win×3)
      RISK SCALE (within tier, clamp 0.5–1.0; core may reach 1.25):
        start 1.0; −0.25 if rolling-10 EV < half all-time EV (decay);
        −0.25 if |max_dd| > 3×avg_win; floor 0.5.
    """
    live, alerts, reasons = s["live"], [], []
    t = TIERS.index(cur_tier)
    new_t = t

    # ── Demotions ──
    if live["n"] >= 6 and live["rolling10_sum"] is not None and live["rolling10_sum"] < 0:
        new_t = max(0, t - 1)
        reasons.append(f"D1: rolling-10 live EV ${live['rolling10_sum']:+.0f} < 0")
    if live["n"] >= 10 and live["pf"] is not None and live["pf"] < 0.8:
        new_t = min(new_t, TIERS.index("pilot"))
        reasons.append(f"D2: live PF {live['pf']} < 0.8")
    if s["shadow_pooled"]["dec"] >= 30 and s["shadow_pooled"]["wilson_lo"] < 30:
        new_t = 0
        reasons.append(f"D3: shadow Wilson-lo {s['shadow_pooled']['wilson_lo']} < 30 @ n={s['shadow_pooled']['dec']}")

    # ── Promotions (only if no demotion fired) ──
    if new_t == t:
        if cur_tier == "experimental" and s["shadow_pooled"]["dec"] >= 30 \
                and s["shadow_pooled"]["wilson_lo"] > BREAKEVEN_WR:
            new_t = t + 1
            reasons.append(f"P1: shadow Wilson-lo {s['shadow_pooled']['wilson_lo']} > {BREAKEVEN_WR} @ n={s['shadow_pooled']['dec']}")
        elif cur_tier == "pilot" and live["n"] >= 5 and (live["expectancy"] or 0) > 0:
            new_t = t + 1
            reasons.append(f"P2: {live['n']} live fills, EV ${live['expectancy']:+.1f}")
        elif cur_tier == "probation" and live["n"] >= 15 \
                and (live["pf"] or 0) >= 1.15 and live["wilson_lo"] >= BREAKEVEN_WR:
            new_t = t + 1
            reasons.append(f"P3: n={live['n']} PF={live['pf']} Wilson-lo={live['wilson_lo']}")
        elif cur_tier == "production" and live["n"] >= 40 and (live["pf"] or 0) >= 1.30 \
                and abs(live["max_dd"]) <= 6 * max(1, live["avg_win"]):
            new_t = t + 1
            reasons.append(f"P4: n={live['n']} PF={live['pf']} DD={live['max_dd']}")

    # ── Edge-decay alerts (report-only; suggest, never silently act beyond rules) ──
    if live["n"] >= 10 and live["rolling10_ev"] is not None and live["expectancy"] is not None:
        if live["expectancy"] > 0 and live["rolling10_ev"] < 0.5 * live["expectancy"]:
            alerts.append("decay: rolling-10 EV < half of all-time EV")
    if s["calibration"]["broken"]:
        alerts.append("calibration BROKEN: high-conf WR < low-conf WR")
    if live["wr"] is not None and s["shadow_d"]["wr"] is not None \
            and live["n"] >= 8 and live["wr"] < s["shadow_d"]["wr"] - 15:
        alerts.append("live WR trails shadow WR by >15pts (execution slippage?)")

    # ── Risk scale ──
    scale = 1.0
    if "decay: rolling-10 EV < half of all-time EV" in alerts:
        scale -= 0.25
    if live["n"] >= 6 and abs(live["max_dd"]) > 3 * max(1, live["avg_win"]):
        scale -= 0.25
        alerts.append("drawdown > 3× avg win")
    scale = max(0.5, scale)
    if TIERS[new_t] == "core":
        scale = min(1.25, scale + 0.25)

    return TIERS[new_t], round(scale, 2), reasons, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    cur_tiers = SEED.copy()
    if os.path.exists(TIER_FILE):
        try:
            saved = json.load(open(TIER_FILE))
            for f, v in saved.get("families", {}).items():
                cur_tiers[f] = v.get("tier", cur_tiers.get(f, "experimental"))
        except Exception:
            pass

    out = {"as_of": datetime.utcnow().isoformat(), "families": {}}
    lines = ["📊 <b>SPY Portfolio Governor — weekly review</b>",
             f"<i>{datetime.utcnow().strftime('%Y-%m-%d')} · deterministic · "
             f"rules in docs/PORTFOLIO_GOVERNANCE.md</i>", ""]
    changes = []

    for fam, tier in sorted(cur_tiers.items()):
        s = family_stats(con, fam)
        new_tier, scale, reasons, alerts = decide(tier, s)
        out["families"][fam] = {
            "tier": new_tier, "risk_scale": scale,
            "prev_tier": tier, "reasons": reasons, "alerts": alerts,
            "stats": {"live": s["live"], "shadow_d": s["shadow_d"],
                      "shadow_b": s["shadow_b"], "regimes": s["regimes"],
                      "calibration": s["calibration"]},
        }
        arrow = "→" if new_tier != tier else "·"
        rec = ("PROMOTE" if TIERS.index(new_tier) > TIERS.index(tier)
               else "DEMOTE" if TIERS.index(new_tier) < TIERS.index(tier)
               else "MONITOR" if alerts else "HOLD")
        if new_tier != tier:
            changes.append(f"{fam}: {tier} → {new_tier} ({'; '.join(reasons)})")
        live = s["live"]
        lines.append(
            f"<b>{fam[:18]}</b>  {tier}{arrow}{new_tier if new_tier != tier else ''}"
            f"\n  live n={live['n']} pnl=${live['pnl'] or 0:+.0f} "
            f"PF={live['pf'] if live['pf'] is not None else '—'} "
            f"WR={str(live['wr']) + '%' if live['wr'] is not None else '—'}"
            f" | shadow d/b n={s['shadow_d']['dec']}/{s['shadow_b']['dec']}"
            f" Wlo={s['shadow_pooled']['wilson_lo']}"
            f" | scale={scale} | <b>{rec}</b>"
            + (("\n  ⚠️ " + "; ".join(alerts)) if alerts else "")
        )

    lines.append("")
    lines.append("Changes: " + ("; ".join(changes) if changes else "none — no rule cleared"))
    report = "\n".join(lines)
    print(report.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))

    if not args.dry_run:
        os.makedirs(os.path.dirname(TIER_FILE), exist_ok=True)
        tmp = TIER_FILE + ".tmp"
        json.dump(out, open(tmp, "w"), indent=2)
        os.replace(tmp, TIER_FILE)
        print(f"\n[governor] wrote {TIER_FILE}")
        try:
            import daily_summary as ds
            tok, chat = ds.load_telegram()
            if tok:
                ds.send(tok, chat, report)
                print("[governor] Telegram sent")
        except Exception as exc:
            print(f"[governor] Telegram failed: {exc}")


if __name__ == "__main__":
    main()
