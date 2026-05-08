"""Layer-1 adaptive intelligence: per-bucket empirical performance.

The rules engine consults this layer before approving a signal. Each historical
trade is bucketized along four dimensions:

  • signal_type   (EMA21_PB_LONG, ORB_BREAKOUT, PC_RATIO_EXTREME, ...)
  • regime        (TRENDING / CHOPPY / MIXED for MES; TREND_UP / RANGE_BOUND / ...
                   for SPY)
  • time_bucket   (PREMARKET / RTH_OPEN / RTH_MID / RTH_CLOSE / EVENING /
                   OVERNIGHT, in CT)
  • vix_bucket    (LOW < 15 / NORMAL 15-22 / ELEVATED 22-30 / CRISIS > 30 /
                   UNKNOWN if vix unavailable)

For each (signal_type, bot, regime, time_bucket, vix_bucket) tuple we keep:

  • n_trades, n_wins, n_losses
  • sum_pnl (signed dollars)
  • avg_winner / avg_loser (used for adaptive R:R)
  • last_updated

From these we derive *adaptive thresholds* per signal:

  • min_confidence_required — pushed up if the bucket bleeds, down if it pays
  • auto_suppress           — True if win_rate < 35% on >= 10 samples
  • allowed_min_rr          — relaxed below 2.0 if the winner/loser ratio earns it

This file is pure I/O + math. No event loop, no daemon logic — the manager
calls update() periodically and the rules engine calls get_bucket_stats() per
signal.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


CT = ZoneInfo("America/Chicago")


# ─────────────────────────────────────────────────────────────────────────────
# Bucketization
# ─────────────────────────────────────────────────────────────────────────────

# Sample size thresholds — below MIN_SAMPLE the bucket falls through to static
# rules. AT MIN_SAMPLE we start adapting cautiously; at TRUST_SAMPLE we trust
# the empirical signal completely.
MIN_SAMPLE = 5
TRUST_SAMPLE = 10


def bucketize_time(ts_iso: str) -> str:
    """Map an ISO timestamp (any TZ) to a CT trading-session bucket.

    Buckets:
      PREMARKET    03:00–08:29 CT
      RTH_OPEN     08:30–09:59 CT (high volatility — most strategies blow up here)
      RTH_MID      10:00–12:59 CT
      RTH_CLOSE    13:00–15:00 CT (close window — power hour + final 60 min)
      MAINT        16:00–17:00 CT (CME maintenance — should never see signals)
      EVENING      17:00–22:59 CT
      OVERNIGHT    23:00–02:59 CT (illiquid, wide spreads)
      UNKNOWN      parse failure
    """
    if not ts_iso:
        return "UNKNOWN"
    try:
        if ts_iso.endswith("Z"):
            ts_iso = ts_iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ct = dt.astimezone(CT)
        h = ct.hour + ct.minute / 60.0
    except (ValueError, AttributeError):
        return "UNKNOWN"
    if 3.0 <= h < 8.5:
        return "PREMARKET"
    if 8.5 <= h < 10.0:
        return "RTH_OPEN"
    if 10.0 <= h < 13.0:
        return "RTH_MID"
    if 13.0 <= h < 15.0:
        return "RTH_CLOSE"
    if 15.0 <= h < 16.0:
        return "RTH_CLOSE"  # final hour merged into close
    if 16.0 <= h < 17.0:
        return "MAINT"
    if 17.0 <= h < 23.0:
        return "EVENING"
    return "OVERNIGHT"


def bucketize_vix(vix: Optional[float]) -> str:
    """Map a VIX value to a regime bucket. None → UNKNOWN."""
    if vix is None or vix <= 0:
        return "UNKNOWN"
    if vix < 15.0:
        return "LOW"
    if vix < 22.0:
        return "NORMAL"
    if vix < 30.0:
        return "ELEVATED"
    return "CRISIS"


def normalize_regime(raw: str) -> str:
    """Collapse the various regime labels the bots emit into a small alphabet.

    The MES bot emits TRENDING / CHOPPY / MIXED. The SPY bot's rules_v2 emits
    TREND_UP / TREND_DOWN / RANGE_BOUND / BREAKOUT / etc. We keep them
    separate because the *direction* of trend matters for SPY directional
    options but doesn't for an MES futures pullback.
    """
    if not raw:
        return "UNKNOWN"
    return raw.upper().replace(" ", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Storage layer
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS bucket_stats (
    signal_type   TEXT    NOT NULL,
    bot           TEXT    NOT NULL,        -- 'mes' | 'spy_options'
    regime        TEXT    NOT NULL,
    time_bucket   TEXT    NOT NULL,
    vix_bucket    TEXT    NOT NULL,
    n_trades      INTEGER NOT NULL DEFAULT 0,
    n_wins        INTEGER NOT NULL DEFAULT 0,
    n_losses      INTEGER NOT NULL DEFAULT 0,
    sum_pnl       REAL    NOT NULL DEFAULT 0.0,
    sum_winner    REAL    NOT NULL DEFAULT 0.0,
    sum_loser     REAL    NOT NULL DEFAULT 0.0,  -- absolute value of losses
    last_updated  TEXT    NOT NULL,
    PRIMARY KEY (signal_type, bot, regime, time_bucket, vix_bucket)
);
CREATE INDEX IF NOT EXISTS idx_bucket_signal_type ON bucket_stats(signal_type);
CREATE INDEX IF NOT EXISTS idx_bucket_updated     ON bucket_stats(last_updated);

-- Append-only trade events log: every closed trade we've seen, so backfill
-- is idempotent and we never double-count.
CREATE TABLE IF NOT EXISTS trade_events (
    event_id      TEXT    NOT NULL PRIMARY KEY,  -- bot:trade_cycle_id or similar
    bot           TEXT    NOT NULL,
    signal_type   TEXT    NOT NULL,
    regime        TEXT    NOT NULL,
    time_bucket   TEXT    NOT NULL,
    vix_bucket    TEXT    NOT NULL,
    pnl           REAL    NOT NULL,
    is_win        INTEGER NOT NULL,
    entry_time    TEXT    NOT NULL,
    exit_time     TEXT    NOT NULL,
    recorded_at   TEXT    NOT NULL
);
"""


@dataclass
class BucketStats:
    """Empirical performance for one bucket."""
    signal_type: str
    bot: str
    regime: str
    time_bucket: str
    vix_bucket: str
    n_trades: int
    n_wins: int
    n_losses: int
    sum_pnl: float
    sum_winner: float
    sum_loser: float

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0.0

    @property
    def expectancy(self) -> float:
        return self.sum_pnl / self.n_trades if self.n_trades else 0.0

    @property
    def avg_winner(self) -> float:
        return self.sum_winner / self.n_wins if self.n_wins else 0.0

    @property
    def avg_loser(self) -> float:
        # avg_loser is positive (we stored absolute values)
        return self.sum_loser / self.n_losses if self.n_losses else 0.0

    @property
    def edge_ratio(self) -> float:
        """avg_winner / avg_loser. Above 1.0 = winners bigger than losers
        (positive expectancy regardless of WR). Below 1.0 = bleeders."""
        L = self.avg_loser
        return (self.avg_winner / L) if L > 0 else 0.0


@dataclass
class AdaptiveThresholds:
    """What the rules engine should override based on bucket stats."""
    bucket: BucketStats
    n_trades: int
    win_rate: float
    expectancy: float
    edge_ratio: float

    # Behavior signals for the rules engine
    auto_suppress: bool                 # hard REJECT no matter what
    confidence_floor: float             # raise (or lower) the min_confidence
    min_rr_required: float              # raise (or lower) the min R:R
    rationale: str                      # human-readable reason
    confident: bool                     # True if n_trades >= TRUST_SAMPLE


def open_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path, timeout=5.0)
    con.executescript(SCHEMA)
    return con


def upsert_event(
    con: sqlite3.Connection,
    *,
    event_id: str,
    bot: str,
    signal_type: str,
    regime: str,
    time_bucket: str,
    vix_bucket: str,
    pnl: float,
    entry_time: str,
    exit_time: str,
) -> bool:
    """Record one closed trade. Idempotent on event_id. Returns True if new."""
    is_win = 1 if pnl > 0 else 0
    cur = con.execute("SELECT 1 FROM trade_events WHERE event_id = ?", (event_id,))
    if cur.fetchone():
        return False
    con.execute(
        """INSERT INTO trade_events
           (event_id, bot, signal_type, regime, time_bucket, vix_bucket,
            pnl, is_win, entry_time, exit_time, recorded_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (event_id, bot, signal_type, regime, time_bucket, vix_bucket,
         pnl, is_win, entry_time, exit_time, datetime.utcnow().isoformat()),
    )
    # Update bucket aggregate
    abs_pnl = abs(pnl)
    if pnl > 0:
        con.execute(
            """INSERT INTO bucket_stats
               (signal_type, bot, regime, time_bucket, vix_bucket,
                n_trades, n_wins, n_losses, sum_pnl, sum_winner, sum_loser, last_updated)
               VALUES (?,?,?,?,?, 1, 1, 0, ?, ?, 0, ?)
               ON CONFLICT(signal_type, bot, regime, time_bucket, vix_bucket) DO UPDATE SET
                 n_trades   = n_trades + 1,
                 n_wins     = n_wins + 1,
                 sum_pnl    = sum_pnl + excluded.sum_pnl,
                 sum_winner = sum_winner + excluded.sum_winner,
                 last_updated = excluded.last_updated""",
            (signal_type, bot, regime, time_bucket, vix_bucket,
             pnl, pnl, datetime.utcnow().isoformat()),
        )
    else:
        con.execute(
            """INSERT INTO bucket_stats
               (signal_type, bot, regime, time_bucket, vix_bucket,
                n_trades, n_wins, n_losses, sum_pnl, sum_winner, sum_loser, last_updated)
               VALUES (?,?,?,?,?, 1, 0, 1, ?, 0, ?, ?)
               ON CONFLICT(signal_type, bot, regime, time_bucket, vix_bucket) DO UPDATE SET
                 n_trades   = n_trades + 1,
                 n_losses   = n_losses + 1,
                 sum_pnl    = sum_pnl + excluded.sum_pnl,
                 sum_loser  = sum_loser + excluded.sum_loser,
                 last_updated = excluded.last_updated""",
            (signal_type, bot, regime, time_bucket, vix_bucket,
             pnl, abs_pnl, datetime.utcnow().isoformat()),
        )
    con.commit()
    return True


def get_bucket(
    con: sqlite3.Connection,
    *,
    signal_type: str,
    bot: str,
    regime: str,
    time_bucket: str,
    vix_bucket: str,
) -> Optional[BucketStats]:
    """Return stats for an exact bucket, or None if not present."""
    row = con.execute(
        """SELECT signal_type, bot, regime, time_bucket, vix_bucket,
                  n_trades, n_wins, n_losses, sum_pnl, sum_winner, sum_loser
           FROM bucket_stats
           WHERE signal_type = ? AND bot = ? AND regime = ?
             AND time_bucket = ? AND vix_bucket = ?""",
        (signal_type, bot, regime, time_bucket, vix_bucket),
    ).fetchone()
    if not row:
        return None
    return BucketStats(*row)


def get_bucket_with_fallback(
    con: sqlite3.Connection,
    *,
    signal_type: str,
    bot: str,
    regime: str,
    time_bucket: str,
    vix_bucket: str,
) -> Optional[BucketStats]:
    """Return the most-specific bucket with at least MIN_SAMPLE trades,
    falling back through three levels of generality:

      L0: full match (signal_type, bot, regime, time_bucket, vix_bucket)
      L1: drop vix_bucket (less specific)
      L2: drop time_bucket too — just (signal_type, bot, regime)
      L3: just (signal_type, bot)

    Returns None if even L3 has fewer than MIN_SAMPLE.
    """
    # L0
    s = get_bucket(
        con, signal_type=signal_type, bot=bot, regime=regime,
        time_bucket=time_bucket, vix_bucket=vix_bucket,
    )
    if s and s.n_trades >= MIN_SAMPLE:
        return s

    # L1: aggregate over vix_bucket
    row = con.execute(
        """SELECT ?, ?, ?, ?, 'ANY',
                  SUM(n_trades), SUM(n_wins), SUM(n_losses),
                  SUM(sum_pnl), SUM(sum_winner), SUM(sum_loser)
           FROM bucket_stats
           WHERE signal_type = ? AND bot = ? AND regime = ? AND time_bucket = ?""",
        (signal_type, bot, regime, time_bucket,
         signal_type, bot, regime, time_bucket),
    ).fetchone()
    if row and row[5] and row[5] >= MIN_SAMPLE:
        return BucketStats(*row)

    # L2: aggregate over time_bucket and vix_bucket
    row = con.execute(
        """SELECT ?, ?, ?, 'ANY', 'ANY',
                  SUM(n_trades), SUM(n_wins), SUM(n_losses),
                  SUM(sum_pnl), SUM(sum_winner), SUM(sum_loser)
           FROM bucket_stats
           WHERE signal_type = ? AND bot = ? AND regime = ?""",
        (signal_type, bot, regime, signal_type, bot, regime),
    ).fetchone()
    if row and row[5] and row[5] >= MIN_SAMPLE:
        return BucketStats(*row)

    # L3: aggregate over regime, time, vix
    row = con.execute(
        """SELECT ?, ?, 'ANY', 'ANY', 'ANY',
                  SUM(n_trades), SUM(n_wins), SUM(n_losses),
                  SUM(sum_pnl), SUM(sum_winner), SUM(sum_loser)
           FROM bucket_stats
           WHERE signal_type = ? AND bot = ?""",
        (signal_type, bot, signal_type, bot),
    ).fetchone()
    if row and row[5] and row[5] >= MIN_SAMPLE:
        return BucketStats(*row)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive threshold derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_thresholds(
    stats: BucketStats,
    *,
    base_min_confidence: float = 0.50,
    base_min_rr: float = 2.0,
) -> AdaptiveThresholds:
    """Map empirical performance to threshold overrides.

    Decision philosophy:
      • Below 35% WR with N>=10 (TRUST sample) → AUTO-SUPPRESS. The bucket
        is bleeding; no point dressing it up with confidence.
      • 35-45% WR → raise confidence floor to 0.75 (require near-certainty).
      • 45-55% WR → keep base; bucket is roughly random.
      • 55-65% WR → lower confidence floor to 0.45 (give the system more rope).
      • >65% WR → confident signal; lower confidence to 0.40 AND relax R:R to 1.5.
      • If edge_ratio (winner_size / loser_size) >= 2.0 with N>=10, override
        R:R requirement to 1.5 even with mediocre WR — the math still pays.
      • If edge_ratio < 0.6 with N>=10, raise R:R to 2.5 and confidence to 0.70.
    """
    n = stats.n_trades
    wr = stats.win_rate
    er = stats.edge_ratio
    confident = n >= TRUST_SAMPLE

    # Defaults
    auto_suppress = False
    floor = base_min_confidence
    rr = base_min_rr
    notes: List[str] = [f"n={n}, WR={wr*100:.0f}%, edge_ratio={er:.2f}, "
                        f"avg_winner=${stats.avg_winner:.2f}, avg_loser=${stats.avg_loser:.2f}"]

    if confident:
        # Only suppress when the *math* is bad, not just the WR. A 30%-WR
        # trend-follower with 4× winner/loser ratio is still profitable —
        # killing it would destroy the strategy. Suppress only if BOTH WR
        # is poor AND expectancy is negative AND edge ratio is weak.
        if wr < 0.35 and (stats.expectancy < 0 or er < 1.5):
            auto_suppress = True
            notes.append(
                f"AUTO-SUPPRESS: WR<35% AND (expectancy={stats.expectancy:+.2f} OR edge<1.5) on n>=10"
            )
        elif wr < 0.35 and er >= 1.5 and stats.expectancy >= 0:
            # Low WR but math pays. Don't suppress, but raise confidence so
            # we only take the cleanest setups.
            floor = max(floor, 0.65)
            notes.append("Low WR but edge ratio pays — confidence raised, not suppressed")
        elif wr < 0.45:
            floor = max(floor, 0.75)
            rr = max(rr, 2.5)
            notes.append("Bucket bleeding (WR 35-45%) — confidence raised to 0.75, R:R 2.5")
        elif wr >= 0.65:
            floor = min(floor, 0.40)
            rr = min(rr, 1.5)
            notes.append("Bucket strong (WR>=65%) — confidence eased to 0.40, R:R 1.5 ok")
        elif wr >= 0.55:
            floor = min(floor, 0.45)
            notes.append("Bucket pays (WR 55-65%) — confidence eased")
        # 45-55%: leave defaults
    else:
        # Cautious sample size — don't go aggressive in either direction
        if wr < 0.30:
            floor = max(floor, 0.65)
            notes.append(f"Small sample (n={n}) WR<30% — modest tightening")

    # Edge-ratio overlay (works regardless of WR)
    if confident and er >= 2.0 and not auto_suppress:
        rr = min(rr, 1.5)
        notes.append("Edge ratio >=2.0 — R:R 1.5 acceptable (math pays even at low WR)")
    elif confident and 0 < er < 0.6:
        rr = max(rr, 2.5)
        floor = max(floor, 0.70)
        notes.append("Edge ratio <0.6 — winners much smaller than losers; tightening")

    return AdaptiveThresholds(
        bucket=stats,
        n_trades=n,
        win_rate=wr,
        expectancy=stats.expectancy,
        edge_ratio=er,
        auto_suppress=auto_suppress,
        confidence_floor=floor,
        min_rr_required=rr,
        rationale=" | ".join(notes),
        confident=confident,
    )


def adaptive_lookup(
    db_path: str,
    *,
    signal_type: str,
    bot: str,
    regime: str,
    ts_iso: str,
    vix: Optional[float],
    base_min_confidence: float = 0.50,
    base_min_rr: float = 2.0,
) -> Optional[AdaptiveThresholds]:
    """One-shot lookup: returns the thresholds the rules engine should apply
    for this signal, or None if no usable empirical data is available.
    """
    con = open_db(db_path)
    try:
        stats = get_bucket_with_fallback(
            con,
            signal_type=signal_type,
            bot=bot,
            regime=normalize_regime(regime),
            time_bucket=bucketize_time(ts_iso),
            vix_bucket=bucketize_vix(vix),
        )
        if stats is None or stats.n_trades < MIN_SAMPLE:
            return None
        return derive_thresholds(
            stats,
            base_min_confidence=base_min_confidence,
            base_min_rr=base_min_rr,
        )
    finally:
        con.close()


def summary(db_path: str) -> Dict:
    """Top-of-file summary for heartbeat / debugging."""
    if not os.path.exists(db_path):
        return {"buckets": 0, "trades": 0, "top": []}
    con = open_db(db_path)
    try:
        n_buckets = con.execute("SELECT COUNT(*) FROM bucket_stats").fetchone()[0]
        n_trades = con.execute("SELECT SUM(n_trades) FROM bucket_stats").fetchone()[0] or 0
        # Top-3 best-performing and worst-performing buckets
        top = con.execute(
            """SELECT signal_type, bot, regime, time_bucket, vix_bucket,
                      n_trades, n_wins, n_losses, sum_pnl
               FROM bucket_stats
               WHERE n_trades >= 5
               ORDER BY (CAST(n_wins AS REAL)/n_trades) DESC, n_trades DESC
               LIMIT 5"""
        ).fetchall()
        bot = con.execute(
            """SELECT signal_type, bot, regime, time_bucket, vix_bucket,
                      n_trades, n_wins, n_losses, sum_pnl
               FROM bucket_stats
               WHERE n_trades >= 5
               ORDER BY (CAST(n_wins AS REAL)/n_trades) ASC, n_trades DESC
               LIMIT 5"""
        ).fetchall()
        return {
            "buckets": n_buckets,
            "trades": n_trades,
            "best": top,
            "worst": bot,
        }
    finally:
        con.close()
