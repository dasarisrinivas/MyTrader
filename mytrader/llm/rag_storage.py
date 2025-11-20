"""
Storage module for RAG (Retrieval Augmented Generation) system.
Persists trades and market snapshots to SQLite for historical retrieval.
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    uuid: str
    timestamp_utc: str
    contract_month: str
    entry_price: float
    entry_qty: int
    exit_price: Optional[float]
    exit_qty: Optional[int]
    pnl: Optional[float]
    fees: float
    hold_seconds: Optional[int]
    decision_features: Dict
    decision_rationale: Dict

class RAGStorage:
    def __init__(self, db_path: str = "data/rag_storage.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                uuid TEXT PRIMARY KEY,
                timestamp_utc TEXT,
                contract_month TEXT,
                entry_price REAL,
                entry_qty INTEGER,
                exit_price REAL,
                exit_qty INTEGER,
                pnl REAL,
                fees REAL,
                hold_seconds INTEGER,
                decision_features TEXT,
                decision_rationale TEXT,
                
                -- Bucketing columns for fast retrieval
                volatility_bucket TEXT,
                time_of_day_bucket TEXT,
                signal_type TEXT
            )
        """)
        
        # Market snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                timestamp_utc TEXT PRIMARY KEY,
                ohlcv TEXT,
                vwap REAL,
                volatility REAL,
                indicators TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def save_trade(self, trade: TradeRecord, buckets: Dict[str, str] = None):
        """Save a completed trade record."""
        if buckets is None:
            buckets = {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO trades (
                    uuid, timestamp_utc, contract_month, entry_price, entry_qty,
                    exit_price, exit_qty, pnl, fees, hold_seconds,
                    decision_features, decision_rationale,
                    volatility_bucket, time_of_day_bucket, signal_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.uuid,
                trade.timestamp_utc,
                trade.contract_month,
                trade.entry_price,
                trade.entry_qty,
                trade.exit_price,
                trade.exit_qty,
                trade.pnl,
                trade.fees,
                trade.hold_seconds,
                json.dumps(trade.decision_features),
                json.dumps(trade.decision_rationale),
                buckets.get('volatility'),
                buckets.get('time_of_day'),
                buckets.get('signal_type')
            ))
            conn.commit()
            logger.info(f"Saved trade {trade.uuid} to RAG storage")
        except Exception as e:
            logger.error(f"Failed to save trade {trade.uuid}: {e}")
        finally:
            conn.close()

    def save_snapshot(self, timestamp: str, data: Dict):
        """Save a market snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO market_snapshots (
                    timestamp_utc, ohlcv, vwap, volatility, indicators
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                json.dumps(data.get('ohlcv', {})),
                data.get('vwap'),
                data.get('volatility'),
                json.dumps(data.get('indicators', {}))
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
        finally:
            conn.close()

    def retrieve_similar_trades(self, buckets: Dict[str, str], limit: int = 5) -> List[Dict]:
        """
        Retrieve similar trades based on buckets.
        buckets: {'volatility': 'HIGH', 'time_of_day': 'MORNING', 'signal_type': 'TREND'}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if 'volatility' in buckets:
            query += " AND volatility_bucket = ?"
            params.append(buckets['volatility'])
            
        if 'time_of_day' in buckets:
            query += " AND time_of_day_bucket = ?"
            params.append(buckets['time_of_day'])
            
        if 'signal_type' in buckets:
            query += " AND signal_type = ?"
            params.append(buckets['signal_type'])
            
        query += " ORDER BY timestamp_utc DESC LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                d = dict(row)
                # Parse JSON fields
                d['decision_features'] = json.loads(d['decision_features']) if d['decision_features'] else {}
                d['decision_rationale'] = json.loads(d['decision_rationale']) if d['decision_rationale'] else {}
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return []
        finally:
            conn.close()

    def get_bucket_stats(self, buckets: Dict[str, str]) -> Dict:
        """Get aggregate stats for a bucket combination."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT 
                COUNT(*) as count,
                AVG(pnl) as avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                AVG(hold_seconds) as avg_hold
            FROM trades 
            WHERE 1=1
        """
        params = []
        
        if 'volatility' in buckets:
            query += " AND volatility_bucket = ?"
            params.append(buckets['volatility'])
            
        if 'time_of_day' in buckets:
            query += " AND time_of_day_bucket = ?"
            params.append(buckets['time_of_day'])
            
        try:
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row and row[0] > 0:
                count, avg_pnl, wins, avg_hold = row
                win_rate = wins / count if count > 0 else 0
                return {
                    "count": count,
                    "avg_pnl": avg_pnl,
                    "win_rate": win_rate,
                    "avg_hold_seconds": avg_hold
                }
            return {"count": 0, "avg_pnl": 0, "win_rate": 0, "avg_hold_seconds": 0}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
        finally:
            conn.close()
