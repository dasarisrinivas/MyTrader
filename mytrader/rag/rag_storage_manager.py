"""RAG Storage Manager - Handles trade logging and document management via AWS S3.

This module manages the entire RAG data infrastructure:
- Saves trade logs with full metadata to S3
- Manages static and dynamic documents in S3
- Provides document retrieval for RAG queries
- Uses CST (Central Standard Time) for all timestamps

All data is stored in AWS S3:
- Bucket: rag-bot-storage
- Prefix: spy-futures-bot/

S3 Key Structure:
    spy-futures-bot/trade-logs/{YYYY-MM-DD}/{timestamp}_{trade_id}.json
    spy-futures-bot/daily-summaries/{YYYY-MM-DD}/market_summary.json
    spy-futures-bot/weekly-summaries/{YYYY-Www}/weekly_summary.json
    spy-futures-bot/mistake-notes/{YYYY-MM-DD}/{timestamp}_{trade_id}.md
    spy-futures-bot/docs-static/{category}/{filename}
    spy-futures-bot/docs-dynamic/{category}/{filename}
    spy-futures-bot/vectors/index.faiss
    spy-futures-bot/vectors/metadata.pkl
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from loguru import logger

# Import S3 storage
from .s3_storage import S3Storage, S3StorageWithCache, get_s3_storage, S3StorageError

# Import CST utilities
try:
    from ..utils.timezone_utils import now_cst, today_cst, format_cst, CST
except ImportError:
    # Fallback if utils not available
    from zoneinfo import ZoneInfo
    CST = ZoneInfo("America/Chicago")
    def now_cst():
        return datetime.now(CST)
    def today_cst():
        return datetime.now(CST).strftime("%Y-%m-%d")
    def format_cst(dt, fmt="%Y-%m-%d %H:%M:%S CST"):
        if dt is None:
            return "N/A"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc).astimezone(CST)
        return dt.strftime(fmt)


def _safe_parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO timestamps and ensure timezone-aware UTC datetimes."""
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return now_cst()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


@dataclass
class TradeRecord:
    """Complete trade record for RAG storage."""
    # Identification
    trade_id: str
    timestamp: str
    
    # Trade details
    action: str  # BUY, SELL
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int = 1
    
    # Risk management
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Indicators at entry
    rsi: float = 50.0
    macd_hist: float = 0.0
    ema_9: float = 0.0
    ema_20: float = 0.0
    atr: float = 0.0
    
    # Market context
    pdh: float = 0.0  # Previous day high
    pdl: float = 0.0  # Previous day low
    weekly_high: float = 0.0
    weekly_low: float = 0.0
    pivot: float = 0.0
    price_vs_pdh_pct: float = 0.0
    price_vs_pdl_pct: float = 0.0
    
    # LLM decision
    llm_action: str = ""
    llm_confidence: float = 0.0
    llm_reasoning: str = ""
    
    # RAG context
    rag_docs_used: List[str] = field(default_factory=list)
    rag_similarity_scores: List[float] = field(default_factory=list)
    
    # Rule engine signals
    rule_engine_signal: str = ""
    rule_engine_score: float = 0.0
    filters_passed: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    
    # Result (filled after exit)
    result: str = ""  # WIN, LOSS, BREAKEVEN
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_minutes: float = 0.0
    exit_reason: str = ""  # TP_HIT, SL_HIT, MANUAL, TRAILING
    
    # Market conditions
    market_trend: str = ""  # UPTREND, DOWNTREND, CHOP
    volatility_regime: str = ""  # HIGH, MEDIUM, LOW
    time_of_day: str = ""  # OPEN, MIDDAY, CLOSE
    day_of_week: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RAGStorageManager:
    """Manages RAG data storage, retrieval, and organization via AWS S3.
    
    S3 Key Structure:
    spy-futures-bot/
        trade-logs/{YYYY-MM-DD}/{timestamp}_{trade_id}.json
        daily-summaries/{YYYY-MM-DD}/market_summary.json
        weekly-summaries/{YYYY-Www}/weekly_summary.json
        mistake-notes/{YYYY-MM-DD}/{timestamp}_{trade_id}.md
        docs-static/{category}/{filename}
        docs-dynamic/{category}/{filename}
        vectors/index.faiss
        vectors/metadata.pkl
    """
    
    def __init__(
        self,
        bucket_name: str = "rag-bot-storage-897729113303",
        prefix: str = "spy-futures-bot/",
    ):
        """Initialize RAG storage manager with S3 backend.
        
        Args:
            bucket_name: S3 bucket name
            prefix: Key prefix for all objects
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        
        # Initialize S3 storage
        try:
            self.s3 = S3StorageWithCache(bucket_name=bucket_name, prefix=prefix)
            logger.info(f"RAGStorageManager initialized with S3: s3://{bucket_name}/{prefix}")
        except S3StorageError as e:
            logger.error(f"Failed to initialize S3 storage: {e}")
            raise
    
    # =========================================================================
    # Trade Logging
    # =========================================================================
    
    def save_trade(self, trade: TradeRecord) -> str:
        """Save a trade record to S3.
        
        Args:
            trade: TradeRecord to save
            
        Returns:
            S3 key of saved file
        """
        # Parse timestamp to get folder path
        try:
            ts = _safe_parse_timestamp(trade.timestamp)
        except Exception:
            ts = now_cst()
        
        # Generate S3 key: trade-logs/YYYY-MM-DD/HHMMSS_tradeid.json
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H%M%S")
        key = f"trade-logs/{date_str}/{time_str}_{trade.trade_id}.json"
        
        # Save trade to S3
        full_key = self.s3.save_to_s3(key, trade.to_dict())
        
        logger.info(f"Saved trade to S3: {key}")
        return full_key
    
    def load_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        result_filter: Optional[str] = None,  # WIN, LOSS
        action_filter: Optional[str] = None,  # BUY, SELL
        limit: int = 100,
    ) -> List[TradeRecord]:
        """Load trades from S3 storage with optional filters.
        
        Args:
            start_date: Only load trades after this date
            end_date: Only load trades before this date
            result_filter: Filter by result (WIN, LOSS)
            action_filter: Filter by action (BUY, SELL)
            limit: Maximum number of trades to return
            
        Returns:
            List of TradeRecord objects
        """
        trades = []
        
        # List all trade log keys
        keys = self.s3.list_keys("trade-logs/", max_keys=limit * 3)
        
        # Sort by key (reverse for most recent first)
        keys.sort(reverse=True)
        
        for key in keys:
            if len(trades) >= limit:
                break
            
            try:
                # Load trade data from S3
                data = self.s3.read_from_s3(key)
                if data is None:
                    continue
                    
                trade = TradeRecord.from_dict(data)
                
                # Apply filters
                trade_ts = _safe_parse_timestamp(trade.timestamp)
                
                if start_date and trade_ts < start_date:
                    continue
                if end_date and trade_ts > end_date:
                    continue
                if result_filter and trade.result != result_filter:
                    continue
                if action_filter and trade.action != action_filter:
                    continue
                
                trades.append(trade)
                
            except Exception as e:
                logger.warning(f"Failed to load trade {key}: {e}")
        
        return trades

    def get_recent_trade_records(
        self,
        max_age_minutes: int = 30,
        limit: int = 50,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        result_filter: Optional[str] = None,
        action_filter: Optional[str] = None,
    ) -> List[TradeRecord]:
        """Fetch recent trades using the cached S3 wrapper when available."""
        if not hasattr(self.s3, "get_recent_trade_logs"):
            return self.load_trades(
                start_date=start_date,
                end_date=end_date,
                result_filter=result_filter,
                action_filter=action_filter,
                limit=limit,
            )

        window_minutes = max_age_minutes
        now_ts = datetime.now(timezone.utc)
        if start_date and end_date:
            window_minutes = max(
                window_minutes,
                int((end_date - start_date).total_seconds() / 60) + 5,
            )
        elif start_date:
            window_minutes = max(
                window_minutes,
                int((now_ts - start_date).total_seconds() / 60) + 5,
            )

        raw_trades = self.s3.get_recent_trade_logs(
            max_age_minutes=window_minutes,
            max_results=max(limit * 2, limit + 10),
        )

        records: List[TradeRecord] = []
        for trade_data in raw_trades.values():
            try:
                record = TradeRecord.from_dict(trade_data)
            except Exception as e:
                logger.debug(f"Skipping malformed trade log in cache: {e}")
                continue

            ts = _safe_parse_timestamp(record.timestamp)
            if start_date and ts < start_date:
                continue
            if end_date and ts > end_date:
                continue
            if result_filter and record.result.upper() != result_filter.upper():
                continue
            if action_filter and record.action.upper() != action_filter.upper():
                continue

            records.append(record)

        records.sort(key=lambda t: _safe_parse_timestamp(t.timestamp), reverse=True)
        return records[:limit]
    
    def get_similar_trades(
        self,
        action: str,
        market_trend: str,
        volatility_regime: str,
        price_near_pdh: bool = False,
        price_near_pdl: bool = False,
        limit: int = 5,
    ) -> List[TradeRecord]:
        """Find similar historical trades based on conditions.
        
        Args:
            action: BUY or SELL
            market_trend: UPTREND, DOWNTREND, CHOP
            volatility_regime: HIGH, MEDIUM, LOW
            price_near_pdh: True if price is near PDH
            price_near_pdl: True if price is near PDL
            limit: Max trades to return
            
        Returns:
            List of similar TradeRecord objects
        """
        all_trades = self.load_trades(limit=500)
        
        similar = []
        for trade in all_trades:
            score = 0
            
            # Match action
            if trade.action == action:
                score += 3
            
            # Match trend
            if trade.market_trend == market_trend:
                score += 2
            
            # Match volatility
            if trade.volatility_regime == volatility_regime:
                score += 1
            
            # Match level proximity
            if price_near_pdh and abs(trade.price_vs_pdh_pct) < 0.5:
                score += 2
            if price_near_pdl and abs(trade.price_vs_pdl_pct) < 0.5:
                score += 2
            
            if score >= 3:  # Minimum similarity threshold
                similar.append((score, trade))
        
        # Sort by score and return top matches
        similar.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in similar[:limit]]
    
    def get_trade_stats(
        self,
        days: int = 30,
        action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregated trade statistics.
        
        Args:
            days: Number of days to analyze
            action: Optional filter by action
            
        Returns:
            Statistics dictionary
        """
        start_date = now_cst() - timedelta(days=days)
        trades = self.load_trades(start_date=start_date, action_filter=action, limit=1000)
        
        if not trades:
            return {"total": 0, "win_rate": 0.0, "avg_pnl": 0.0}
        
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        total_pnl = sum(t.pnl for t in trades)
        
        return {
            "total": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades) if trades else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades) if trades else 0.0,
            "avg_duration_min": sum(t.duration_minutes for t in trades) / len(trades) if trades else 0.0,
        }
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    def save_daily_summary(self, summary: Dict[str, Any], date: Optional[datetime] = None) -> str:
        """Save a daily market summary document to S3.
        
        Args:
            summary: Market summary data
            date: Date for the summary (default: today)
            
        Returns:
            S3 key of saved file
        """
        if date is None:
            date = now_cst()
        
        # Add metadata
        summary["_generated_at"] = now_cst().isoformat()
        summary["_date"] = date.strftime("%Y-%m-%d")
        
        # Save to S3: daily-summaries/YYYY-MM-DD/market_summary.json
        key = f"daily-summaries/{date.strftime('%Y-%m-%d')}/market_summary.json"
        full_key = self.s3.save_to_s3(key, summary)
        
        logger.info(f"Saved daily summary to S3: {key}")
        return full_key
    
    def save_weekly_summary(self, summary: Dict[str, Any], week_start: datetime) -> str:
        """Save a weekly market summary document to S3.
        
        Args:
            summary: Weekly summary data
            week_start: Start date of the week
            
        Returns:
            S3 key of saved file
        """
        summary["_generated_at"] = now_cst().isoformat()
        summary["_week_start"] = week_start.strftime("%Y-%m-%d")
        
        # Save to S3: weekly-summaries/YYYY-Www/weekly_summary.json
        key = f"weekly-summaries/{week_start.strftime('%Y-W%W')}/weekly_summary.json"
        full_key = self.s3.save_to_s3(key, summary)
        
        logger.info(f"Saved weekly summary to S3: {key}")
        return full_key
    
    def save_mistake_note(self, trade: TradeRecord, analysis: str) -> str:
        """Save a mistake analysis note for a losing trade to S3.
        
        Args:
            trade: The losing trade
            analysis: Analysis text
            
        Returns:
            S3 key of saved file
        """
        try:
            ts = _safe_parse_timestamp(trade.timestamp)
        except Exception:
            ts = now_cst()
        
        content = f"""# Mistake Analysis – {ts.strftime('%Y-%m-%d %H:%M')} CST

## Trade Details
- **Action:** {trade.action}
- **Entry:** {trade.entry_price}
- **Exit:** {trade.exit_price}
- **P&L:** ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)
- **Duration:** {trade.duration_minutes:.1f} minutes

## Market Context
- **Trend:** {trade.market_trend}
- **Volatility:** {trade.volatility_regime}
- **PDH/PDL:** {trade.pdh:.2f} / {trade.pdl:.2f}
- **Price vs PDH:** {trade.price_vs_pdh_pct:.2f}%
- **Price vs PDL:** {trade.price_vs_pdl_pct:.2f}%

## Indicators
- **RSI:** {trade.rsi:.1f}
- **MACD Hist:** {trade.macd_hist:.4f}
- **ATR:** {trade.atr:.2f}

## LLM Decision
- **Confidence:** {trade.llm_confidence:.0f}%
- **Reasoning:** {trade.llm_reasoning}

## Analysis
{analysis}

## Filters
- **Passed:** {', '.join(trade.filters_passed) or 'None'}
- **Blocked:** {', '.join(trade.filters_blocked) or 'None'}

## Lessons Learned
- [To be filled by daily review]

---
*Auto-generated by RAG System*
"""
        
        # Save to S3: mistake-notes/YYYY-MM-DD/HHMMSS_tradeid.md
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H%M%S")
        key = f"mistake-notes/{date_str}/{time_str}_{trade.trade_id}.md"
        
        full_key = self.s3.save_to_s3(key, content, content_type="text/markdown")
        
        logger.info(f"Saved mistake note to S3: {key}")
        return full_key
    
    def load_all_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load all documents from S3 for embedding.
        
        Returns:
            List of (doc_id, content, metadata) tuples
        """
        documents = []
        
        # Load static docs from S3
        static_categories = ["market_structure", "contract_specs", "glossary", 
                           "indicator_definitions", "strategy_rules"]
        for category in static_categories:
            keys = self.s3.list_keys(f"docs-static/{category}/")
            for key in keys:
                try:
                    content = self.s3.read_from_s3(key, as_json=False)
                    if content:
                        doc_id = f"static:{key}"
                        metadata = {
                            "type": "static",
                            "category": category,
                            "s3_key": key,
                        }
                        documents.append((doc_id, content, metadata))
                except Exception as e:
                    logger.warning(f"Failed to load static doc {key}: {e}")
        
        # Load dynamic docs from S3
        dynamic_categories = ["daily", "weekly", "mistakes"]
        for category in dynamic_categories:
            if category == "daily":
                keys = self.s3.list_keys("daily-summaries/")
            elif category == "weekly":
                keys = self.s3.list_keys("weekly-summaries/")
            else:
                keys = self.s3.list_keys("mistake-notes/")
            
            for key in keys:
                try:
                    # For JSON files, read as JSON then convert to string
                    if key.endswith(".json"):
                        data = self.s3.read_from_s3(key, as_json=True)
                        if data:
                            content = json.dumps(data, indent=2)
                    else:
                        content = self.s3.read_from_s3(key, as_json=False)
                    
                    if content:
                        doc_id = f"dynamic:{key}"
                        metadata = {
                            "type": "dynamic",
                            "category": category,
                            "s3_key": key,
                        }
                        documents.append((doc_id, content, metadata))
                except Exception as e:
                    logger.warning(f"Failed to load dynamic doc {key}: {e}")
        
        # Load recent trades as documents
        recent_trades = self.get_recent_trade_records(limit=50)
        for trade in recent_trades:
            doc_id = f"trade:{trade.trade_id}"
            content = self._trade_to_document(trade)
            metadata = {
                "type": "trade",
                "action": trade.action,
                "result": trade.result,
                "timestamp": trade.timestamp,
            }
            documents.append((doc_id, content, metadata))
        
        logger.info(f"Loaded {len(documents)} documents for embedding")
        return documents
    
    def _trade_to_document(self, trade: TradeRecord) -> str:
        """Convert a trade record to a searchable document.
        
        Args:
            trade: TradeRecord to convert
            
        Returns:
            Document text
        """
        return f"""Trade: {trade.action} at {trade.entry_price}
Result: {trade.result} with P&L ${trade.pnl:.2f}
Market: {trade.market_trend} trend, {trade.volatility_regime} volatility
Time: {trade.time_of_day}, {trade.day_of_week}
Indicators: RSI={trade.rsi:.1f}, MACD={trade.macd_hist:.4f}, ATR={trade.atr:.2f}
Levels: PDH={trade.pdh:.2f}, PDL={trade.pdl:.2f}
LLM Confidence: {trade.llm_confidence:.0f}%
Reasoning: {trade.llm_reasoning}
Filters Passed: {', '.join(trade.filters_passed)}
Exit: {trade.exit_reason} after {trade.duration_minutes:.1f} minutes"""
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def get_latest_daily_summary(self) -> Optional[Dict[str, Any]]:
        """Get the most recent daily market summary from S3.
        
        Returns:
            Summary dict or None if not found
        """
        keys = self.s3.list_keys("daily-summaries/")
        if not keys:
            return None
        
        # Find the most recent summary
        keys.sort(reverse=True)
        for key in keys:
            if "market_summary.json" in key:
                return self.s3.read_from_s3(key)
        
        return None
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Remove data older than specified days from S3.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of objects removed
        """
        cutoff = now_cst() - timedelta(days=days_to_keep)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        removed = 0
        
        # Clean up old trades
        keys = self.s3.list_keys("trade-logs/")
        for key in keys:
            try:
                # Extract date from key: trade-logs/YYYY-MM-DD/...
                parts = key.split("/")
                if len(parts) >= 3:
                    date_str = parts[-2]
                    if date_str < cutoff_str:
                        if self.s3.delete_from_s3(key):
                            removed += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {key}: {e}")
        
        logger.info(f"Removed {removed} old objects from S3")
        return removed
    
    def save_static_doc(self, category: str, filename: str, content: str) -> str:
        """Save a static reference document to S3.
        
        Args:
            category: Document category (e.g., 'market_structure', 'glossary')
            filename: Document filename
            content: Document content
            
        Returns:
            S3 key
        """
        key = f"docs-static/{category}/{filename}"
        return self.s3.save_to_s3(key, content)
    
    def save_dynamic_doc(self, category: str, filename: str, content: str) -> str:
        """Save a dynamic document to S3.
        
        Args:
            category: Document category
            filename: Document filename
            content: Document content
            
        Returns:
            S3 key
        """
        key = f"docs-dynamic/{category}/{filename}"
        return self.s3.save_to_s3(key, content)


# Create singleton instance
_storage_manager: Optional[RAGStorageManager] = None


def get_rag_storage(
    bucket_name: str = "rag-bot-storage-897729113303",
    prefix: str = "spy-futures-bot/",
) -> RAGStorageManager:
    """Get the singleton RAG storage manager with S3 backend.
    
    Args:
        bucket_name: S3 bucket name
        prefix: Key prefix for all objects
        
    Returns:
        RAGStorageManager instance
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = RAGStorageManager(bucket_name=bucket_name, prefix=prefix)
    return _storage_manager
