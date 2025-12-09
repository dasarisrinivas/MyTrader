"""RAG Storage Manager - Handles folder structure, trade logging, and document management.

This module manages the entire RAG data infrastructure:
- Auto-creates folder structure on startup
- Saves trade logs with full metadata
- Manages static and dynamic documents
- Provides document retrieval for RAG queries
- Uses CST (Central Standard Time) for all timestamps
"""
import json
import os
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from loguru import logger

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
    """Manages RAG data storage, retrieval, and organization.
    
    Folder Structure:
    /rag_data
        /docs_static           → reference docs (never change)
        /docs_dynamic          → updated regularly
        /trades                → auto-saved trade logs
        /vectors               → FAISS embeddings
    """
    
    def __init__(self, base_path: str = "rag_data"):
        """Initialize RAG storage manager.
        
        Args:
            base_path: Root directory for RAG data
        """
        self.base_path = Path(base_path)
        
        # Define folder structure
        self.folders = {
            "docs_static": self.base_path / "docs_static",
            "docs_static_market": self.base_path / "docs_static" / "market_structure",
            "docs_static_contracts": self.base_path / "docs_static" / "contract_specs",
            "docs_static_glossary": self.base_path / "docs_static" / "glossary",
            "docs_static_indicators": self.base_path / "docs_static" / "indicator_definitions",
            "docs_static_strategies": self.base_path / "docs_static" / "strategy_rules",
            
            "docs_dynamic": self.base_path / "docs_dynamic",
            "docs_dynamic_daily": self.base_path / "docs_dynamic" / "daily_market_summaries",
            "docs_dynamic_weekly": self.base_path / "docs_dynamic" / "weekly_market_summaries",
            "docs_dynamic_mistakes": self.base_path / "docs_dynamic" / "system_mistake_notes",
            
            "trades": self.base_path / "trades",
            "vectors": self.base_path / "vectors",
        }
        
        # Ensure all folders exist
        self._ensure_folders()
        
        logger.info(f"RAGStorageManager initialized at {self.base_path}")
    
    def _ensure_folders(self) -> None:
        """Create all required folders if they don't exist."""
        for name, path in self.folders.items():
            path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured {len(self.folders)} RAG folders exist")
    
    # =========================================================================
    # Trade Logging
    # =========================================================================
    
    def save_trade(self, trade: TradeRecord) -> str:
        """Save a trade record to the appropriate folder.
        
        Args:
            trade: TradeRecord to save
            
        Returns:
            Path to saved file
        """
        # Parse timestamp to get folder path
        try:
            ts = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
        except:
            ts = now_cst()
        
        # Create YYYY/MM folder structure
        year_folder = self.folders["trades"] / str(ts.year)
        month_folder = year_folder / f"{ts.month:02d}"
        month_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"trade_{ts.strftime('%Y_%m_%d_%H%M')}_{trade.trade_id}.json"
        filepath = month_folder / filename
        
        # Save trade
        with open(filepath, "w") as f:
            json.dump(trade.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved trade to {filepath}")
        return str(filepath)
    
    def load_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        result_filter: Optional[str] = None,  # WIN, LOSS
        action_filter: Optional[str] = None,  # BUY, SELL
        limit: int = 100,
    ) -> List[TradeRecord]:
        """Load trades from storage with optional filters.
        
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
        trades_folder = self.folders["trades"]
        
        # Walk through year/month folders
        for year_folder in sorted(trades_folder.iterdir(), reverse=True):
            if not year_folder.is_dir():
                continue
                
            for month_folder in sorted(year_folder.iterdir(), reverse=True):
                if not month_folder.is_dir():
                    continue
                
                for trade_file in sorted(month_folder.glob("trade_*.json"), reverse=True):
                    if len(trades) >= limit:
                        break
                    
                    try:
                        with open(trade_file) as f:
                            data = json.load(f)
                        trade = TradeRecord.from_dict(data)
                        
                        # Apply filters
                        trade_ts = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
                        
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
                        logger.warning(f"Failed to load trade {trade_file}: {e}")
        
        return trades
    
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
        """Save a daily market summary document.
        
        Args:
            summary: Market summary data
            date: Date for the summary (default: today)
            
        Returns:
            Path to saved file
        """
        if date is None:
            date = now_cst()
        
        filename = f"market_summary_{date.strftime('%Y_%m_%d')}.json"
        filepath = self.folders["docs_dynamic_daily"] / filename
        
        # Add metadata
        summary["_generated_at"] = now_cst().isoformat()
        summary["_date"] = date.strftime("%Y-%m-%d")
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved daily summary to {filepath}")
        return str(filepath)
    
    def save_weekly_summary(self, summary: Dict[str, Any], week_start: datetime) -> str:
        """Save a weekly market summary document.
        
        Args:
            summary: Weekly summary data
            week_start: Start date of the week
            
        Returns:
            Path to saved file
        """
        filename = f"weekly_summary_{week_start.strftime('%Y_W%W')}.json"
        filepath = self.folders["docs_dynamic_weekly"] / filename
        
        summary["_generated_at"] = now_cst().isoformat()
        summary["_week_start"] = week_start.strftime("%Y-%m-%d")
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved weekly summary to {filepath}")
        return str(filepath)
    
    def save_mistake_note(self, trade: TradeRecord, analysis: str) -> str:
        """Save a mistake analysis note for a losing trade.
        
        Args:
            trade: The losing trade
            analysis: Analysis text
            
        Returns:
            Path to saved file
        """
        try:
            ts = datetime.fromisoformat(trade.timestamp.replace("Z", "+00:00"))
        except:
            ts = now_cst()
        
        filename = f"mistake_{ts.strftime('%Y_%m_%d')}_{trade.trade_id}.md"
        filepath = self.folders["docs_dynamic_mistakes"] / filename
        
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
        
        with open(filepath, "w") as f:
            f.write(content)
        
        logger.info(f"Saved mistake note to {filepath}")
        return str(filepath)
    
    def load_all_documents(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load all documents for embedding.
        
        Returns:
            List of (doc_id, content, metadata) tuples
        """
        documents = []
        
        # Load static docs (markdown and text)
        for folder_key in ["docs_static_market", "docs_static_contracts", 
                          "docs_static_glossary", "docs_static_indicators",
                          "docs_static_strategies"]:
            folder = self.folders[folder_key]
            for filepath in folder.glob("**/*"):
                if filepath.suffix in [".md", ".txt", ".json"]:
                    try:
                        content = filepath.read_text()
                        doc_id = f"static:{filepath.relative_to(self.base_path)}"
                        metadata = {
                            "type": "static",
                            "category": folder_key.replace("docs_static_", ""),
                            "filename": filepath.name,
                        }
                        documents.append((doc_id, content, metadata))
                    except Exception as e:
                        logger.warning(f"Failed to load {filepath}: {e}")
        
        # Load dynamic docs
        for folder_key in ["docs_dynamic_daily", "docs_dynamic_weekly", 
                          "docs_dynamic_mistakes"]:
            folder = self.folders[folder_key]
            for filepath in folder.glob("**/*"):
                if filepath.suffix in [".md", ".txt", ".json"]:
                    try:
                        content = filepath.read_text()
                        doc_id = f"dynamic:{filepath.relative_to(self.base_path)}"
                        metadata = {
                            "type": "dynamic",
                            "category": folder_key.replace("docs_dynamic_", ""),
                            "filename": filepath.name,
                        }
                        documents.append((doc_id, content, metadata))
                    except Exception as e:
                        logger.warning(f"Failed to load {filepath}: {e}")
        
        # Load recent trades as documents
        recent_trades = self.load_trades(limit=50)
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
        """Get the most recent daily market summary.
        
        Returns:
            Summary dict or None if not found
        """
        folder = self.folders["docs_dynamic_daily"]
        files = sorted(folder.glob("market_summary_*.json"), reverse=True)
        
        if files:
            with open(files[0]) as f:
                return json.load(f)
        return None
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Remove data older than specified days.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of files removed
        """
        cutoff = now_cst() - timedelta(days=days_to_keep)
        removed = 0
        
        # Clean up trades
        for year_folder in self.folders["trades"].iterdir():
            if not year_folder.is_dir():
                continue
            
            try:
                year = int(year_folder.name)
                if year < cutoff.year:
                    shutil.rmtree(year_folder)
                    removed += 1
                    logger.info(f"Removed old trade folder: {year_folder}")
            except ValueError:
                continue
        
        return removed


# Create singleton instance
_storage_manager: Optional[RAGStorageManager] = None


def get_rag_storage() -> RAGStorageManager:
    """Get the singleton RAG storage manager.
    
    Returns:
        RAGStorageManager instance
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = RAGStorageManager()
    return _storage_manager
