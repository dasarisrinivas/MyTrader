"""RAG Daily Updater - Generates daily market summaries and rebuilds embeddings.

This module runs after market close to:
1. Generate a daily market summary (SPY range, volatility, trend, etc.)
2. Analyze any losing trades from the day
3. Rebuild FAISS embeddings with new documents
4. Save everything to the appropriate RAG folders
"""
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from mytrader.rag.rag_storage_manager import (
    RAGStorageManager,
    TradeRecord,
    get_rag_storage,
)


class RAGDailyUpdater:
    """Generates daily market summaries and updates RAG embeddings.
    
    This should be run daily after market close (typically 4:30 PM ET).
    """
    
    def __init__(
        self,
        storage: Optional[RAGStorageManager] = None,
        embedding_builder: Optional[Any] = None,  # Will be EmbeddingBuilder
    ):
        """Initialize the daily updater.
        
        Args:
            storage: RAG storage manager instance
            embedding_builder: Embedding builder for FAISS index
        """
        self.storage = storage or get_rag_storage()
        self.embedding_builder = embedding_builder
        
        logger.info("RAGDailyUpdater initialized")
    
    def run_daily_update(
        self,
        market_data: Dict[str, Any],
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete daily update process.
        
        Args:
            market_data: Market data for the day (OHLCV, indicators, etc.)
            force_rebuild: Force rebuild embeddings even if no new docs
            
        Returns:
            Summary of what was updated
        """
        today = datetime.now(timezone.utc).date()
        logger.info(f"Running daily RAG update for {today}")
        
        results = {
            "date": str(today),
            "daily_summary_saved": False,
            "losing_trades_analyzed": 0,
            "embeddings_rebuilt": False,
            "documents_indexed": 0,
        }
        
        # Step 1: Generate and save daily market summary
        try:
            summary = self._generate_daily_summary(market_data)
            self.storage.save_daily_summary(summary)
            results["daily_summary_saved"] = True
            logger.info("Daily market summary saved")
        except Exception as e:
            logger.error(f"Failed to generate daily summary: {e}")
        
        # Step 2: Analyze losing trades from today
        try:
            start_of_day = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
            end_of_day = start_of_day + timedelta(days=1)
            
            losing_trades = self.storage.load_trades(
                start_date=start_of_day,
                end_date=end_of_day,
                result_filter="LOSS",
            )
            
            for trade in losing_trades:
                analysis = self._analyze_losing_trade(trade, market_data)
                self.storage.save_mistake_note(trade, analysis)
                results["losing_trades_analyzed"] += 1
            
            logger.info(f"Analyzed {results['losing_trades_analyzed']} losing trades")
        except Exception as e:
            logger.error(f"Failed to analyze losing trades: {e}")
        
        # Step 3: Rebuild embeddings if we have an embedding builder
        if self.embedding_builder and (force_rebuild or results["daily_summary_saved"]):
            try:
                documents = self.storage.load_all_documents()
                results["documents_indexed"] = len(documents)
                
                if documents:
                    self.embedding_builder.build_index(documents)
                    results["embeddings_rebuilt"] = True
                    logger.info(f"Rebuilt embeddings with {len(documents)} documents")
            except Exception as e:
                logger.error(f"Failed to rebuild embeddings: {e}")
        
        # Step 4: Check if weekly summary is needed (Friday or forced)
        if today.weekday() == 4:  # Friday
            try:
                week_start = today - timedelta(days=4)
                weekly_summary = self._generate_weekly_summary(market_data, week_start)
                self.storage.save_weekly_summary(
                    weekly_summary,
                    datetime.combine(week_start, datetime.min.time()).replace(tzinfo=timezone.utc)
                )
                logger.info("Weekly summary saved")
            except Exception as e:
                logger.error(f"Failed to generate weekly summary: {e}")
        
        logger.info(f"Daily RAG update completed: {results}")
        return results
    
    def _generate_daily_summary(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a daily market summary document.
        
        Args:
            market_data: Market data for the day
            
        Returns:
            Summary dictionary
        """
        # Extract key metrics from market data
        summary = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            
            # Price action
            "open": market_data.get("open", 0),
            "high": market_data.get("high", 0),
            "low": market_data.get("low", 0),
            "close": market_data.get("close", 0),
            "range": market_data.get("high", 0) - market_data.get("low", 0),
            "range_pct": self._calculate_range_pct(market_data),
            
            # Trend analysis
            "trend": self._determine_trend(market_data),
            "trend_strength": market_data.get("trend_strength", 50),
            
            # Volatility
            "atr": market_data.get("atr", 0),
            "volatility_regime": self._determine_volatility_regime(market_data),
            
            # Key levels
            "pdh": market_data.get("pdh", 0),
            "pdl": market_data.get("pdl", 0),
            "pivot": market_data.get("pivot", 0),
            "weekly_high": market_data.get("weekly_high", 0),
            "weekly_low": market_data.get("weekly_low", 0),
            
            # Indicators
            "rsi_close": market_data.get("rsi", 50),
            "macd_signal": self._get_macd_signal(market_data),
            "ema_alignment": self._get_ema_alignment(market_data),
            
            # Trading activity (from today's trades)
            "trades_taken": market_data.get("trades_taken", 0),
            "win_rate": market_data.get("win_rate", 0),
            "total_pnl": market_data.get("total_pnl", 0),
            
            # Notable events
            "events": market_data.get("events", []),
            "news_impact": market_data.get("news_impact", "none"),
        }
        
        # Generate text summary for embedding
        summary["text_summary"] = self._generate_text_summary(summary)
        
        return summary
    
    def _generate_text_summary(self, summary: Dict[str, Any]) -> str:
        """Generate a human-readable text summary for embedding.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Text summary
        """
        return f"""Market Summary for {summary['date']}:

The market opened at {summary['open']:.2f} and closed at {summary['close']:.2f}.
The day's range was {summary['range']:.2f} points ({summary['range_pct']:.2f}%).

Trend: {summary['trend']} with strength {summary['trend_strength']}/100
Volatility: {summary['volatility_regime']} (ATR: {summary['atr']:.2f})

Key Levels:
- Previous Day High: {summary['pdh']:.2f}
- Previous Day Low: {summary['pdl']:.2f}
- Pivot: {summary['pivot']:.2f}

Indicators:
- RSI at close: {summary['rsi_close']:.1f}
- MACD: {summary['macd_signal']}
- EMA Alignment: {summary['ema_alignment']}

Trading Activity:
- Trades taken: {summary['trades_taken']}
- Win rate: {summary['win_rate']:.1%}
- Total P&L: ${summary['total_pnl']:.2f}

News Impact: {summary['news_impact']}
"""
    
    def _generate_weekly_summary(
        self,
        market_data: Dict[str, Any],
        week_start: datetime,
    ) -> Dict[str, Any]:
        """Generate a weekly market summary.
        
        Args:
            market_data: Latest market data
            week_start: Start date of the week
            
        Returns:
            Weekly summary dictionary
        """
        # Load all trades from the week
        end_date = week_start + timedelta(days=7)
        start_dt = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        
        trades = self.storage.load_trades(
            start_date=start_dt,
            end_date=end_dt,
            limit=500,
        )
        
        # Calculate weekly statistics
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        total_pnl = sum(t.pnl for t in trades)
        
        # Analyze by day of week
        day_stats = {}
        for trade in trades:
            day = trade.day_of_week
            if day not in day_stats:
                day_stats[day] = {"trades": 0, "wins": 0, "pnl": 0}
            day_stats[day]["trades"] += 1
            if trade.result == "WIN":
                day_stats[day]["wins"] += 1
            day_stats[day]["pnl"] += trade.pnl
        
        return {
            "week_start": str(week_start),
            "week_end": str(end_date),
            "weekly_high": market_data.get("weekly_high", 0),
            "weekly_low": market_data.get("weekly_low", 0),
            "weekly_range": market_data.get("weekly_high", 0) - market_data.get("weekly_low", 0),
            "dominant_trend": self._get_dominant_trend(trades),
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "avg_trade_pnl": total_pnl / len(trades) if trades else 0,
            "day_stats": day_stats,
            "common_mistakes": self._identify_common_mistakes(trades),
        }
    
    def _analyze_losing_trade(
        self,
        trade: TradeRecord,
        market_data: Dict[str, Any],
    ) -> str:
        """Analyze why a trade was a loss and generate insights.
        
        Args:
            trade: The losing trade
            market_data: Market data for context
            
        Returns:
            Analysis text
        """
        analysis_points = []
        
        # Check entry against trend
        if trade.action == "BUY" and trade.market_trend == "DOWNTREND":
            analysis_points.append("⚠️ **Counter-trend entry**: Bought in a downtrend")
        elif trade.action == "SELL" and trade.market_trend == "UPTREND":
            analysis_points.append("⚠️ **Counter-trend entry**: Sold in an uptrend")
        
        # Check volatility
        if trade.volatility_regime == "HIGH" and trade.atr > 2.0:
            analysis_points.append(f"⚠️ **High volatility**: ATR was {trade.atr:.2f}, consider wider stops")
        elif trade.volatility_regime == "LOW":
            analysis_points.append("⚠️ **Low volatility**: Market may have been choppy")
        
        # Check level proximity
        if trade.action == "BUY" and abs(trade.price_vs_pdh_pct) < 0.3:
            analysis_points.append("⚠️ **Near resistance**: Bought too close to PDH")
        elif trade.action == "SELL" and abs(trade.price_vs_pdl_pct) < 0.3:
            analysis_points.append("⚠️ **Near support**: Sold too close to PDL")
        
        # Check RSI
        if trade.action == "BUY" and trade.rsi > 70:
            analysis_points.append(f"⚠️ **Overbought**: RSI was {trade.rsi:.1f} at entry")
        elif trade.action == "SELL" and trade.rsi < 30:
            analysis_points.append(f"⚠️ **Oversold**: RSI was {trade.rsi:.1f} at entry")
        
        # Check LLM confidence
        if trade.llm_confidence < 60:
            analysis_points.append(f"⚠️ **Low confidence**: LLM confidence was only {trade.llm_confidence:.0f}%")
        
        # Check time of day
        if trade.time_of_day == "OPEN":
            analysis_points.append("⚠️ **Opening volatility**: Trade taken during market open")
        
        # Check duration
        if trade.duration_minutes < 5:
            analysis_points.append(f"⚠️ **Quick exit**: Trade lasted only {trade.duration_minutes:.1f} minutes")
        
        # Check blocked filters
        if trade.filters_blocked:
            analysis_points.append(f"⚠️ **Overridden filters**: {', '.join(trade.filters_blocked)}")
        
        if not analysis_points:
            analysis_points.append("No obvious issues detected. May have been market noise or black swan event.")
        
        return "\n".join(analysis_points)
    
    def _calculate_range_pct(self, market_data: Dict[str, Any]) -> float:
        """Calculate the daily range as a percentage."""
        high = market_data.get("high", 0)
        low = market_data.get("low", 0)
        open_price = market_data.get("open", 1)
        
        if open_price > 0:
            return ((high - low) / open_price) * 100
        return 0
    
    def _determine_trend(self, market_data: Dict[str, Any]) -> str:
        """Determine the market trend."""
        ema_9 = market_data.get("ema_9", 0)
        ema_20 = market_data.get("ema_20", 0)
        close = market_data.get("close", 0)
        
        if close > ema_9 > ema_20:
            return "UPTREND"
        elif close < ema_9 < ema_20:
            return "DOWNTREND"
        else:
            return "CHOP"
    
    def _determine_volatility_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine the volatility regime."""
        atr = market_data.get("atr", 0)
        atr_avg = market_data.get("atr_20_avg", 1)
        
        ratio = atr / atr_avg if atr_avg > 0 else 1
        
        if ratio > 1.3:
            return "HIGH"
        elif ratio < 0.7:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _get_macd_signal(self, market_data: Dict[str, Any]) -> str:
        """Get MACD signal description."""
        macd_hist = market_data.get("macd_hist", 0)
        macd_hist_prev = market_data.get("macd_hist_prev", 0)
        
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            return "BULLISH_INCREASING"
        elif macd_hist > 0:
            return "BULLISH_DECREASING"
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            return "BEARISH_INCREASING"
        else:
            return "BEARISH_DECREASING"
    
    def _get_ema_alignment(self, market_data: Dict[str, Any]) -> str:
        """Get EMA alignment description."""
        ema_9 = market_data.get("ema_9", 0)
        ema_20 = market_data.get("ema_20", 0)
        ema_50 = market_data.get("ema_50", 0)
        
        if ema_9 > ema_20 > ema_50:
            return "BULLISH_STACKED"
        elif ema_9 < ema_20 < ema_50:
            return "BEARISH_STACKED"
        else:
            return "MIXED"
    
    def _get_dominant_trend(self, trades: List[TradeRecord]) -> str:
        """Get the dominant trend from a list of trades."""
        if not trades:
            return "UNKNOWN"
        
        trends = [t.market_trend for t in trades if t.market_trend]
        if not trends:
            return "UNKNOWN"
        
        # Count occurrences
        from collections import Counter
        trend_counts = Counter(trends)
        return trend_counts.most_common(1)[0][0]
    
    def _identify_common_mistakes(self, trades: List[TradeRecord]) -> List[str]:
        """Identify common mistakes from a list of trades."""
        mistakes = []
        losing_trades = [t for t in trades if t.result == "LOSS"]
        
        if not losing_trades:
            return ["No losing trades this week!"]
        
        # Counter-trend trades
        counter_trend = sum(
            1 for t in losing_trades 
            if (t.action == "BUY" and t.market_trend == "DOWNTREND") or
               (t.action == "SELL" and t.market_trend == "UPTREND")
        )
        if counter_trend > len(losing_trades) * 0.3:
            mistakes.append(f"Counter-trend entries: {counter_trend}/{len(losing_trades)} losses")
        
        # Near level entries
        near_level = sum(
            1 for t in losing_trades
            if abs(t.price_vs_pdh_pct) < 0.3 or abs(t.price_vs_pdl_pct) < 0.3
        )
        if near_level > len(losing_trades) * 0.3:
            mistakes.append(f"Entries near key levels: {near_level}/{len(losing_trades)} losses")
        
        # Low confidence trades
        low_conf = sum(1 for t in losing_trades if t.llm_confidence < 60)
        if low_conf > len(losing_trades) * 0.3:
            mistakes.append(f"Low confidence entries: {low_conf}/{len(losing_trades)} losses")
        
        return mistakes if mistakes else ["No clear patterns in losses"]


def create_daily_updater(storage: Optional[RAGStorageManager] = None) -> RAGDailyUpdater:
    """Factory function to create a RAGDailyUpdater.
    
    Args:
        storage: Optional storage manager
        
    Returns:
        RAGDailyUpdater instance
    """
    return RAGDailyUpdater(storage=storage)
