"""
Scheduler Emulator for Backtest Mode

Emulates EventBridge scheduled triggers for Agent 1 (nightly data ingestion)
and Agent 4 (nightly learning at 11 PM CST) in backtest mode.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import pytz

from ..utils.logger import logger
from .lambda_wrappers import (
    Agent1DataIngestionWrapper,
    Agent4LearningWrapper,
)


class SchedulerEmulator:
    """Emulates scheduled agent triggers for backtest mode."""
    
    CST = pytz.timezone('America/Chicago')
    
    def __init__(self, artifacts_dir: Path):
        """
        Initialize scheduler emulator.
        
        Args:
            artifacts_dir: Directory for saving artifacts
        """
        self.artifacts_dir = artifacts_dir
        self.agent1_wrapper = Agent1DataIngestionWrapper(artifacts_dir)
        self.agent4_wrapper = Agent4LearningWrapper(artifacts_dir)
    
    def run_nightly_ingestion(self, date: str, raw_trade_data: Optional[list] = None) -> Dict[str, Any]:
        """
        Run Agent 1: Data Ingestion & Feature Builder for a specific date.
        
        In production, this is triggered by EventBridge nightly schedule.
        In backtest, we call it once per day at start-of-day.
        
        Args:
            date: Date string in YYYY-MM-DD format
            raw_trade_data: Optional list of raw trade records for the day
        
        Returns:
            Agent 1 response with processed count and feature keys
        """
        logger.info(f"🔵 Agent 1: Running nightly data ingestion for {date}")
        
        # If no raw data provided, try to load from artifacts or create empty
        if raw_trade_data is None:
            # In backtest, we may not have raw data yet - create empty for now
            raw_trade_data = []
        
        event = {
            'source': 'direct',
            'raw_data': raw_trade_data,
            'date': date
        }
        
        result = self.agent1_wrapper.invoke(event)
        
        logger.info(
            f"✅ Agent 1: Processed {result.get('processed_count', 0)} trades, "
            f"features: {result.get('features_key', 'N/A')}"
        )
        
        return result
    
    def run_nightly_learning(
        self,
        date: str,
        tz: str = "America/Chicago",
        losing_trades: Optional[list] = None,
        daily_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run Agent 4: Strategy Optimization & Learning Agent for a specific date.
        
        In production, this is triggered by EventBridge at 11 PM CST.
        In backtest, we call it once per day at end-of-day (emulating 11 PM CST).
        
        Args:
            date: Date string in YYYY-MM-DD format
            tz: Timezone (default: America/Chicago for CST)
            losing_trades: Optional list of losing trades for the day
            daily_summary: Optional summary of daily performance metrics
        
        Returns:
            Agent 4 response with patterns identified and rules updated
        """
        logger.info(f"🟣 Agent 4: Running nightly learning for {date} (11 PM CST emulation)")
        
        # If no losing trades provided, try to load from artifacts
        if losing_trades is None:
            # Try to load from structured data or create empty
            losing_trades = []
        
        event = {
            'analysis_type': 'daily',
            'date_range': {
                'start': date,
                'end': date
            },
            'losing_trades': losing_trades,
            'daily_summary': daily_summary or {},
            'backtest_date': date,
        }
        
        result = self.agent4_wrapper.invoke(event)
        
        patterns_count = len(result.get('patterns_identified', []))
        rules_count = len(result.get('rules_updated', []))
        
        logger.info(
            f"✅ Agent 4: Identified {patterns_count} patterns, "
            f"updated {rules_count} rules"
        )
        
        return result
    
    def should_run_agent4(self, current_time: datetime, date: str) -> bool:
        """
        Check if Agent 4 should run based on time (11 PM CST).
        
        Args:
            current_time: Current datetime (UTC)
            date: Date string in YYYY-MM-DD format
        
        Returns:
            True if Agent 4 should run
        """
        # Convert to CST
        cst_time = current_time.astimezone(self.CST)
        
        # Check if it's 11 PM or later on the given date
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        if cst_time.date() == target_date and cst_time.hour >= 23:
            return True
        
        # Also allow if we're past the date (end-of-day processing)
        if cst_time.date() > target_date:
            return True
        
        return False
    
    def get_agent1_manifest(self, date: str) -> Optional[Dict[str, Any]]:
        """Load Agent 1 manifest for a date."""
        manifest_path = self.artifacts_dir / date / 'agent1_features_manifest.json'
        if manifest_path.exists():
            import json
            with open(manifest_path) as f:
                return json.load(f)
        return None
    
    def get_agent4_update(self, date: str) -> Optional[Dict[str, Any]]:
        """Load Agent 4 learning update for a date."""
        update_path = self.artifacts_dir / date / 'agent4_learning_update.json'
        if update_path.exists():
            import json
            with open(update_path) as f:
                return json.load(f)
        return None
