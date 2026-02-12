"""
Artifact Logging System for Backtest

Strict artifact logging and validation to prove agents ran and outputs exist.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..utils.logger import logger


class ArtifactLogger:
    """Manages artifact logging and validation for backtest."""
    
    def __init__(self, artifacts_dir: Path):
        """
        Initialize artifact logger.
        
        Args:
            artifacts_dir: Base directory for artifacts (e.g., artifacts/backtest)
        """
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def get_date_dir(self, date: str) -> Path:
        """Get directory for a specific date."""
        date_dir = self.artifacts_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir
    
    def log_trade(self, date: str, trade: Dict[str, Any]) -> Path:
        """
        Log a trade execution.
        
        Args:
            date: Date string in YYYY-MM-DD format
            trade: Trade record with execution details
        
        Returns:
            Path to the trades.ndjson file
        """
        date_dir = self.get_date_dir(date)
        trades_file = date_dir / 'trades.ndjson'
        
        trade_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **trade
        }
        
        with open(trades_file, 'a') as f:
            f.write(json.dumps(trade_record, default=str) + '\n')
        
        return trades_file
    
    def validate_artifacts(self, date: str) -> Dict[str, Any]:
        """
        Validate that all required artifacts exist for a date.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            Validation result with missing artifacts list
        """
        date_dir = self.get_date_dir(date)
        
        required_artifacts = {
            'agent1_features_manifest.json': 'Agent 1 features manifest',
            'agent2_decisions.ndjson': 'Agent 2 decisions log',
            'agent3_risk.ndjson': 'Agent 3 risk evaluations',
            'agent4_learning_update.json': 'Agent 4 learning update',
        }
        
        missing = []
        present = []
        
        for filename, description in required_artifacts.items():
            filepath = date_dir / filename
            if filepath.exists():
                present.append(filename)
            else:
                missing.append({'file': filename, 'description': description})
        
        # Trades file is optional (may have no trades)
        trades_file = date_dir / 'trades.ndjson'
        if trades_file.exists():
            present.append('trades.ndjson')
        
        return {
            'date': date,
            'valid': len(missing) == 0,
            'missing': missing,
            'present': present,
            'missing_count': len(missing)
        }
    
    def get_artifact_stats(self, date: str) -> Dict[str, Any]:
        """
        Get statistics from artifacts for a date.
        
        Args:
            date: Date string in YYYY-MM-DD format
        
        Returns:
            Statistics dictionary
        """
        date_dir = self.get_date_dir(date)
        stats = {
            'date': date,
            'agent1_runs': 0,
            'agent2_decisions': 0,
            'agent3_evaluations': 0,
            'agent4_runs': 0,
            'trades': 0,
        }
        
        # Count Agent 1 runs
        agent1_manifest = date_dir / 'agent1_features_manifest.json'
        if agent1_manifest.exists():
            stats['agent1_runs'] = 1
        
        # Count Agent 2 decisions
        agent2_file = date_dir / 'agent2_decisions.ndjson'
        if agent2_file.exists():
            with open(agent2_file) as f:
                stats['agent2_decisions'] = sum(1 for _ in f)
        
        # Count Agent 3 evaluations
        agent3_file = date_dir / 'agent3_risk.ndjson'
        if agent3_file.exists():
            with open(agent3_file) as f:
                stats['agent3_evaluations'] = sum(1 for _ in f)
        
        # Count Agent 4 runs
        agent4_update = date_dir / 'agent4_learning_update.json'
        if agent4_update.exists():
            stats['agent4_runs'] = 1
        
        # Count trades
        trades_file = date_dir / 'trades.ndjson'
        if trades_file.exists():
            with open(trades_file) as f:
                stats['trades'] = sum(1 for _ in f)
        
        return stats
    
    def generate_summary(self, dates: List[str]) -> Dict[str, Any]:
        """
        Generate summary across all dates.
        
        Args:
            dates: List of date strings in YYYY-MM-DD format
        
        Returns:
            Summary dictionary with aggregated statistics
        """
        total_stats = {
            'days_processed': len(dates),
            'agent1_runs': 0,
            'agent2_decisions': 0,
            'agent3_evaluations': 0,
            'agent4_runs': 0,
            'total_trades': 0,
            'missing_artifacts': [],
            'validation_results': [],
        }
        
        for date in dates:
            validation = self.validate_artifacts(date)
            stats = self.get_artifact_stats(date)
            
            total_stats['agent1_runs'] += stats['agent1_runs']
            total_stats['agent2_decisions'] += stats['agent2_decisions']
            total_stats['agent3_evaluations'] += stats['agent3_evaluations']
            total_stats['agent4_runs'] += stats['agent4_runs']
            total_stats['total_trades'] += stats['trades']
            
            if not validation['valid']:
                total_stats['missing_artifacts'].append({
                    'date': date,
                    'missing': validation['missing']
                })
            
            total_stats['validation_results'].append(validation)
        
        # Calculate expected vs actual
        expected_agent1 = len(dates)  # Once per day
        expected_agent4 = len(dates)  # Once per day
        
        total_stats['agent1_expected'] = expected_agent1
        total_stats['agent1_missing'] = expected_agent1 - total_stats['agent1_runs']
        total_stats['agent4_expected'] = expected_agent4
        total_stats['agent4_missing'] = expected_agent4 - total_stats['agent4_runs']
        
        return total_stats
