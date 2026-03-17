"""
P&L Updater for AWS Agent Integration

This module tracks and uploads P&L data to AWS for risk management
and learning agents.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from loguru import logger


class PnLUpdater:
    """Track and upload P&L data to AWS.
    
    This class maintains:
    - Real-time P&L tracking
    - Trade history for the day
    - Account metrics for risk evaluation
    """
    
    def __init__(
        self,
        s3_bucket: str = None,
        region_name: str = 'us-east-1',
        initial_balance: float = 100000.0,
    ):
        """Initialize P&L Updater.
        
        Args:
            s3_bucket: S3 bucket for P&L data
            region_name: AWS region
            initial_balance: Starting account balance
        """
        self.s3_bucket = s3_bucket
        self.region_name = region_name
        self.initial_balance = initial_balance
        
        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_trades: List[Dict] = []
        self._winning_streak = 0
        self._losing_streak = 0
        self._peak_balance = initial_balance
        
        # Session tracking
        self._start_time = datetime.now(timezone.utc)
        self._current_date = self._start_time.strftime('%Y-%m-%d')
        
        # S3 client
        if BOTO3_AVAILABLE and s3_bucket:
            self.s3_client = boto3.client('s3', region_name=region_name)
        else:
            self.s3_client = None
        
        logger.info(f"Initialized PnLUpdater (balance: ${initial_balance:,.2f})")
    
    def record_trade(
        self,
        trade_id: str,
        action: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Record a completed trade.
        
        Args:
            trade_id: Unique trade identifier
            action: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            quantity: Number of contracts
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            pnl: Realized P&L
            metadata: Additional trade metadata
            
        Returns:
            Trade record with computed fields
        """
        # Check for day rollover
        self._check_day_rollover()
        
        # Determine outcome
        outcome = 'WIN' if pnl > 0 else 'LOSS'
        
        # Update streaks
        if outcome == 'WIN':
            self._winning_streak += 1
            self._losing_streak = 0
        else:
            self._losing_streak += 1
            self._winning_streak = 0
        
        # Calculate duration
        duration_seconds = int((exit_time - entry_time).total_seconds())
        
        # Build trade record
        trade_record = {
            'trade_id': trade_id,
            'date': self._current_date,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'duration_seconds': duration_seconds,
            'pnl': pnl,
            'outcome': outcome,
            'cumulative_pnl': self._daily_pnl + pnl,
            **(metadata or {}),
        }
        
        # Update daily metrics
        self._daily_pnl += pnl
        self._daily_trades.append(trade_record)
        
        # Update peak balance
        current_balance = self.initial_balance + self._daily_pnl
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
        
        logger.info(
            f"Recorded trade {trade_id}: {outcome} ${pnl:+.2f} "
            f"(daily P&L: ${self._daily_pnl:+.2f})"
        )
        
        return trade_record
    
    def get_account_metrics(self, current_position: int = 0) -> Dict[str, Any]:
        """Get current account metrics for risk evaluation.
        
        Args:
            current_position: Current open position size
            
        Returns:
            Account metrics dictionary
        """
        current_balance = self.initial_balance + self._daily_pnl
        drawdown = self._peak_balance - current_balance
        drawdown_pct = (drawdown / self._peak_balance * 100) if self._peak_balance > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for t in self._daily_trades if t.get('outcome') == 'WIN')
        total_trades = len(self._daily_trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        return {
            'current_pnl_today': self._daily_pnl,
            'current_position': current_position,
            'losing_streak': self._losing_streak,
            'winning_streak': self._winning_streak,
            'trades_today': total_trades,
            'account_balance': current_balance,
            'initial_balance': self.initial_balance,
            'peak_balance': self._peak_balance,
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct,
            'win_rate_today': win_rate,
            'open_risk': self._calculate_open_risk(current_position),
            'date': self._current_date,
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }
    
    def _calculate_open_risk(self, position_size: int) -> float:
        """Calculate open risk based on position size."""
        # Assume $50 risk per contract (10 points stop on ES micro)
        risk_per_contract = 50.0
        return abs(position_size) * risk_per_contract
    
    def upload_to_s3(self) -> Dict[str, Any]:
        """Upload current P&L data to S3.
        
        Returns:
            Upload result
        """
        if not self.s3_client:
            return {'status': 'skipped', 'reason': 'S3 not configured'}
        
        try:
            # Upload account metrics
            metrics = self.get_account_metrics()
            metrics_key = f"pnl/{self._current_date}/metrics.json"
            
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=metrics_key,
                Body=json.dumps(metrics, indent=2).encode('utf-8'),
                ContentType='application/json',
            )
            
            # Upload trade history
            if self._daily_trades:
                trades_key = f"pnl/{self._current_date}/trades.json"
                
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=trades_key,
                    Body=json.dumps(self._daily_trades, indent=2).encode('utf-8'),
                    ContentType='application/json',
                )
            
            logger.info(f"Uploaded P&L data to S3 ({len(self._daily_trades)} trades)")
            
            return {
                'status': 'success',
                'metrics_key': metrics_key,
                'trades_key': f"pnl/{self._current_date}/trades.json" if self._daily_trades else None,
                'trade_count': len(self._daily_trades),
            }
            
        except ClientError as e:
            logger.error(f"Failed to upload P&L data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def load_from_s3(self, date: str = None) -> Dict[str, Any]:
        """Load P&L data from S3.
        
        Args:
            date: Date to load (YYYY-MM-DD), defaults to current date
            
        Returns:
            Loaded data or error
        """
        if not self.s3_client:
            return {'status': 'skipped', 'reason': 'S3 not configured'}
        
        date = date or self._current_date
        
        try:
            # Load metrics
            metrics_key = f"pnl/{date}/metrics.json"
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=metrics_key,
            )
            metrics = json.loads(response['Body'].read().decode('utf-8'))
            
            # Load trades
            trades_key = f"pnl/{date}/trades.json"
            try:
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key=trades_key,
                )
                trades = json.loads(response['Body'].read().decode('utf-8'))
            except ClientError:
                trades = []
            
            # Update local state if loading current date
            if date == self._current_date:
                self._daily_pnl = metrics.get('current_pnl_today', 0)
                self._daily_trades = trades
                self._losing_streak = metrics.get('losing_streak', 0)
                self._winning_streak = metrics.get('winning_streak', 0)
                self._peak_balance = metrics.get('peak_balance', self.initial_balance)
            
            return {
                'status': 'success',
                'date': date,
                'metrics': metrics,
                'trades': trades,
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return {'status': 'not_found', 'date': date}
            logger.error(f"Failed to load P&L data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _check_day_rollover(self) -> None:
        """Check if day has rolled over and reset if needed."""
        current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        if current_date != self._current_date:
            logger.info(f"Day rollover: {self._current_date} -> {current_date}")
            
            # Upload previous day's data
            self.upload_to_s3()
            
            # Reset for new day
            self._current_date = current_date
            self._daily_pnl = 0.0
            self._daily_trades = []
            self._winning_streak = 0
            self._losing_streak = 0
    
    def reset_daily(self) -> None:
        """Manually reset daily tracking."""
        self._daily_pnl = 0.0
        self._daily_trades = []
        self._winning_streak = 0
        self._losing_streak = 0
        logger.info("Reset daily P&L tracking")
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """Get summary of daily trading activity."""
        wins = [t for t in self._daily_trades if t.get('outcome') == 'WIN']
        losses = [t for t in self._daily_trades if t.get('outcome') == 'LOSS']
        
        return {
            'date': self._current_date,
            'total_trades': len(self._daily_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self._daily_trades) if self._daily_trades else 0,
            'total_pnl': self._daily_pnl,
            'avg_win': sum(t.get('pnl', 0) for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.get('pnl', 0) for t in losses) / len(losses) if losses else 0,
            'largest_win': max((t.get('pnl', 0) for t in wins), default=0),
            'largest_loss': min((t.get('pnl', 0) for t in losses), default=0),
            'current_streak': f"+{self._winning_streak}" if self._winning_streak > 0 else f"-{self._losing_streak}",
        }
