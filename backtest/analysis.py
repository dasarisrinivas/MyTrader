"""
Backtest Analysis Module
========================

Comprehensive performance analysis and diagnostics:
- Standard metrics (return, Sharpe, drawdown, profit factor)
- Strategy diagnostics (by regime, session, block reasons)
- TrendContinuationOptimizer analysis
- Learnings extraction and recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Standard performance metrics."""
    
    # Returns
    total_return: float = 0.0
    total_pnl: float = 0.0
    cagr: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown: float = 0.0
    
    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Average trade
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    
    # Largest
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Duration
    avg_trade_duration_minutes: float = 0.0
    
    # Exposure
    time_in_market_pct: float = 0.0
    avg_position_size: float = 0.0
    
    # Costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    # Volatility
    annualized_volatility: float = 0.0
    daily_volatility: float = 0.0


@dataclass
class RegimeAnalysis:
    """Performance breakdown by market regime."""
    
    # ATR buckets
    low_atr_trades: int = 0
    low_atr_win_rate: float = 0.0
    low_atr_pnl: float = 0.0
    
    medium_atr_trades: int = 0
    medium_atr_win_rate: float = 0.0
    medium_atr_pnl: float = 0.0
    
    high_atr_trades: int = 0
    high_atr_win_rate: float = 0.0
    high_atr_pnl: float = 0.0
    
    # VIX buckets (if available)
    low_vix_trades: int = 0
    low_vix_pnl: float = 0.0
    
    high_vix_trades: int = 0
    high_vix_pnl: float = 0.0
    
    # ADX buckets
    trending_trades: int = 0
    trending_win_rate: float = 0.0
    trending_pnl: float = 0.0
    
    ranging_trades: int = 0
    ranging_win_rate: float = 0.0
    ranging_pnl: float = 0.0


@dataclass
class SessionAnalysis:
    """Performance breakdown by trading session."""
    
    # RTH (Regular Trading Hours)
    rth_trades: int = 0
    rth_win_rate: float = 0.0
    rth_pnl: float = 0.0
    rth_avg_trade: float = 0.0
    
    # Overnight (outside RTH)
    overnight_trades: int = 0
    overnight_win_rate: float = 0.0
    overnight_pnl: float = 0.0
    overnight_avg_trade: float = 0.0
    
    # By hour
    hourly_pnl: Dict[int, float] = field(default_factory=dict)
    hourly_win_rate: Dict[int, float] = field(default_factory=dict)
    
    # By day of week
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    daily_win_rate: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizerAnalysis:
    """Analysis of TrendContinuationOptimizer performance."""
    
    # Events
    near_tp_events: int = 0
    extensions_attempted: int = 0
    extensions_approved: int = 0
    approval_rate: float = 0.0
    
    # Extended trade outcomes
    extended_trades_win: int = 0
    extended_trades_loss: int = 0
    extension_win_rate: float = 0.0
    
    # P&L impact
    incremental_r_gained: float = 0.0  # R-multiple gained from extensions
    extension_pnl: float = 0.0
    
    # Stopouts after extension
    stopouts_after_extension: int = 0
    
    # Comparison vs take-profit
    avg_extended_trade_pnl: float = 0.0
    avg_non_extended_trade_pnl: float = 0.0


@dataclass
class Learnings:
    """Key learnings and recommendations from backtest."""
    
    # Performance summary
    overall_assessment: str = ""
    
    # Regime insights
    best_regime: str = ""
    worst_regime: str = ""
    regime_notes: List[str] = field(default_factory=list)
    
    # Session insights
    best_session: str = ""
    session_notes: List[str] = field(default_factory=list)
    
    # Filter analysis
    helpful_filters: List[str] = field(default_factory=list)
    harmful_filters: List[str] = field(default_factory=list)
    
    # Drawdown analysis
    drawdown_causes: List[str] = field(default_factory=list)
    
    # Parameter sensitivity
    sensitive_parameters: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class BacktestAnalyzer:
    """
    Comprehensive backtest analysis.
    
    Takes backtest results and produces:
    - Standard performance metrics
    - Regime-based analysis
    - Session-based analysis
    - TrendContinuationOptimizer analysis
    - Actionable learnings
    """
    
    def __init__(
        self,
        trades: List[Dict],
        equity_curve: pd.Series,
        block_reasons: Dict[str, int],
        optimizer_stats: Dict[str, Any],
        initial_capital: float = 50000.0,
        risk_free_rate: float = 0.02,
    ):
        self.trades = trades
        self.equity_curve = equity_curve
        self.block_reasons = block_reasons
        self.optimizer_stats = optimizer_stats
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Convert trades to DataFrame for analysis
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def analyze(self) -> Dict:
        """Run full analysis and return results."""
        metrics = self.compute_metrics()
        regime = self.analyze_by_regime()
        session = self.analyze_by_session()
        optimizer = self.analyze_optimizer()
        learnings = self.extract_learnings(metrics, regime, session, optimizer)
        
        return {
            "metrics": metrics.__dict__,
            "regime_analysis": regime.__dict__,
            "session_analysis": {
                **{k: v for k, v in session.__dict__.items() if not isinstance(v, dict)},
                "hourly_pnl": session.hourly_pnl,
                "hourly_win_rate": session.hourly_win_rate,
                "daily_pnl": session.daily_pnl,
                "daily_win_rate": session.daily_win_rate,
            },
            "optimizer_analysis": optimizer.__dict__,
            "learnings": {
                **{k: v for k, v in learnings.__dict__.items() if not isinstance(v, list)},
                "regime_notes": learnings.regime_notes,
                "session_notes": learnings.session_notes,
                "helpful_filters": learnings.helpful_filters,
                "harmful_filters": learnings.harmful_filters,
                "drawdown_causes": learnings.drawdown_causes,
                "sensitive_parameters": learnings.sensitive_parameters,
                "recommendations": learnings.recommendations,
            },
            "block_reasons": self.block_reasons,
        }
    
    def compute_metrics(self) -> PerformanceMetrics:
        """Compute standard performance metrics."""
        metrics = PerformanceMetrics()
        
        if self.equity_curve.empty:
            return metrics
        
        # Returns
        metrics.total_pnl = float(self.equity_curve.iloc[-1] - self.initial_capital)
        metrics.total_return = metrics.total_pnl / self.initial_capital
        
        # CAGR
        days = len(self.equity_curve)
        years = days / (252 * 390)  # Approximate trading minutes per year
        if years > 0:
            metrics.cagr = (1 + metrics.total_return) ** (1 / years) - 1
        
        # Returns series
        returns = self.equity_curve.pct_change().dropna()
        
        if len(returns) > 1:
            # Annualized volatility (assuming 1-minute bars)
            metrics.daily_volatility = float(returns.std() * np.sqrt(390))
            metrics.annualized_volatility = float(returns.std() * np.sqrt(252 * 390))
            
            # Sharpe ratio (annualized)
            if metrics.annualized_volatility > 0:
                excess_return = metrics.cagr - self.risk_free_rate
                metrics.sharpe_ratio = excess_return / metrics.annualized_volatility
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = float(downside_returns.std() * np.sqrt(252 * 390))
                if downside_std > 0:
                    metrics.sortino_ratio = (metrics.cagr - self.risk_free_rate) / downside_std
        
        # Drawdown analysis
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve / cummax - 1)
        metrics.max_drawdown = float(drawdown.min())
        metrics.avg_drawdown = float(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0.0
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            dd_groups = (~in_drawdown).cumsum()
            dd_lengths = in_drawdown.groupby(dd_groups).sum()
            if len(dd_lengths) > 0:
                # Convert minutes to days
                metrics.max_drawdown_duration_days = int(dd_lengths.max() / 390)
        
        # Calmar ratio
        if metrics.max_drawdown != 0:
            metrics.calmar_ratio = abs(metrics.cagr / metrics.max_drawdown)
        
        # Trade statistics
        if not self.trades_df.empty and "realized_pnl" in self.trades_df.columns:
            pnls = self.trades_df["realized_pnl"]
            
            metrics.total_trades = len(pnls)
            metrics.winning_trades = int((pnls > 0).sum())
            metrics.losing_trades = int((pnls <= 0).sum())
            metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
            
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            
            metrics.avg_win = float(wins.mean()) if len(wins) > 0 else 0
            metrics.avg_loss = float(losses.mean()) if len(losses) > 0 else 0
            metrics.avg_trade = float(pnls.mean())
            
            metrics.largest_win = float(wins.max()) if len(wins) > 0 else 0
            metrics.largest_loss = float(losses.min()) if len(losses) > 0 else 0
            
            # Profit factor
            total_wins = wins.sum() if len(wins) > 0 else 0
            total_losses = abs(losses.sum()) if len(losses) > 0 else 0
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            
            # Expectancy
            metrics.expectancy = (
                metrics.win_rate * metrics.avg_win + 
                (1 - metrics.win_rate) * metrics.avg_loss
            )
            
            # Trade duration
            if "entry_time" in self.trades_df.columns and "exit_time" in self.trades_df.columns:
                try:
                    entry_times = pd.to_datetime(self.trades_df["entry_time"])
                    exit_times = pd.to_datetime(self.trades_df["exit_time"])
                    durations = (exit_times - entry_times).dt.total_seconds() / 60
                    metrics.avg_trade_duration_minutes = float(durations.mean())
                except:
                    pass
        
        return metrics
    
    def analyze_by_regime(self) -> RegimeAnalysis:
        """Analyze performance by market regime (ATR, ADX buckets)."""
        analysis = RegimeAnalysis()
        
        if self.trades_df.empty:
            return analysis
        
        # Check if we have metadata with indicators
        if "entry_metadata" not in self.trades_df.columns and "metadata" not in self.trades_df.columns:
            return analysis
        
        # Extract ATR from metadata
        def get_atr(row):
            meta = row.get("entry_metadata") or row.get("metadata") or {}
            if isinstance(meta, dict):
                return meta.get("atr", meta.get("atr_value", np.nan))
            return np.nan
        
        def get_adx(row):
            meta = row.get("entry_metadata") or row.get("metadata") or {}
            if isinstance(meta, dict):
                return meta.get("adx_value", np.nan)
            return np.nan
        
        self.trades_df["atr"] = self.trades_df.apply(get_atr, axis=1)
        self.trades_df["adx"] = self.trades_df.apply(get_adx, axis=1)
        
        # ATR buckets (percentile-based)
        if self.trades_df["atr"].notna().any():
            atr_33 = self.trades_df["atr"].quantile(0.33)
            atr_67 = self.trades_df["atr"].quantile(0.67)
            
            low_atr = self.trades_df[self.trades_df["atr"] <= atr_33]
            med_atr = self.trades_df[(self.trades_df["atr"] > atr_33) & (self.trades_df["atr"] <= atr_67)]
            high_atr = self.trades_df[self.trades_df["atr"] > atr_67]
            
            analysis.low_atr_trades = len(low_atr)
            analysis.low_atr_pnl = float(low_atr["realized_pnl"].sum()) if len(low_atr) > 0 else 0
            analysis.low_atr_win_rate = float((low_atr["realized_pnl"] > 0).mean()) if len(low_atr) > 0 else 0
            
            analysis.medium_atr_trades = len(med_atr)
            analysis.medium_atr_pnl = float(med_atr["realized_pnl"].sum()) if len(med_atr) > 0 else 0
            analysis.medium_atr_win_rate = float((med_atr["realized_pnl"] > 0).mean()) if len(med_atr) > 0 else 0
            
            analysis.high_atr_trades = len(high_atr)
            analysis.high_atr_pnl = float(high_atr["realized_pnl"].sum()) if len(high_atr) > 0 else 0
            analysis.high_atr_win_rate = float((high_atr["realized_pnl"] > 0).mean()) if len(high_atr) > 0 else 0
        
        # ADX buckets (trending vs ranging)
        if self.trades_df["adx"].notna().any():
            trending = self.trades_df[self.trades_df["adx"] >= 20]
            ranging = self.trades_df[self.trades_df["adx"] < 20]
            
            analysis.trending_trades = len(trending)
            analysis.trending_pnl = float(trending["realized_pnl"].sum()) if len(trending) > 0 else 0
            analysis.trending_win_rate = float((trending["realized_pnl"] > 0).mean()) if len(trending) > 0 else 0
            
            analysis.ranging_trades = len(ranging)
            analysis.ranging_pnl = float(ranging["realized_pnl"].sum()) if len(ranging) > 0 else 0
            analysis.ranging_win_rate = float((ranging["realized_pnl"] > 0).mean()) if len(ranging) > 0 else 0
        
        return analysis
    
    def analyze_by_session(self) -> SessionAnalysis:
        """Analyze performance by trading session."""
        analysis = SessionAnalysis()
        
        if self.trades_df.empty or "entry_time" not in self.trades_df.columns:
            return analysis
        
        try:
            # Parse timestamps
            self.trades_df["entry_dt"] = pd.to_datetime(self.trades_df["entry_time"])
            
            # Convert to Eastern Time for session analysis
            self.trades_df["entry_hour"] = self.trades_df["entry_dt"].dt.hour
            self.trades_df["entry_dow"] = self.trades_df["entry_dt"].dt.day_name()
            
            # RTH classification (9:30 AM - 4:00 PM ET, approximately 14:30-21:00 UTC)
            self.trades_df["is_rth"] = self.trades_df["entry_hour"].between(14, 20)
            
            # RTH analysis
            rth = self.trades_df[self.trades_df["is_rth"]]
            overnight = self.trades_df[~self.trades_df["is_rth"]]
            
            analysis.rth_trades = len(rth)
            analysis.rth_pnl = float(rth["realized_pnl"].sum()) if len(rth) > 0 else 0
            analysis.rth_win_rate = float((rth["realized_pnl"] > 0).mean()) if len(rth) > 0 else 0
            analysis.rth_avg_trade = float(rth["realized_pnl"].mean()) if len(rth) > 0 else 0
            
            analysis.overnight_trades = len(overnight)
            analysis.overnight_pnl = float(overnight["realized_pnl"].sum()) if len(overnight) > 0 else 0
            analysis.overnight_win_rate = float((overnight["realized_pnl"] > 0).mean()) if len(overnight) > 0 else 0
            analysis.overnight_avg_trade = float(overnight["realized_pnl"].mean()) if len(overnight) > 0 else 0
            
            # By hour
            hourly = self.trades_df.groupby("entry_hour")["realized_pnl"]
            analysis.hourly_pnl = hourly.sum().to_dict()
            analysis.hourly_win_rate = self.trades_df.groupby("entry_hour").apply(
                lambda x: (x["realized_pnl"] > 0).mean()
            ).to_dict()
            
            # By day of week
            daily = self.trades_df.groupby("entry_dow")["realized_pnl"]
            analysis.daily_pnl = daily.sum().to_dict()
            analysis.daily_win_rate = self.trades_df.groupby("entry_dow").apply(
                lambda x: (x["realized_pnl"] > 0).mean()
            ).to_dict()
            
        except Exception as e:
            logger.warning(f"Session analysis failed: {e}")
        
        return analysis
    
    def analyze_optimizer(self) -> OptimizerAnalysis:
        """Analyze TrendContinuationOptimizer performance."""
        analysis = OptimizerAnalysis()
        
        if not self.optimizer_stats:
            return analysis
        
        analysis.near_tp_events = self.optimizer_stats.get("near_tp_events", 0)
        analysis.extensions_approved = self.optimizer_stats.get("extensions_approved", 0)
        
        if analysis.near_tp_events > 0:
            analysis.approval_rate = analysis.extensions_approved / analysis.near_tp_events
        
        # Analyze extended vs non-extended trades
        if not self.trades_df.empty and "entry_metadata" in self.trades_df.columns:
            def was_extended(row):
                meta = row.get("entry_metadata") or row.get("metadata") or {}
                if isinstance(meta, dict):
                    return meta.get("was_extended", False)
                return False
            
            self.trades_df["was_extended"] = self.trades_df.apply(was_extended, axis=1)
            
            extended = self.trades_df[self.trades_df["was_extended"]]
            non_extended = self.trades_df[~self.trades_df["was_extended"]]
            
            if len(extended) > 0:
                analysis.extended_trades_win = int((extended["realized_pnl"] > 0).sum())
                analysis.extended_trades_loss = int((extended["realized_pnl"] <= 0).sum())
                analysis.extension_win_rate = analysis.extended_trades_win / len(extended)
                analysis.extension_pnl = float(extended["realized_pnl"].sum())
                analysis.avg_extended_trade_pnl = float(extended["realized_pnl"].mean())
            
            if len(non_extended) > 0:
                analysis.avg_non_extended_trade_pnl = float(non_extended["realized_pnl"].mean())
        
        return analysis
    
    def extract_learnings(
        self,
        metrics: PerformanceMetrics,
        regime: RegimeAnalysis,
        session: SessionAnalysis,
        optimizer: OptimizerAnalysis
    ) -> Learnings:
        """Extract actionable learnings from analysis."""
        learnings = Learnings()
        
        # Overall assessment
        if metrics.total_return > 0 and metrics.sharpe_ratio > 1.0:
            learnings.overall_assessment = "POSITIVE: Strategy shows promise with profitable returns and acceptable risk-adjusted performance."
        elif metrics.total_return > 0:
            learnings.overall_assessment = "MIXED: Strategy is profitable but risk-adjusted returns need improvement."
        elif metrics.total_return > -0.1:
            learnings.overall_assessment = "MARGINAL: Strategy is near breakeven. Significant tuning needed."
        else:
            learnings.overall_assessment = "NEGATIVE: Strategy is unprofitable. Major changes required."
        
        # Regime insights
        if regime.trending_pnl > regime.ranging_pnl:
            learnings.best_regime = "TRENDING (ADX >= 20)"
            learnings.regime_notes.append(
                f"Performs better in trending markets: {regime.trending_win_rate:.1%} win rate, "
                f"${regime.trending_pnl:.0f} total P&L"
            )
        else:
            learnings.best_regime = "RANGING (ADX < 20)"
            learnings.regime_notes.append(
                f"Surprisingly performs better in ranging markets: {regime.ranging_win_rate:.1%} win rate"
            )
        
        # ATR insights
        atr_pnls = {
            "Low ATR": regime.low_atr_pnl,
            "Medium ATR": regime.medium_atr_pnl,
            "High ATR": regime.high_atr_pnl,
        }
        best_atr = max(atr_pnls, key=atr_pnls.get)
        worst_atr = min(atr_pnls, key=atr_pnls.get)
        
        if atr_pnls[best_atr] > 0 and atr_pnls[worst_atr] < 0:
            learnings.regime_notes.append(
                f"ATR matters: Best in {best_atr} (${atr_pnls[best_atr]:.0f}), "
                f"Worst in {worst_atr} (${atr_pnls[worst_atr]:.0f})"
            )
        
        # Session insights
        if session.rth_pnl > session.overnight_pnl:
            learnings.best_session = "RTH (9:30 AM - 4:00 PM ET)"
            learnings.session_notes.append(
                f"RTH outperforms: ${session.rth_pnl:.0f} vs ${session.overnight_pnl:.0f} overnight"
            )
        else:
            learnings.best_session = "Overnight"
            learnings.session_notes.append(
                "Overnight session is more profitable - consider focusing here"
            )
        
        # Hourly patterns
        if session.hourly_pnl:
            best_hour = max(session.hourly_pnl, key=session.hourly_pnl.get)
            worst_hour = min(session.hourly_pnl, key=session.hourly_pnl.get)
            
            if session.hourly_pnl[best_hour] > 0 and session.hourly_pnl[worst_hour] < 0:
                learnings.session_notes.append(
                    f"Best hour: {best_hour}:00 UTC (${session.hourly_pnl[best_hour]:.0f}), "
                    f"Worst: {worst_hour}:00 UTC (${session.hourly_pnl[worst_hour]:.0f})"
                )
        
        # Block reason analysis
        if self.block_reasons:
            total_blocks = sum(self.block_reasons.values())
            top_blocks = sorted(self.block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for reason, count in top_blocks:
                pct = count / total_blocks * 100
                if pct > 20:
                    if "ATR" in reason or "CHOP" in reason:
                        learnings.helpful_filters.append(
                            f"{reason} ({pct:.1f}% of blocks) - likely helping avoid bad setups"
                        )
                    elif "LIMIT" in reason or "LOSS" in reason:
                        learnings.helpful_filters.append(
                            f"{reason} ({pct:.1f}% of blocks) - risk management working"
                        )
                    else:
                        learnings.harmful_filters.append(
                            f"{reason} ({pct:.1f}% of blocks) - review if too restrictive"
                        )
        
        # Drawdown analysis
        if metrics.max_drawdown < -0.05:
            learnings.drawdown_causes.append(
                f"Max drawdown of {metrics.max_drawdown:.1%} is significant"
            )
            
            if metrics.largest_loss < -500:
                learnings.drawdown_causes.append(
                    f"Large single loss of ${metrics.largest_loss:.0f} - review stop placement"
                )
            
            if metrics.losing_trades > metrics.winning_trades * 1.5:
                learnings.drawdown_causes.append(
                    "Low win rate contributing to drawdowns - consider tighter entry filters"
                )
        
        # Recommendations
        if metrics.win_rate < 0.4:
            learnings.recommendations.append(
                "Win rate below 40% - tighten entry criteria or improve signal quality"
            )
        
        if metrics.profit_factor < 1.0:
            learnings.recommendations.append(
                "Profit factor < 1 - strategy is unprofitable, needs fundamental changes"
            )
        elif metrics.profit_factor < 1.5:
            learnings.recommendations.append(
                "Profit factor between 1-1.5 - marginally profitable, focus on reducing losses"
            )
        
        if metrics.avg_trade_duration_minutes > 60:
            learnings.recommendations.append(
                f"Average trade duration {metrics.avg_trade_duration_minutes:.0f} min - "
                "consider tighter time-based exits"
            )
        
        if optimizer.approval_rate > 0.5 and optimizer.extension_win_rate < 0.5:
            learnings.recommendations.append(
                "Trend extensions have low win rate - consider stricter extension criteria"
            )
        
        if regime.ranging_trades > regime.trending_trades and regime.ranging_win_rate < 0.4:
            learnings.recommendations.append(
                "Too many trades in ranging markets with poor results - "
                "increase ADX threshold for entry"
            )
        
        return learnings
    
    def save_results(self, output_dir: Path) -> None:
        """Save analysis results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full analysis as JSON
        analysis = self.analyze()
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save trades CSV
        if not self.trades_df.empty:
            self.trades_df.to_csv(output_dir / "trades.csv", index=False)
        
        # Save equity curve
        if not self.equity_curve.empty:
            self.equity_curve.to_csv(output_dir / "equity_curve.csv")
        
        logger.info(f"Analysis results saved to {output_dir}")


def analyze_backtest(
    results: Dict,
    initial_capital: float = 50000.0
) -> Dict:
    """
    Convenience function to analyze backtest results.
    
    Args:
        results: Results from BacktestEngine.run()
        initial_capital: Initial capital for metrics calculation
        
    Returns:
        Full analysis dict
    """
    analyzer = BacktestAnalyzer(
        trades=results.get("trades", []),
        equity_curve=results.get("equity_curve", pd.Series()),
        block_reasons=results.get("block_reasons", {}),
        optimizer_stats=results.get("optimizer_stats", {}),
        initial_capital=initial_capital,
    )
    
    return analyzer.analyze()
