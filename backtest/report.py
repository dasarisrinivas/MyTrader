"""
Backtest Report Generator
=========================

Generates comprehensive HTML/Markdown reports with:
- Executive summary
- Performance charts
- Trade analysis
- Regime breakdown
- Learnings and recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    title: str = "Backtest Report"
    symbol: str = "MES"
    strategy_name: str = "MES 1-Minute Trend Strategy"
    
    # Output format
    format: str = "html"  # "html" or "md"
    
    # Chart settings
    include_charts: bool = True
    chart_width: int = 800
    chart_height: int = 400
    
    # Section toggles
    include_executive_summary: bool = True
    include_performance_metrics: bool = True
    include_trade_analysis: bool = True
    include_regime_analysis: bool = True
    include_session_analysis: bool = True
    include_optimizer_analysis: bool = True
    include_block_reasons: bool = True
    include_learnings: bool = True
    include_trade_list: bool = True


class ReportGenerator:
    """
    Generates HTML/Markdown reports from backtest analysis.
    """
    
    def __init__(
        self,
        analysis: Dict,
        equity_curve: pd.Series,
        trades: List[Dict],
        config: Optional[ReportConfig] = None
    ):
        self.analysis = analysis
        self.equity_curve = equity_curve
        self.trades = trades
        self.config = config or ReportConfig()
    
    def generate(self, output_path: Path) -> None:
        """Generate report and save to file."""
        if self.config.format == "html":
            content = self._generate_html()
        else:
            content = self._generate_markdown()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(content)
        
        logger.info(f"Report saved to {output_path}")
    
    def _generate_html(self) -> str:
        """Generate HTML report."""
        metrics = self.analysis.get("metrics", {})
        regime = self.analysis.get("regime_analysis", {})
        session = self.analysis.get("session_analysis", {})
        optimizer = self.analysis.get("optimizer_analysis", {})
        learnings = self.analysis.get("learnings", {})
        block_reasons = self.analysis.get("block_reasons", {})
        
        # Build HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #4a90d9;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        h2 {{
            color: #16213e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #4a90d9;
            padding-left: 15px;
        }}
        h3 {{
            color: #1a1a2e;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .summary-box h2 {{
            color: white;
            border-left-color: rgba(255,255,255,0.5);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4a90d9;
        }}
        .metric-value.positive {{ color: #27ae60; }}
        .metric-value.negative {{ color: #e74c3c; }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #4a90d9;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }}
        .learning-box {{
            background: #fff3cd;
            border-left: 4px solid #f0ad4e;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .recommendation-box {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .block-reason {{
            display: inline-block;
            background: #f8f9fa;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 15px;
            font-size: 0.85em;
            border: 1px solid #dee2e6;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.85em;
        }}
        ul {{
            margin-left: 20px;
            margin-bottom: 15px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <h1>{self.config.title}</h1>
    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <p><strong>Strategy:</strong> {self.config.strategy_name} | <strong>Symbol:</strong> {self.config.symbol}</p>
"""
        
        # Executive Summary
        if self.config.include_executive_summary:
            html += self._html_executive_summary(metrics, learnings)
        
        # Performance Metrics
        if self.config.include_performance_metrics:
            html += self._html_performance_metrics(metrics)
        
        # Equity Curve Chart
        if self.config.include_charts and not self.equity_curve.empty:
            html += self._html_equity_chart()
        
        # Trade Analysis
        if self.config.include_trade_analysis:
            html += self._html_trade_analysis(metrics)
        
        # Regime Analysis
        if self.config.include_regime_analysis:
            html += self._html_regime_analysis(regime)
        
        # Session Analysis
        if self.config.include_session_analysis:
            html += self._html_session_analysis(session)
        
        # Optimizer Analysis
        if self.config.include_optimizer_analysis:
            html += self._html_optimizer_analysis(optimizer)
        
        # Block Reasons
        if self.config.include_block_reasons and block_reasons:
            html += self._html_block_reasons(block_reasons)
        
        # Learnings
        if self.config.include_learnings:
            html += self._html_learnings(learnings)
        
        # Trade List
        if self.config.include_trade_list and self.trades:
            html += self._html_trade_list()
        
        html += """
</body>
</html>
"""
        return html
    
    def _html_executive_summary(self, metrics: Dict, learnings: Dict) -> str:
        """Generate executive summary section."""
        total_return = metrics.get("total_return", 0) * 100
        total_pnl = metrics.get("total_pnl", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0) * 100
        win_rate = metrics.get("win_rate", 0) * 100
        total_trades = metrics.get("total_trades", 0)
        
        return_class = "positive" if total_return > 0 else "negative"
        assessment = learnings.get("overall_assessment", "")
        
        return f"""
    <div class="summary-box">
        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid" style="margin-top: 15px;">
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value {return_class}">{total_return:+.1f}%</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Total Return</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value {return_class}">${total_pnl:,.0f}</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Total P&L</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value">{sharpe:.2f}</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Sharpe Ratio</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value negative">{max_dd:.1f}%</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Max Drawdown</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Win Rate</div>
            </div>
            <div class="metric-card" style="background: rgba(255,255,255,0.1);">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label" style="color: rgba(255,255,255,0.8);">Total Trades</div>
            </div>
        </div>
        <p style="margin-top: 15px; font-size: 1.1em;">{assessment}</p>
    </div>
"""
    
    def _html_performance_metrics(self, metrics: Dict) -> str:
        """Generate detailed performance metrics section."""
        return f"""
    <h2>📈 Performance Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{metrics.get('cagr', 0)*100:.1f}%</div>
            <div class="metric-label">CAGR</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('sortino_ratio', 0):.2f}</div>
            <div class="metric-label">Sortino Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('calmar_ratio', 0):.2f}</div>
            <div class="metric-label">Calmar Ratio</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
            <div class="metric-label">Profit Factor</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('annualized_volatility', 0)*100:.1f}%</div>
            <div class="metric-label">Ann. Volatility</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.get('expectancy', 0):.2f}</div>
            <div class="metric-label">Expectancy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('avg_trade_duration_minutes', 0):.0f}m</div>
            <div class="metric-label">Avg Duration</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${metrics.get('total_commission', 0):.0f}</div>
            <div class="metric-label">Total Commission</div>
        </div>
    </div>
"""
    
    def _html_equity_chart(self) -> str:
        """Generate equity curve chart section (SVG-based)."""
        # Simple SVG chart generation
        if len(self.equity_curve) < 2:
            return ""
        
        # Downsample for performance
        step = max(1, len(self.equity_curve) // 500)
        values = self.equity_curve.iloc[::step].values
        
        min_val = values.min()
        max_val = values.max()
        range_val = max_val - min_val if max_val != min_val else 1
        
        width = self.config.chart_width
        height = self.config.chart_height
        padding = 50
        
        # Normalize values
        x_scale = (width - 2 * padding) / (len(values) - 1) if len(values) > 1 else 1
        y_scale = (height - 2 * padding) / range_val
        
        points = []
        for i, v in enumerate(values):
            x = padding + i * x_scale
            y = height - padding - (v - min_val) * y_scale
            points.append(f"{x:.1f},{y:.1f}")
        
        path = "M " + " L ".join(points)
        
        # Calculate drawdown areas
        cummax = np.maximum.accumulate(values)
        dd = (values - cummax)
        
        dd_path_parts = []
        for i, (v, d) in enumerate(zip(values, dd)):
            x = padding + i * x_scale
            y_equity = height - padding - (v - min_val) * y_scale
            y_cummax = height - padding - (cummax[i] - min_val) * y_scale
            
            if i == 0:
                dd_path_parts.append(f"M {x:.1f},{y_cummax:.1f}")
            else:
                dd_path_parts.append(f"L {x:.1f},{y_cummax:.1f}")
        
        # Reverse for bottom path
        for i in range(len(values) - 1, -1, -1):
            x = padding + i * x_scale
            y_equity = height - padding - (values[i] - min_val) * y_scale
            dd_path_parts.append(f"L {x:.1f},{y_equity:.1f}")
        
        dd_path_parts.append("Z")
        dd_path = " ".join(dd_path_parts)
        
        return f"""
    <h2>💹 Equity Curve</h2>
    <div class="chart-container">
        <svg width="{width}" height="{height}" style="display: block; margin: 0 auto;">
            <!-- Grid lines -->
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#eee" stroke-width="1"/>
                </pattern>
            </defs>
            <rect x="{padding}" y="{padding}" width="{width - 2*padding}" height="{height - 2*padding}" fill="url(#grid)"/>
            
            <!-- Drawdown area -->
            <path d="{dd_path}" fill="rgba(231, 76, 60, 0.1)"/>
            
            <!-- Equity line -->
            <path d="{path}" fill="none" stroke="#4a90d9" stroke-width="2"/>
            
            <!-- Axes -->
            <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333" stroke-width="1"/>
            <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333" stroke-width="1"/>
            
            <!-- Labels -->
            <text x="{width/2}" y="{height - 10}" text-anchor="middle" fill="#666" font-size="12">Time</text>
            <text x="{padding - 10}" y="{height - padding + 15}" text-anchor="end" fill="#666" font-size="11">${min_val:,.0f}</text>
            <text x="{padding - 10}" y="{padding + 5}" text-anchor="end" fill="#666" font-size="11">${max_val:,.0f}</text>
        </svg>
    </div>
"""
    
    def _html_trade_analysis(self, metrics: Dict) -> str:
        """Generate trade analysis section."""
        winning = metrics.get("winning_trades", 0)
        losing = metrics.get("losing_trades", 0)
        
        return f"""
    <h2>📊 Trade Analysis</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Total Trades</td>
            <td>{metrics.get('total_trades', 0)}</td>
        </tr>
        <tr>
            <td>Winning Trades</td>
            <td class="positive">{winning}</td>
        </tr>
        <tr>
            <td>Losing Trades</td>
            <td class="negative">{losing}</td>
        </tr>
        <tr>
            <td>Win Rate</td>
            <td>{metrics.get('win_rate', 0)*100:.1f}%</td>
        </tr>
        <tr>
            <td>Average Win</td>
            <td class="positive">${metrics.get('avg_win', 0):,.2f}</td>
        </tr>
        <tr>
            <td>Average Loss</td>
            <td class="negative">${metrics.get('avg_loss', 0):,.2f}</td>
        </tr>
        <tr>
            <td>Largest Win</td>
            <td class="positive">${metrics.get('largest_win', 0):,.2f}</td>
        </tr>
        <tr>
            <td>Largest Loss</td>
            <td class="negative">${metrics.get('largest_loss', 0):,.2f}</td>
        </tr>
    </table>
"""
    
    def _html_regime_analysis(self, regime: Dict) -> str:
        """Generate regime analysis section."""
        return f"""
    <h2>🌡️ Performance by Market Regime</h2>
    
    <h3>ATR Volatility Buckets</h3>
    <table>
        <tr>
            <th>Regime</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>Total P&L</th>
        </tr>
        <tr>
            <td>Low ATR</td>
            <td>{regime.get('low_atr_trades', 0)}</td>
            <td>{regime.get('low_atr_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if regime.get('low_atr_pnl', 0) > 0 else 'negative'}">${regime.get('low_atr_pnl', 0):,.0f}</td>
        </tr>
        <tr>
            <td>Medium ATR</td>
            <td>{regime.get('medium_atr_trades', 0)}</td>
            <td>{regime.get('medium_atr_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if regime.get('medium_atr_pnl', 0) > 0 else 'negative'}">${regime.get('medium_atr_pnl', 0):,.0f}</td>
        </tr>
        <tr>
            <td>High ATR</td>
            <td>{regime.get('high_atr_trades', 0)}</td>
            <td>{regime.get('high_atr_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if regime.get('high_atr_pnl', 0) > 0 else 'negative'}">${regime.get('high_atr_pnl', 0):,.0f}</td>
        </tr>
    </table>
    
    <h3>ADX Trend Strength</h3>
    <table>
        <tr>
            <th>Regime</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>Total P&L</th>
        </tr>
        <tr>
            <td>Trending (ADX ≥ 20)</td>
            <td>{regime.get('trending_trades', 0)}</td>
            <td>{regime.get('trending_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if regime.get('trending_pnl', 0) > 0 else 'negative'}">${regime.get('trending_pnl', 0):,.0f}</td>
        </tr>
        <tr>
            <td>Ranging (ADX < 20)</td>
            <td>{regime.get('ranging_trades', 0)}</td>
            <td>{regime.get('ranging_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if regime.get('ranging_pnl', 0) > 0 else 'negative'}">${regime.get('ranging_pnl', 0):,.0f}</td>
        </tr>
    </table>
"""
    
    def _html_session_analysis(self, session: Dict) -> str:
        """Generate session analysis section."""
        return f"""
    <h2>🕐 Performance by Session</h2>
    <table>
        <tr>
            <th>Session</th>
            <th>Trades</th>
            <th>Win Rate</th>
            <th>Total P&L</th>
            <th>Avg Trade</th>
        </tr>
        <tr>
            <td>RTH (9:30 AM - 4:00 PM ET)</td>
            <td>{session.get('rth_trades', 0)}</td>
            <td>{session.get('rth_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if session.get('rth_pnl', 0) > 0 else 'negative'}">${session.get('rth_pnl', 0):,.0f}</td>
            <td>${session.get('rth_avg_trade', 0):.2f}</td>
        </tr>
        <tr>
            <td>Overnight</td>
            <td>{session.get('overnight_trades', 0)}</td>
            <td>{session.get('overnight_win_rate', 0)*100:.1f}%</td>
            <td class="{'positive' if session.get('overnight_pnl', 0) > 0 else 'negative'}">${session.get('overnight_pnl', 0):,.0f}</td>
            <td>${session.get('overnight_avg_trade', 0):.2f}</td>
        </tr>
    </table>
"""
    
    def _html_optimizer_analysis(self, optimizer: Dict) -> str:
        """Generate optimizer analysis section."""
        return f"""
    <h2>⚙️ Trend Continuation Optimizer</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Near-TP Events</td><td>{optimizer.get('near_tp_events', 0)}</td></tr>
        <tr><td>Extensions Approved</td><td>{optimizer.get('extensions_approved', 0)}</td></tr>
        <tr><td>Approval Rate</td><td>{optimizer.get('approval_rate', 0)*100:.1f}%</td></tr>
        <tr><td>Extension Win Rate</td><td>{optimizer.get('extension_win_rate', 0)*100:.1f}%</td></tr>
        <tr><td>Avg Extended Trade P&L</td><td>${optimizer.get('avg_extended_trade_pnl', 0):.2f}</td></tr>
        <tr><td>Avg Non-Extended Trade P&L</td><td>${optimizer.get('avg_non_extended_trade_pnl', 0):.2f}</td></tr>
    </table>
"""
    
    def _html_block_reasons(self, block_reasons: Dict) -> str:
        """Generate block reasons section."""
        sorted_reasons = sorted(block_reasons.items(), key=lambda x: x[1], reverse=True)[:15]
        
        reasons_html = ""
        for reason, count in sorted_reasons:
            reasons_html += f'<span class="block-reason">{reason}: {count}</span>\n'
        
        return f"""
    <h2>🚫 Trade Block Reasons (Top 15)</h2>
    <p>Total blocks analyzed: {sum(block_reasons.values())}</p>
    <div style="margin: 15px 0;">
        {reasons_html}
    </div>
"""
    
    def _html_learnings(self, learnings: Dict) -> str:
        """Generate learnings section."""
        html = """
    <h2>📚 Key Learnings & Recommendations</h2>
"""
        
        # Regime notes
        regime_notes = learnings.get("regime_notes", [])
        if regime_notes:
            html += """
    <h3>Market Regime Insights</h3>
"""
            for note in regime_notes:
                html += f'<div class="learning-box">📊 {note}</div>\n'
        
        # Session notes
        session_notes = learnings.get("session_notes", [])
        if session_notes:
            html += """
    <h3>Session Insights</h3>
"""
            for note in session_notes:
                html += f'<div class="learning-box">🕐 {note}</div>\n'
        
        # Helpful filters
        helpful = learnings.get("helpful_filters", [])
        if helpful:
            html += """
    <h3>Helpful Filters</h3>
    <ul>
"""
            for f in helpful:
                html += f'<li class="positive">✅ {f}</li>\n'
            html += "</ul>\n"
        
        # Harmful filters
        harmful = learnings.get("harmful_filters", [])
        if harmful:
            html += """
    <h3>Filters to Review</h3>
    <ul>
"""
            for f in harmful:
                html += f'<li class="warning">⚠️ {f}</li>\n'
            html += "</ul>\n"
        
        # Drawdown causes
        dd_causes = learnings.get("drawdown_causes", [])
        if dd_causes:
            html += """
    <h3>Drawdown Analysis</h3>
    <ul>
"""
            for cause in dd_causes:
                html += f'<li class="negative">📉 {cause}</li>\n'
            html += "</ul>\n"
        
        # Recommendations
        recs = learnings.get("recommendations", [])
        if recs:
            html += """
    <h3>Recommendations</h3>
"""
            for rec in recs:
                html += f'<div class="recommendation-box">💡 {rec}</div>\n'
        
        return html
    
    def _html_trade_list(self) -> str:
        """Generate trade list table."""
        if not self.trades:
            return ""
        
        # Show last 50 trades
        recent_trades = self.trades[-50:] if len(self.trades) > 50 else self.trades
        
        rows = ""
        for trade in recent_trades:
            pnl = trade.get("realized_pnl", 0)
            pnl_class = "positive" if pnl > 0 else "negative"
            
            # Handle timestamps - convert to string if needed
            entry_time = trade.get('entry_time', '')
            if hasattr(entry_time, 'strftime'):
                entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                entry_time = str(entry_time)[:19]
            
            rows += f"""
        <tr>
            <td>{entry_time}</td>
            <td>{trade.get('direction', '')}</td>
            <td>{trade.get('quantity', 0)}</td>
            <td>${trade.get('entry_price', 0):.2f}</td>
            <td>${trade.get('exit_price', 0):.2f}</td>
            <td class="{pnl_class}">${pnl:.2f}</td>
            <td>{trade.get('exit_reason', '')}</td>
        </tr>
"""
        
        return f"""
    <h2>📋 Recent Trades (Last 50)</h2>
    <table>
        <tr>
            <th>Entry Time</th>
            <th>Direction</th>
            <th>Qty</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>P&L</th>
            <th>Reason</th>
        </tr>
        {rows}
    </table>
"""
    
    def _generate_markdown(self) -> str:
        """Generate Markdown report."""
        metrics = self.analysis.get("metrics", {})
        learnings = self.analysis.get("learnings", {})
        
        md = f"""# {self.config.title}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Strategy:** {self.config.strategy_name}  
**Symbol:** {self.config.symbol}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Return | {metrics.get('total_return', 0)*100:+.1f}% |
| Total P&L | ${metrics.get('total_pnl', 0):,.0f} |
| Sharpe Ratio | {metrics.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {metrics.get('max_drawdown', 0)*100:.1f}% |
| Win Rate | {metrics.get('win_rate', 0)*100:.1f}% |
| Total Trades | {metrics.get('total_trades', 0)} |

**Assessment:** {learnings.get('overall_assessment', 'N/A')}

## Performance Metrics

| Metric | Value |
|--------|-------|
| CAGR | {metrics.get('cagr', 0)*100:.1f}% |
| Sortino Ratio | {metrics.get('sortino_ratio', 0):.2f} |
| Calmar Ratio | {metrics.get('calmar_ratio', 0):.2f} |
| Profit Factor | {metrics.get('profit_factor', 0):.2f} |
| Expectancy | ${metrics.get('expectancy', 0):.2f} |
| Avg Trade Duration | {metrics.get('avg_trade_duration_minutes', 0):.0f} min |

## Key Learnings

### Regime Insights
"""
        for note in learnings.get("regime_notes", []):
            md += f"- {note}\n"
        
        md += "\n### Session Insights\n"
        for note in learnings.get("session_notes", []):
            md += f"- {note}\n"
        
        md += "\n### Recommendations\n"
        for rec in learnings.get("recommendations", []):
            md += f"- 💡 {rec}\n"
        
        md += f"""
## Block Reasons (Top 10)

| Reason | Count |
|--------|-------|
"""
        block_reasons = self.analysis.get("block_reasons", {})
        sorted_reasons = sorted(block_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
        for reason, count in sorted_reasons:
            md += f"| {reason} | {count} |\n"
        
        return md


def generate_report(
    analysis: Dict,
    equity_curve: pd.Series,
    trades: List[Dict],
    output_path: Path,
    config: Optional[ReportConfig] = None
) -> None:
    """
    Convenience function to generate a report.
    
    Args:
        analysis: Analysis results from BacktestAnalyzer
        equity_curve: Equity curve series
        trades: List of trade dicts
        output_path: Path to save report
        config: Report configuration
    """
    generator = ReportGenerator(analysis, equity_curve, trades, config)
    generator.generate(output_path)
