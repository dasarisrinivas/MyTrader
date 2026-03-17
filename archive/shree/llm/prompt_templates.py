"""LLM prompt templates for autonomous trading system."""
from __future__ import annotations

from typing import Dict, List, Optional


class PromptTemplates:
    """Templates for LLM-based analysis and decision-making."""
    
    @staticmethod
    def daily_summary_prompt(
        metrics: dict,
        patterns: List[dict],
        recent_trades: List[dict]
    ) -> str:
        """Generate prompt for daily performance summary.
        
        Args:
            metrics: Daily performance metrics
            patterns: Identified behavioral patterns
            recent_trades: Recent trade details
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are an expert trading system analyst reviewing today's performance for an algorithmic trading bot.

DAILY PERFORMANCE DATA:
Date: {metrics['date']}
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.1%} ({metrics['winning_trades']} wins, {metrics['losing_trades']} losses)
Net P&L: ${metrics['net_pnl']:,.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: ${metrics['max_drawdown']:,.2f}
Average Holding Time: {metrics['avg_holding_time_minutes']:.1f} minutes

SIGNAL BREAKDOWN:
- RSI Signals: {metrics['rsi_signal_count']}
- MACD Signals: {metrics['macd_signal_count']}
- Sentiment Signals: {metrics['sentiment_signal_count']}
- Momentum Signals: {metrics['momentum_signal_count']}

LLM ENHANCEMENT:
- Enhanced Trades: {metrics['llm_enhanced_trades']}
- LLM Accuracy: {metrics['llm_accuracy']:.1%}
- Average Confidence: {metrics['llm_avg_confidence']:.1%}

IDENTIFIED PATTERNS:
"""
        
        if patterns:
            for i, pattern in enumerate(patterns, 1):
                prompt += f"""
{i}. [{pattern['severity'].upper()}] {pattern['pattern_type']}
   Description: {pattern['description']}
   Affected Trades: {pattern['affected_trades']}
   Impact: ${pattern['impact_pnl']:,.2f}
   Current Recommendation: {pattern['recommendation']}
"""
        else:
            prompt += "No significant patterns detected.\n"
        
        prompt += """
RECENT TRADE SAMPLE (Last 5):
"""
        for trade in recent_trades[:5]:
            prompt += f"""
- {trade['action']} {trade['quantity']} @ ${trade['entry_price']:.2f} → ${trade.get('exit_price', 0):.2f}
  Result: {trade['outcome']} | P&L: ${trade['realized_pnl']:.2f}
  Duration: {trade['trade_duration_minutes']:.1f} min
"""
            if trade.get('reasoning'):
                prompt += f"  LLM Reasoning: {trade['reasoning'][:100]}...\n"
        
        prompt += """

TASK:
Provide a concise natural-language summary of today's trading performance. Focus on:
1. What worked well and what didn't
2. Which signal types were most/least effective
3. Key takeaways about market conditions
4. Specific areas for improvement

Keep your summary to 3-4 paragraphs, written as if briefing a senior trader.
"""
        
        return prompt
    
    @staticmethod
    def self_assessment_prompt(
        daily_metrics: dict,
        historical_metrics: List[dict],
        patterns: List[dict],
        current_config: dict
    ) -> str:
        """Generate prompt for LLM self-assessment.
        
        Args:
            daily_metrics: Today's metrics
            historical_metrics: Past 7-30 days metrics
            patterns: Identified patterns
            current_config: Current strategy configuration
            
        Returns:
            Formatted prompt for LLM
        """
        # Calculate historical averages
        if historical_metrics:
            avg_win_rate = sum(m['win_rate'] for m in historical_metrics) / len(historical_metrics)
            avg_pnl = sum(m['net_pnl'] for m in historical_metrics) / len(historical_metrics)
            avg_trades = sum(m['total_trades'] for m in historical_metrics) / len(historical_metrics)
        else:
            avg_win_rate = avg_pnl = avg_trades = 0.0
        
        prompt = f"""You are an AI trading system conducting a self-assessment to improve performance.

CURRENT PERFORMANCE (Today):
- Win Rate: {daily_metrics['win_rate']:.1%}
- Net P&L: ${daily_metrics['net_pnl']:,.2f}
- Total Trades: {daily_metrics['total_trades']}
- Profit Factor: {daily_metrics['profit_factor']:.2f}
- Max Drawdown: ${daily_metrics['max_drawdown']:,.2f}

HISTORICAL AVERAGE (Last {len(historical_metrics)} days):
- Win Rate: {avg_win_rate:.1%}
- Net P&L: ${avg_pnl:,.2f}
- Trades/Day: {avg_trades:.1f}

CURRENT STRATEGY CONFIGURATION:
RSI Buy Threshold: {current_config.get('rsi_buy', 35)}
RSI Sell Threshold: {current_config.get('rsi_sell', 65)}
Sentiment Buy Threshold: {current_config.get('sentiment_buy', -0.5)}
Sentiment Sell Threshold: {current_config.get('sentiment_sell', 0.5)}
Stop Loss (ticks): {current_config.get('stop_loss_ticks', 20)}
Take Profit (ticks): {current_config.get('take_profit_ticks', 40)}
Max Position Size: {current_config.get('max_position_size', 2)}
LLM Confidence Threshold: {current_config.get('min_confidence_threshold', 0.7)}

IDENTIFIED BEHAVIORAL PATTERNS:
"""
        
        for i, pattern in enumerate(patterns, 1):
            prompt += f"""
{i}. {pattern['pattern_type']} (Severity: {pattern['severity']})
   {pattern['description']}
   Recommendation: {pattern['recommendation']}
"""
        
        prompt += """

ANALYSIS QUESTIONS:
1. Am I overreacting to certain signal types (RSI, MACD, sentiment, momentum)?
2. Am I missing opportunities due to overly conservative thresholds?
3. Are my stop-loss/take-profit ratios optimal for current market conditions?
4. Is my sentiment integration weight appropriate?
5. Should I adjust my confidence threshold for LLM-enhanced trades?

TASK:
Analyze the data and detect behavioral trends. Provide:
1. A brief assessment of what patterns you notice in your own trading behavior
2. Specific weaknesses or areas of overconfidence
3. Market condition insights (volatility, trend, sentiment effectiveness)

Write 2-3 paragraphs as if you're a trading system reflecting on its own decision-making patterns.
"""
        
        return prompt
    
    @staticmethod
    def strategy_adjustment_prompt(
        self_assessment: str,
        daily_metrics: dict,
        patterns: List[dict],
        current_config: dict,
        constraints: dict
    ) -> str:
        """Generate prompt for strategy parameter adjustments.
        
        Args:
            self_assessment: LLM's self-assessment text
            daily_metrics: Today's metrics
            patterns: Identified patterns
            current_config: Current configuration
            constraints: Safety constraints (min/max values)
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are an AI trading system optimization engine. Based on your self-assessment and performance data, suggest specific parameter adjustments.

YOUR SELF-ASSESSMENT:
{self_assessment}

CURRENT CONFIGURATION:
rsi_buy: {current_config.get('rsi_buy', 35)}
rsi_sell: {current_config.get('rsi_sell', 65)}
sentiment_buy: {current_config.get('sentiment_buy', -0.5)}
sentiment_sell: {current_config.get('sentiment_sell', 0.5)}
sentiment_weight: {current_config.get('sentiment_weight', 0.5)}
stop_loss_ticks: {current_config.get('stop_loss_ticks', 20)}
take_profit_ticks: {current_config.get('take_profit_ticks', 40)}
min_confidence_threshold: {current_config.get('min_confidence_threshold', 0.7)}

SAFETY CONSTRAINTS (Do not exceed these bounds):
- rsi_buy: [{constraints.get('rsi_buy_min', 20)} - {constraints.get('rsi_buy_max', 45)}]
- rsi_sell: [{constraints.get('rsi_sell_min', 55)} - {constraints.get('rsi_sell_max', 80)}]
- sentiment_buy: [{constraints.get('sentiment_buy_min', -1.0)} - {constraints.get('sentiment_buy_max', 0.0)}]
- sentiment_sell: [{constraints.get('sentiment_sell_min', 0.0)} - {constraints.get('sentiment_sell_max', 1.0)}]
- sentiment_weight: [0.1 - 0.9]
- stop_loss_ticks: [{constraints.get('stop_loss_min', 10)} - {constraints.get('stop_loss_max', 50)}]
- take_profit_ticks: [{constraints.get('take_profit_min', 15)} - {constraints.get('take_profit_max', 100)}]
- min_confidence_threshold: [0.5 - 0.9]

ADJUSTMENT GUIDELINES:
- Make incremental changes (5-15% adjustments, not drastic overhauls)
- Only suggest changes that address identified patterns
- Tighter thresholds = fewer trades, looser = more trades
- Higher confidence threshold = more conservative LLM filtering
- Consider risk/reward balance with stop-loss and take-profit ratios

TASK:
Output a JSON object with suggested parameter changes and reasoning. Format:

{{
  "suggested_changes": {{
    "parameter_name": new_value,
    ...
  }},
  "reasoning": "Brief explanation of why these changes address current issues",
  "expected_impact": "What you expect these changes to achieve",
  "confidence": 0.0-1.0,
  "risk_level": "low" | "medium" | "high"
}}

Only include parameters you want to change. Leave others unchanged.
Ensure all values are within the safety constraints.
Be conservative - it's better to suggest no changes than risky ones.
"""
        
        return prompt
    
    @staticmethod
    def weekly_review_prompt(
        weekly_metrics: List[dict],
        parameter_changes_history: List[dict],
        overall_performance: dict
    ) -> str:
        """Generate prompt for weekly cumulative review.
        
        Args:
            weekly_metrics: Metrics for each day of the week
            parameter_changes_history: History of parameter adjustments
            overall_performance: Aggregated weekly performance
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are conducting a weekly review of your autonomous trading system's performance and adaptations.

WEEKLY PERFORMANCE SUMMARY:
Total Trades: {overall_performance.get('total_trades', 0)}
Overall Win Rate: {overall_performance.get('win_rate', 0):.1%}
Total Net P&L: ${overall_performance.get('total_pnl', 0):,.2f}
Average Daily P&L: ${overall_performance.get('avg_daily_pnl', 0):,.2f}
Best Day: ${overall_performance.get('best_day_pnl', 0):,.2f}
Worst Day: ${overall_performance.get('worst_day_pnl', 0):,.2f}
Weekly Sharpe Ratio: {overall_performance.get('sharpe_ratio', 0):.2f}
Max Drawdown: ${overall_performance.get('max_drawdown', 0):,.2f}

DAILY BREAKDOWN:
"""
        
        for day_metrics in weekly_metrics:
            prompt += f"""
{day_metrics['date']}: {day_metrics['total_trades']} trades, {day_metrics['win_rate']:.1%} WR, ${day_metrics['net_pnl']:,.2f} P&L
"""
        
        prompt += """
PARAMETER ADJUSTMENTS MADE THIS WEEK:
"""
        
        if parameter_changes_history:
            for change in parameter_changes_history:
                prompt += f"""
Date: {change['timestamp']}
Changes: {change['changes']}
Reasoning: {change['reasoning']}
Performance After: {change.get('performance_after', 'Pending')}
"""
        else:
            prompt += "No parameter adjustments made this week.\n"
        
        prompt += """

ANALYSIS QUESTIONS:
1. Did the parameter adjustments improve performance?
2. Are there consistent patterns across multiple days?
3. Should we continue current parameters or make further adjustments?
4. Are there weekly/day-of-week patterns to consider?

TASK:
Provide a comprehensive weekly review that includes:
1. Overall assessment of the week's trading performance
2. Evaluation of whether parameter changes were beneficial
3. Identification of longer-term trends or patterns
4. Recommendations for next week's focus areas
5. Whether to keep current configuration or suggest further modifications

Write 4-5 paragraphs as a strategic review for improving the trading system.
"""
        
        return prompt
    
    @staticmethod
    def few_shot_examples() -> Dict[str, List[Dict[str, str]]]:
        """Provide few-shot examples for better LLM performance.
        
        Returns:
            Dictionary of example prompts and responses
        """
        return {
            "strategy_adjustment": [
                {
                    "scenario": "Low win rate with sentiment signals",
                    "assessment": "Sentiment signals showing 42% win rate vs 58% for RSI signals. Overtrading on sentiment spikes.",
                    "good_response": {
                        "suggested_changes": {
                            "sentiment_weight": 0.35,
                            "min_confidence_threshold": 0.75,
                            "sentiment_buy": -0.65
                        },
                        "reasoning": "Reduce sentiment influence by lowering weight to 0.35 and tightening buy threshold. Increase LLM confidence requirement to filter marginal sentiment trades.",
                        "expected_impact": "Fewer sentiment-driven trades, improved signal quality, win rate increase of 5-10%",
                        "confidence": 0.78,
                        "risk_level": "low"
                    }
                },
                {
                    "scenario": "High profit factor but low total P&L",
                    "assessment": "Profit factor 2.1 with only 8 trades/day. Missing opportunities due to overly tight thresholds.",
                    "good_response": {
                        "suggested_changes": {
                            "rsi_buy": 38,
                            "rsi_sell": 62,
                            "min_confidence_threshold": 0.65
                        },
                        "reasoning": "Loosen RSI thresholds slightly (35→38, 65→62) and reduce confidence requirement to capture more valid signals without sacrificing quality.",
                        "expected_impact": "Increase trade frequency by 20-30%, maintain or improve profit factor",
                        "confidence": 0.72,
                        "risk_level": "medium"
                    }
                },
                {
                    "scenario": "High drawdown with large losses",
                    "assessment": "Max drawdown $2,100 with avg loss 2x avg win. Stop losses hit frequently.",
                    "good_response": {
                        "suggested_changes": {
                            "stop_loss_ticks": 15,
                            "take_profit_ticks": 35,
                            "sentiment_weight": 0.45
                        },
                        "reasoning": "Tighten stop loss to limit downside (20→15 ticks) while maintaining favorable risk/reward. Reduce sentiment weight as it correlates with larger losses.",
                        "expected_impact": "Reduce max loss per trade, lower drawdown by 25-30%, improve risk management",
                        "confidence": 0.81,
                        "risk_level": "low"
                    }
                }
            ]
        }
    
    @staticmethod
    def format_llm_request(
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> dict:
        """Format prompt for AWS Bedrock Claude API.
        
        Args:
            prompt: The main prompt text
            system_message: Optional system message
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Formatted request dictionary
        """
        if system_message is None:
            system_message = """You are an expert AI trading system analyst with deep knowledge of:
- Technical analysis and trading strategies
- Risk management and position sizing
- Statistical performance evaluation
- Machine learning model behavior
- Market microstructure and conditions

Provide precise, data-driven analysis and recommendations. Be conservative with suggested changes.
Always prioritize risk management and capital preservation."""
        
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_message,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
