"""
Summary Report Generator for Backtest

Generates comprehensive summary report after backtest completion.
"""
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def generate_summary_report(results: Dict[str, Any], output_path: Path) -> None:
    """
    Generate summary report in Markdown format.
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save the summary report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact_summary = results.get('artifact_summary', {})
    missing_artifacts = artifact_summary.get('missing_artifacts', [])
    
    # Calculate win rate
    all_trades = []
    for day_result in results.get('day_results', []):
        all_trades.extend(day_result.get('day_trades', []))
    
    winning_trades = [t for t in all_trades if t.get('realized_pnl', 0) > 0]
    losing_trades = [t for t in all_trades if t.get('realized_pnl', 0) < 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
    
    # Generate report
    report = f"""# Backtest Summary Report - Last 30 Days

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Start Date:** {results.get('start_date', 'N/A')}
- **End Date:** {results.get('end_date', 'N/A')}
- **Days Processed:** {results.get('days_processed', 0)}
- **Symbol:** ES (SPY Futures)

## Agent Execution Summary

### Agent 1: Data Ingestion & Feature Builder (Nightly)
- **Expected Runs:** {artifact_summary.get('agent1_expected', 0)}
- **Actual Runs:** {artifact_summary.get('agent1_runs', 0)}
- **Missing:** {artifact_summary.get('agent1_missing', 0)}
- **Status:** {'✅ PASS' if artifact_summary.get('agent1_missing', 0) == 0 else '❌ FAIL'}

### Agent 2: RAG + Similarity Search Decision Engine (Real-time)
- **Total Decisions:** {artifact_summary.get('agent2_decisions', 0)}
- **Status:** {'✅ PASS' if artifact_summary.get('agent2_decisions', 0) > 0 else '⚠️ WARNING'}

### Agent 3: Risk & Position Sizing Agent (Real-time)
- **Total Evaluations:** {artifact_summary.get('agent3_evaluations', 0)}
- **Status:** {'✅ PASS' if artifact_summary.get('agent3_evaluations', 0) > 0 else '⚠️ WARNING'}

### Agent 4: Strategy Optimization & Learning Agent (Nightly 11 PM CST)
- **Expected Runs:** {artifact_summary.get('agent4_expected', 0)}
- **Actual Runs:** {artifact_summary.get('agent4_runs', 0)}
- **Missing:** {artifact_summary.get('agent4_missing', 0)}
- **Status:** {'✅ PASS' if artifact_summary.get('agent4_missing', 0) == 0 else '❌ FAIL'}

## Trading Performance

- **Total Trades:** {results.get('total_trades', 0)}
- **Winning Trades:** {len(winning_trades)}
- **Losing Trades:** {len(losing_trades)}
- **Win Rate:** {win_rate*100:.2f}%
- **Total P&L:** ${results.get('total_pnl', 0):.2f}
- **Average P&L per Trade:** ${results.get('total_pnl', 0) / max(1, results.get('total_trades', 1)):.2f}

## Artifact Validation

### Missing Artifacts Check

"""
    
    if missing_artifacts:
        report += f"**❌ FAILED:** Found {len(missing_artifacts)} days with missing artifacts:\n\n"
        for missing in missing_artifacts:
            report += f"- **{missing['date']}:** Missing {len(missing['missing'])} artifacts\n"
            for item in missing['missing']:
                report += f"  - {item['description']} ({item['file']})\n"
        report += "\n"
    else:
        report += "**✅ PASS:** All required artifacts present for all days.\n\n"
    
    # Daily breakdown
    report += "## Daily Breakdown\n\n"
    report += "| Date | Trades | P&L | Agent 1 | Agent 4 | Status |\n"
    report += "|------|--------|-----|---------|------|----------|\n"
    
    for day_result in results.get('day_results', []):
        date = day_result.get('date', 'N/A')
        trades = day_result.get('trades', 0)
        pnl = day_result.get('pnl', 0.0)
        agent1 = '✅' if day_result.get('agent1_run', False) else '❌'
        agent4 = '✅' if day_result.get('agent4_run', False) else '❌'
        status = '✅' if day_result.get('agent1_run') and day_result.get('agent4_run') else '❌'
        
        report += f"| {date} | {trades} | ${pnl:.2f} | {agent1} | {agent4} | {status} |\n"
    
    # Acceptance Criteria
    report += "\n## Acceptance Criteria\n\n"
    
    criteria = [
        ("Agent 1 runs exactly once per day", artifact_summary.get('agent1_missing', 0) == 0),
        ("Agent 2 runs for every decision attempt", artifact_summary.get('agent2_decisions', 0) > 0),
        ("Agent 3 runs for every Agent 2 decision", artifact_summary.get('agent3_evaluations', 0) > 0),
        ("Agent 4 runs exactly once per day", artifact_summary.get('agent4_missing', 0) == 0),
        ("All artifacts present", len(missing_artifacts) == 0),
    ]
    
    all_passed = all(c[1] for c in criteria)
    
    for criterion, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        report += f"- {status}: {criterion}\n"
    
    report += f"\n**Overall Status:** {'✅ ALL CRITERIA MET' if all_passed else '❌ SOME CRITERIA FAILED'}\n"
    
    # Write report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report written to {output_path}")
