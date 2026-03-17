"""HTML dashboard generator for SPY options backtest results.

Produces a single self-contained HTML file with 8 Plotly chart sections.
All JavaScript is embedded inline — no internet connection required to open.

Exports:
    backtest_results/report.html    (full dashboard)
    backtest_results/trades.csv     (trade log)
    backtest_results/equity_curve.csv
    backtest_results/metrics.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.backtest_engine import BacktestResults, SimulatedTrade

OUTPUT_DIR = Path("backtest_results")


class Reporter:
    def __init__(
        self,
        results: BacktestResults,
        metrics: dict,
        output_dir: Path = OUTPUT_DIR,
    ) -> None:
        self.results = results
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self) -> dict[str, Path]:
        """Generate all output files. Returns dict of {name: path}."""
        paths: dict[str, Path] = {}
        paths["trades_csv"] = self._export_trades_csv()
        paths["equity_csv"] = self._export_equity_csv()
        paths["metrics_json"] = self._export_metrics_json()
        paths["report_html"] = self._build_html_report()
        return paths

    # ------------------------------------------------------------------
    # CSV / JSON exports
    # ------------------------------------------------------------------

    def _export_trades_csv(self) -> Path:
        path = self.output_dir / "trades.csv"
        if not self.results.trades:
            pd.DataFrame().to_csv(path)
            return path
        rows = []
        for t in self.results.trades:
            rows.append({
                "trade_id": t.trade_id,
                "entry_date": t.entry_date,
                "expiry": t.expiry,
                "right": t.right,
                "strike": t.strike,
                "contracts": t.contracts,
                "entry_premium": t.entry_premium,
                "close_premium": t.close_premium,
                "gross_pnl": t.gross_pnl,
                "commissions": t.commissions,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "days_held": t.days_held,
                "entry_delta": t.entry_delta,
                "entry_theta": t.entry_theta,
                "entry_iv": round(t.entry_iv * 100, 1),
                "entry_spy_price": t.entry_spy_price,
                "entry_vix": t.entry_vix,
                "strategy": t.strategy,
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _export_equity_csv(self) -> Path:
        path = self.output_dir / "equity_curve.csv"
        if not self.results.equity_curve.empty:
            self.results.equity_curve.to_csv(path)
        return path

    def _export_metrics_json(self) -> Path:
        path = self.output_dir / "metrics.json"
        path.write_text(json.dumps(self.metrics, indent=2, default=str))
        return path

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _build_html_report(self) -> Path:
        path = self.output_dir / "report.html"

        chart_divs: list[str] = []

        # First chart includes plotly.js (embedded, ~3MB)
        first = True

        def _add(fig: go.Figure, title: str = "") -> None:
            nonlocal first
            div = fig.to_html(
                include_plotlyjs=first,
                full_html=False,
                config={"displayModeBar": True, "scrollZoom": True},
            )
            first = False
            if title:
                chart_divs.append(f'<h2 class="section-title">{title}</h2>')
            chart_divs.append(div)

        m = self.metrics
        trades = self.results.trades
        real_trades = [t for t in trades if t.exit_reason != "end_of_backtest"]
        eq = self.results.equity_curve

        # Section 1 — Summary cards (rendered as a Table figure)
        _add(self._summary_cards(m), "Summary")

        # Section 2 — Equity curve
        _add(self._equity_curve_chart(eq, m), "Equity Curve vs SPY Buy-and-Hold")

        # Section 3 — Monthly returns heatmap
        _add(self._monthly_heatmap(m), "Monthly Returns Heatmap")

        # Section 4 — P&L distribution
        _add(self._pnl_histogram(real_trades), "Trade P&L Distribution")

        # Section 5 — Exit reason pie
        _add(self._exit_reason_chart(m), "Exit Reason Breakdown")

        # Section 6 — Greeks scatter plots
        _add(self._greeks_scatter(real_trades), "Greeks at Entry vs Trade P&L")

        # Section 7 — Trade log table
        _add(self._trade_table(real_trades), "Full Trade Log")

        # Section 8 — PDT impact analysis
        _add(self._pdt_impact_chart(real_trades, m), "PDT Impact Analysis")

        html = _HTML_TEMPLATE.format(
            title=f"SPY Options Backtest — {m.get('start_date', '')} to {m.get('end_date', '')}",
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            body="\n".join(chart_divs),
        )
        path.write_text(html, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Individual chart builders
    # ------------------------------------------------------------------

    def _summary_cards(self, m: dict) -> go.Figure:
        labels = [
            "Total Return", "Annual Return", "Win Rate",
            "Sharpe Ratio", "Max Drawdown", "Profit Factor",
            "Total Trades", "Avg P&L / Trade", "Expected Value",
        ]
        values = [
            f"{m.get('total_return_pct', 0):+.2f}%",
            f"{m.get('annualized_return_pct', 0):+.2f}%",
            f"{m.get('win_rate_pct', 0):.1f}%  ({m.get('winning_trades',0)}/{m.get('total_trades',0)})",
            f"{m.get('sharpe_ratio', 0):.2f}",
            f"{m.get('max_drawdown_pct', 0):.2f}%",
            f"{m.get('profit_factor', 0):.2f}",
            str(m.get("total_trades", 0)),
            f"${m.get('avg_pnl_per_trade', 0):+.2f}",
            f"${m.get('expected_value', 0):+.2f}",
        ]
        colors = []
        for v in values[:3]:
            colors.append("#27ae60" if "+" in v or (v[0].isdigit()) else "#e74c3c")
        colors += ["#2c3e50"] * (len(labels) - 3)

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{l}</b>" for l in labels],
                fill_color="#2c3e50",
                font=dict(color="white", size=13),
                align="center",
                height=40,
            ),
            cells=dict(
                values=[values],
                fill_color=[["#ecf0f1"] * len(labels)],
                font=dict(size=15, color=["#27ae60" if "+" in str(v) else "#e74c3c"
                                          if "-" in str(v) else "#2c3e50" for v in values]),
                align="center",
                height=45,
            ),
        ))
        fig.update_layout(
            height=160, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#f8f9fa",
        )
        return fig

    def _equity_curve_chart(self, eq: pd.DataFrame, m: dict) -> go.Figure:
        fig = go.Figure()
        if eq.empty:
            fig.update_layout(title="No equity data")
            return fig

        dates = eq.index
        equity = eq["equity"].values
        initial = m.get("initial_capital", equity[0])

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=dates, y=equity, name="Strategy",
            line=dict(color="#2980b9", width=2),
        ))

        # SPY buy-and-hold benchmark (normalise from same starting capital)
        from backtest.data_downloader import load_spy_daily
        try:
            spy_df = load_spy_daily(self.results.equity_curve.index[0].date().__class__.__mro__[0])
        except Exception:
            spy_df = None

        if spy_df is not None and not spy_df.empty:
            spy_in_range = spy_df[spy_df.index >= dates[0]]
            if not spy_in_range.empty:
                spy_ret = spy_in_range["close"] / float(spy_in_range["close"].iloc[0])
                spy_equity = spy_ret * initial
                fig.add_trace(go.Scatter(
                    x=spy_in_range.index, y=spy_equity, name="SPY Buy-Hold",
                    line=dict(color="#95a5a6", width=1.5, dash="dash"),
                ))

        # Drawdown shading
        peak = pd.Series(equity, index=dates).cummax()
        in_dd = equity < peak.values
        dd_start = None
        for i, flag in enumerate(in_dd):
            if flag and dd_start is None:
                dd_start = dates[i]
            elif not flag and dd_start is not None:
                fig.add_vrect(
                    x0=dd_start, x1=dates[i - 1],
                    fillcolor="rgba(231,76,60,0.15)", layer="below", line_width=0,
                )
                dd_start = None
        if dd_start is not None:
            fig.add_vrect(
                x0=dd_start, x1=dates[-1],
                fillcolor="rgba(231,76,60,0.15)", layer="below", line_width=0,
            )

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Portfolio Value ($)",
            legend=dict(x=0, y=1), hovermode="x unified",
            plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            height=420, margin=dict(l=60, r=20, t=20, b=40),
        )
        return fig

    def _monthly_heatmap(self, m: dict) -> go.Figure:
        monthly = m.get("monthly_returns", [])
        if not monthly:
            fig = go.Figure()
            fig.update_layout(title="No monthly data")
            return fig

        df = pd.DataFrame(monthly)
        df["year"] = df["month"].str[:4]
        df["mon"] = df["month"].str[5:].astype(int)
        years = sorted(df["year"].unique())
        months = list(range(1, 13))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        z = []
        text = []
        for yr in years:
            row_z, row_t = [], []
            for mo in months:
                sub = df[(df["year"] == yr) & (df["mon"] == mo)]
                if sub.empty:
                    row_z.append(None)
                    row_t.append("")
                else:
                    val = float(sub["return_pct"].iloc[0])
                    row_z.append(val)
                    row_t.append(f"{val:+.1f}%")
            z.append(row_z)
            text.append(row_t)

        fig = go.Figure(go.Heatmap(
            z=z, x=month_names, y=years,
            text=text, texttemplate="%{text}",
            colorscale=[[0, "#c0392b"], [0.5, "#f8f9fa"], [1, "#27ae60"]],
            zmid=0,
            colorbar=dict(title="Return %"),
        ))
        fig.update_layout(
            height=max(200, 80 * len(years)),
            xaxis_title="Month", yaxis_title="Year",
            plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            margin=dict(l=60, r=60, t=20, b=40),
        )
        return fig

    def _pnl_histogram(self, trades: list[SimulatedTrade]) -> go.Figure:
        if not trades:
            return go.Figure()
        pnls = [t.net_pnl for t in trades]
        mean_pnl = sum(pnls) / len(pnls)
        std_pnl = (sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)) ** 0.5

        colors = ["#27ae60" if p > 0 else "#e74c3c" for p in pnls]
        fig = go.Figure(go.Histogram(
            x=pnls,
            marker_color=colors,
            nbinsx=30,
            name="P&L",
        ))
        for val, label, color in [
            (mean_pnl, f"Mean ${mean_pnl:+.0f}", "#2980b9"),
            (0, "Breakeven", "#95a5a6"),
            (mean_pnl - std_pnl, f"-1σ ${mean_pnl - std_pnl:+.0f}", "#e67e22"),
            (mean_pnl + std_pnl, f"+1σ ${mean_pnl + std_pnl:+.0f}", "#27ae60"),
        ]:
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text=label, annotation_position="top")

        fig.update_layout(
            xaxis_title="Net P&L per Trade ($)", yaxis_title="Frequency",
            plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            height=380, margin=dict(l=60, r=20, t=20, b=40),
            bargap=0.05,
        )
        return fig

    def _exit_reason_chart(self, m: dict) -> go.Figure:
        counts = m.get("exit_reason_counts", {})
        if not counts:
            return go.Figure()
        labels = list(counts.keys())
        values = list(counts.values())
        colors = {
            "profit_target": "#27ae60",
            "loss_stop": "#e74c3c",
            "delta_stop": "#e67e22",
            "thursday_eod": "#2980b9",
            "emergency_gamma": "#8e44ad",
        }
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.4,
            marker_colors=[colors.get(l, "#95a5a6") for l in labels],
            textinfo="label+percent+value",
        ))
        fig.update_layout(
            height=380, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        return fig

    def _greeks_scatter(self, trades: list[SimulatedTrade]) -> go.Figure:
        if not trades:
            return go.Figure()
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Entry Delta vs P&L", "Entry Theta vs P&L", "VIX at Entry vs P&L"),
        )
        colors = ["#27ae60" if t.net_pnl > 0 else "#e74c3c" for t in trades]
        pnls = [t.net_pnl for t in trades]

        for col, (xs, xlabel) in enumerate([
            ([t.entry_delta for t in trades], "Entry Delta"),
            ([t.entry_theta for t in trades], "Entry Theta"),
            ([t.entry_vix for t in trades], "VIX at Entry"),
        ], start=1):
            fig.add_trace(go.Scatter(
                x=xs, y=pnls, mode="markers",
                marker=dict(color=colors, size=7, opacity=0.7),
                text=[t.trade_id for t in trades],
                hovertemplate=f"{xlabel}: %{{x:.3f}}<br>Net P&L: $%{{y:.2f}}<extra></extra>",
                showlegend=False,
            ), row=1, col=col)
            fig.update_xaxes(title_text=xlabel, row=1, col=col)

        fig.update_yaxes(title_text="Net P&L ($)", row=1, col=1)
        fig.update_layout(
            height=380, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            margin=dict(l=60, r=20, t=40, b=40),
        )
        return fig

    def _trade_table(self, trades: list[SimulatedTrade]) -> go.Figure:
        if not trades:
            return go.Figure()
        cols = ["trade_id", "entry_date", "expiry", "right", "strike",
                "contracts", "entry_premium", "close_premium", "net_pnl",
                "exit_reason", "days_held"]
        header_labels = ["ID", "Entry", "Expiry", "R", "Strike",
                         "Qty", "Entry $", "Close $", "Net P&L", "Exit", "Days"]

        def fmt(t: SimulatedTrade, col: str) -> str:
            val = getattr(t, col)
            if col == "net_pnl":
                return f"${val:+.2f}"
            if col in ("entry_premium", "close_premium"):
                return f"${val:.2f}"
            return str(val)

        cell_vals = [[fmt(t, c) for t in trades] for c in cols]
        row_colors = ["rgba(39,174,96,0.15)" if t.net_pnl > 0 else "rgba(231,76,60,0.15)"
                      for t in trades]

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_labels],
                fill_color="#2c3e50",
                font=dict(color="white", size=12),
                align="center", height=34,
            ),
            cells=dict(
                values=cell_vals,
                fill_color=[row_colors],
                font=dict(size=11),
                align="center", height=28,
            ),
        ))
        fig.update_layout(
            height=min(600, 80 + len(trades) * 30),
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#f8f9fa",
        )
        return fig

    def _pdt_impact_chart(self, trades: list[SimulatedTrade], m: dict) -> go.Figure:
        """Bar: average weekly P&L grouped by number of PDT slots consumed that week."""
        if not trades:
            return go.Figure()

        # Group trades by ISO week, sum P&L and count slots
        from collections import defaultdict
        week_pnl: dict[str, float] = defaultdict(float)
        week_slots: dict[str, int] = defaultdict(int)

        for t in trades:
            key = t.close_date.strftime("%Y-W%W")
            week_pnl[key] += t.net_pnl
            week_slots[key] += 1  # 1 slot per closed leg

        # Group by slot count
        slots_to_pnl: dict[int, list[float]] = defaultdict(list)
        for key in week_pnl:
            slots = min(week_slots[key], 3)
            slots_to_pnl[slots].append(week_pnl[key])

        slot_labels = sorted(slots_to_pnl.keys())
        avg_pnls = [sum(slots_to_pnl[s]) / len(slots_to_pnl[s]) for s in slot_labels]
        counts = [len(slots_to_pnl[s]) for s in slot_labels]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{s} slot(s)\n({counts[i]} weeks)" for i, s in enumerate(slot_labels)],
            y=avg_pnls,
            marker_color=["#27ae60" if v >= 0 else "#e74c3c" for v in avg_pnls],
            text=[f"${v:+.0f}" for v in avg_pnls],
            textposition="auto",
            name="Avg Weekly P&L",
        ))
        fig.update_layout(
            xaxis_title="PDT Slots Used That Week",
            yaxis_title="Avg Weekly Net P&L ($)",
            height=360, plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        return fig


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f0f2f5; color: #2c3e50; }}
    .header {{
      background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%);
      color: white; padding: 28px 40px;
    }}
    .header h1 {{ font-size: 1.7rem; font-weight: 600; margin-bottom: 4px; }}
    .header small {{ opacity: 0.7; font-size: 0.85rem; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 20px; }}
    .section {{ background: white; border-radius: 10px; padding: 20px 24px;
                margin-bottom: 24px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
    .section-title {{
      font-size: 1.1rem; font-weight: 600; color: #2c3e50;
      margin: 12px 0 6px; padding-left: 4px;
      border-left: 4px solid #2980b9;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <small>Generated {generated_at}</small>
  </div>
  <div class="container">
    <div class="section">
      {body}
    </div>
  </div>
</body>
</html>
"""
