"""
src/visualizations.py — Plotly chart builders for SentimentIQ dashboard.

All charts use the dark-dashboard palette:
  Background : transparent (rendered on #0e1117 / navy cards)
  Positive   : #34d399  (emerald)
  Negative   : #f87171  (red)
  Neutral    : #fbbf24  (amber)
  Accent     : #a78bfa  (purple)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------

PALETTE = {
    "Positive": "#34d399",
    "Negative": "#f87171",
    "Neutral":  "#fbbf24",
}
BG        = "rgba(0,0,0,0)"
FONT_COL  = "#c4b5fd"
GRID_COL  = "rgba(255,255,255,0.06)"
FONT_FAM  = "Inter, sans-serif"

_LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family=FONT_FAM, color="#d1d5db"),
    margin=dict(t=40, b=40, l=40, r=40),
)


def _apply_base(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**_LAYOUT_BASE, **extra)
    return fig


# ---------------------------------------------------------------------------
# Donut chart
# ---------------------------------------------------------------------------

def donut_chart(counts: dict[str, int]) -> go.Figure:
    """Sentiment distribution donut chart."""
    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = [PALETTE.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.60,
        marker=dict(colors=colors, line=dict(color="#0d0d1a", width=3)),
        textfont=dict(family=FONT_FAM, size=13, color="#ffffff"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))

    total = sum(values)
    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        legend=dict(font=dict(size=13), bgcolor=BG, bordercolor=BG),
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:11px'>Posts</span>",
            x=0.5, y=0.5,
            font=dict(size=18, color=FONT_COL, family=FONT_FAM),
            showarrow=False,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def bar_chart(counts: dict[str, int]) -> go.Figure:
    """Vertical sentiment count bar chart."""
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [PALETTE.get(l, "#94a3b8") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        text=values,
        textposition="outside",
        textfont=dict(color="#d1d5db", size=14, family=FONT_FAM),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, color="#6b7280"),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COL,
            zeroline=False,
            color="#6b7280",
        ),
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# Time-series trend
# ---------------------------------------------------------------------------

def trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Sentiment count over time (hourly buckets).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``created_at`` (datetime) and ``predicted_sentiment`` columns.
    """
    if "created_at" not in df.columns or df["created_at"].isna().all():
        fig = go.Figure()
        fig.update_layout(
            **_LAYOUT_BASE,
            height=260,
            title=dict(
                text="Sentiment Trend — no timestamp data available",
                font=dict(color="#6b7280", size=13),
            ),
        )
        return fig

    tmp = df.dropna(subset=["created_at"]).copy()
    tmp["hour"] = pd.to_datetime(tmp["created_at"]).dt.floor("h")
    grouped = (
        tmp.groupby(["hour", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()
    for label, color in PALETTE.items():
        sub = grouped[grouped["predicted_sentiment"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["hour"],
            y=sub["count"],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, color="#6b7280"),
        yaxis=dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, color="#6b7280"),
        legend=dict(font=dict(size=12), bgcolor=BG, bordercolor=BG),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_chart(cm: list[list[int]], labels: list[str]) -> go.Figure:
    """
    Annotated confusion matrix heatmap.

    Parameters
    ----------
    cm : list of lists
        Square confusion matrix (rows = true, cols = predicted).
    labels : list[str]
        Class label names corresponding to matrix rows/columns.
    """
    z = np.array(cm, dtype=float)

    # Normalise for colour (keep raw counts for annotation)
    z_norm = z / (z.sum(axis=1, keepdims=True) + 1e-9)

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=str(int(val)),
                font=dict(color="white", size=14, family=FONT_FAM),
                showarrow=False,
            ))

    fig = go.Figure(go.Heatmap(
        z=z_norm,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.5, "#4c1d95"],
            [1.0, "#a78bfa"],
        ],
        showscale=False,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=340,
        xaxis=dict(title="Predicted Label", color="#6b7280", side="bottom"),
        yaxis=dict(title="True Label", color="#6b7280", autorange="reversed"),
        annotations=annotations,
    )
    return fig
