"""
src/visualizations.py — Plotly chart builders for SentimentIQ dashboard.

Monochrome palette (white page, black ink):
  Positive : #1d1d1f  (near-black)
  Neutral  : #8e8e93  (mid gray)
  Negative : #d2d2d7  (light gray, outlined for contrast)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------

PALETTE = {
    "Positive": "#1d1d1f",
    "Neutral":  "#8e8e93",
    "Negative": "#d2d2d7",
}
INK       = "#1d1d1f"
INK_SOFT  = "#6e6e73"
BG        = "rgba(0,0,0,0)"
GRID_COL  = "#e8e8ed"
FONT_FAM  = "-apple-system, BlinkMacSystemFont, 'SF Pro Display', Inter, 'Helvetica Neue', Arial, sans-serif"

_LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family=FONT_FAM, color=INK),
    margin=dict(t=24, b=32, l=32, r=24),
)

_AXIS = dict(showgrid=False, zeroline=False, color=INK_SOFT, linecolor=GRID_COL, tickfont=dict(size=12))
_AXIS_Y = dict(showgrid=True, gridcolor=GRID_COL, zeroline=False, color=INK_SOFT, tickfont=dict(size=12))


# ---------------------------------------------------------------------------
# Donut chart
# ---------------------------------------------------------------------------

def donut_chart(counts: dict[str, int]) -> go.Figure:
    """Sentiment distribution donut chart."""
    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = [PALETTE.get(l, "#8e8e93") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.72,
        sort=False,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=4)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))

    total = sum(values)
    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        legend=dict(font=dict(size=13, color=INK), orientation="h", x=0.5, xanchor="center", y=-0.05),
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:12px;color:{INK_SOFT}'>posts</span>",
            x=0.5, y=0.5,
            font=dict(size=34, color=INK, family=FONT_FAM),
            showarrow=False,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Hate-speech donut
# ---------------------------------------------------------------------------

def hate_donut(flagged: int, clear: int) -> go.Figure:
    """Share of posts flagged as hate speech vs not."""
    fig = go.Figure(go.Pie(
        labels=["Hate flagged", "No hate"],
        values=[flagged, clear],
        hole=0.72,
        sort=False,
        marker=dict(colors=["#1d1d1f", "#e8e8ed"], line=dict(color="#ffffff", width=4)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    pct = 0 if (flagged + clear) == 0 else round(100 * flagged / (flagged + clear))
    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        legend=dict(font=dict(size=13, color=INK), orientation="h", x=0.5, xanchor="center", y=-0.05),
        annotations=[dict(
            text=f"<b>{pct}%</b><br><span style='font-size:12px;color:{INK_SOFT}'>flagged</span>",
            x=0.5, y=0.5, font=dict(size=34, color=INK, family=FONT_FAM), showarrow=False,
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
    colors = [PALETTE.get(l, "#8e8e93") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors, line=dict(color=INK, width=[0, 0, 1])),
        text=values,
        textposition="outside",
        textfont=dict(color=INK, size=14, family=FONT_FAM),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
        cliponaxis=False,
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        xaxis=_AXIS,
        yaxis=_AXIS_Y,
        bargap=0.45,
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
                text="No timestamp data available",
                font=dict(color=INK_SOFT, size=13),
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

    # Distinguish lines by tone *and* dash style so the chart stays readable in grayscale.
    styles = {
        "Positive": dict(color=PALETTE["Positive"], dash="solid"),
        "Neutral":  dict(color=PALETTE["Neutral"],  dash="solid"),
        "Negative": dict(color=INK,                  dash="dot"),
    }

    fig = go.Figure()
    for label, st in styles.items():
        sub = grouped[grouped["predicted_sentiment"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["hour"],
            y=sub["count"],
            name=label,
            mode="lines+markers",
            line=dict(color=st["color"], width=2.5, dash=st["dash"]),
            marker=dict(size=7, color=st["color"]),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Count: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=320,
        xaxis=_AXIS,
        yaxis=_AXIS_Y,
        legend=dict(font=dict(size=12, color=INK), orientation="h", x=0, y=1.12),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_chart(cm: list[list[int]], labels: list[str]) -> go.Figure:
    """
    Annotated confusion matrix heatmap (white → black).

    Parameters
    ----------
    cm : list of lists
        Square confusion matrix (rows = true, cols = predicted).
    labels : list[str]
        Class label names corresponding to matrix rows/columns.
    """
    z = np.array(cm, dtype=float)
    z_norm = z / (z.sum(axis=1, keepdims=True) + 1e-9)

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            dark_cell = z_norm[i, j] > 0.5
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"<b>{int(val):,}</b>",
                font=dict(color="#ffffff" if dark_cell else INK, size=15, family=FONT_FAM),
                showarrow=False,
            ))

    fig = go.Figure(go.Heatmap(
        z=z_norm,
        x=labels,
        y=labels,
        colorscale=[[0.0, "#f5f5f7"], [1.0, "#000000"]],
        zmin=0, zmax=1,
        showscale=False,
        xgap=4, ygap=4,
        customdata=z,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{customdata:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        height=360,
        xaxis=dict(title=dict(text="Predicted", font=dict(color=INK_SOFT, size=12)), color=INK_SOFT, side="bottom"),
        yaxis=dict(title=dict(text="Actual", font=dict(color=INK_SOFT, size=12)), color=INK_SOFT, autorange="reversed"),
        annotations=annotations,
    )
    return fig
