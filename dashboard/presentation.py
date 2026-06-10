"""Visual components for the loan assessment dashboard."""

import base64
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2b4ecb"
RUST = "#af563b"
INK = "#202b3d"
MUTED = "#626e80"
LINE = "#d0d8e3"


def stylesheet():
    rules = []
    for family, name, weight in (
        ("Manrope", "manrope-regular.ttf", 400),
        ("Manrope", "manrope-semibold.ttf", 600),
        ("Manrope", "manrope-extrabold.ttf", 800),
        ("DM Mono", "dm-mono.ttf", 400),
    ):
        data = base64.b64encode((ROOT / "static/fonts" / name).read_bytes()).decode()
        rules.append(
            f"@font-face{{font-family:'{family}';font-weight:{weight};"
            f"src:url(data:font/ttf;base64,{data});font-display:swap}}"
        )
    return (
        "<style>"
        + "\n".join(rules)
        + (ROOT / "dashboard/assets/style.css").read_text()
        + "</style>"
    )


def html(content):
    st.markdown(content, unsafe_allow_html=True)


def risk_hero(probability, threshold, loan_amount, term_months):
    approved = probability <= threshold
    ticks = "".join(f'<span style="left:{i * 10}%">{i * 10}</span>' for i in range(11))
    html(f"""<div class="risk-hero">
<div class="hero-head"><span>DEFAULT RISK · CALIBRATED ESTIMATE</span><span class="model-label">${loan_amount:,.0f} / {term_months} MONTHS</span></div>
<div class="risk-summary"><div><div class="risk-number">{probability * 100:.1f}<small>%</small></div><div class="risk-number-caption">Estimated probability of default</div></div>
<div class="verdict"><div class="verdict-label">POLICY DECISION</div><h2>{"Approve" if approved else "Decline"} <span aria-hidden="true">{"↗" if approved else "↘"}</span></h2><p>{"At or below" if approved else "Above"} the {threshold:.2%} approval cutoff.</p><span class="verdict-badge">RESEARCH ESTIMATE</span></div></div>
<div class="risk-ruler"><div class="probability-track" role="img" aria-label="Default probability {probability:.2%}, approval cutoff {threshold:.2%}, on a zero to one hundred percent scale"><div class="approval-region" style="width:{threshold * 100:.4f}%"></div><span class="threshold-marker" style="left:{threshold * 100:.4f}%"></span><span class="probability-marker" style="left:{probability * 100:.4f}%"></span></div><div class="ruler-ticks" aria-hidden="true">{ticks}</div></div>
<div class="ruler-note"><span>● THIS APPLICATION</span><span>│ APPROVAL CUTOFF {threshold:.2%}</span><span>PROBABILITY, %</span></div></div>""")


def economics_row(profit, installment, interest, loss):
    html(f"""<div class="economics">
<div class="economics-item"><div class="economics-label">Expected profit / loan</div><div class="economics-value">{"−" if profit < 0 else ""}${abs(profit):,.0f}</div><div class="economics-note">${interest:,.0f} weighted interest − ${loss:,.0f} weighted loss</div></div>
<div class="economics-item"><div class="economics-label">Monthly installment</div><div class="economics-value">${installment:,.0f}</div><div class="economics-note">Fixed-rate amortization</div></div>
<div class="economics-item"><div class="economics-label">Loss assumption</div><div class="economics-value">35<span style="font-size:18px">%</span></div><div class="economics-note">Of the original loan principal</div></div></div>""")


def chart_layout(fig, height=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", size=11, color=MUTED),
        height=height,
        margin=dict(l=5, r=20, t=25, b=30),
        showlegend=False,
        hoverlabel=dict(bgcolor="#f8fafc", font_color=INK, font_family="Manrope"),
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, tickfont=dict(size=10))
    return fig


def profit_sensitivity(probability, interest, loss, threshold):
    probabilities = np.linspace(0, 1, 101)
    profits = (1 - probabilities) * interest - probabilities * loss
    current = (1 - probability) * interest - probability * loss
    fig = go.Figure(
        go.Scatter(
            x=probabilities * 100,
            y=profits,
            mode="lines",
            line=dict(color=BLUE, width=3),
            hovertemplate="Default probability %{x:.0f}%<br>Expected profit $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=MUTED)
    fig.add_vline(x=threshold * 100, line_width=1, line_dash="dot", line_color=RUST)
    fig.add_trace(
        go.Scatter(
            x=[probability * 100],
            y=[current],
            mode="markers",
            marker=dict(size=12, color=BLUE, line=dict(width=3, color="#edf1f5")),
            hovertemplate="This application<br>PD %{x:.1f}%<br>Expected profit $%{y:,.0f}<extra></extra>",
        )
    )
    chart_layout(fig)
    fig.update_xaxes(
        title="Default probability (%)", range=[0, 100], showgrid=False, dtick=20
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", nticks=5)
    return fig


def shap_chart(explanation, names):
    values = np.asarray(explanation.values[0])
    indices = np.argsort(np.abs(values))[-9:][::-1]
    other = float(values.sum() - values[indices].sum())
    labels = [
        names.get(explanation.feature_names[i], explanation.feature_names[i])
        for i in indices
    ]
    contributions = [float(values[i]) for i in indices]
    labels.append("Other features")
    contributions.append(other)
    fig = go.Figure(
        go.Bar(
            y=labels[::-1],
            x=contributions[::-1],
            orientation="h",
            marker_color=[RUST if v > 0 else BLUE for v in contributions[::-1]],
            hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
        )
    )
    chart_layout(fig, height=380)
    fig.update_layout(margin=dict(l=5, r=20, t=15, b=40))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(
        title="Contribution to default log-odds",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=MUTED,
    )
    return fig


def drivers(explanation, names, format_input):
    values = explanation.values[0]
    for idx in np.argsort(np.abs(values))[::-1][:3]:
        name = explanation.feature_names[idx]
        value = float(values[idx])
        html(
            f"""<div class="driver"><div><div class="driver-name">{escape(names.get(name, name))}</div><div class="driver-input">{escape(format_input(name, explanation.data[0][idx]))}</div></div><div class="driver-effect {"lower" if value < 0 else ""}">{value:+.3f}<small>{"Lowers" if value < 0 else "Raises"} estimated risk</small></div></div>"""
        )
