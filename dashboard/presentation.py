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
