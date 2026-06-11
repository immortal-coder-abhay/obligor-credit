"""Loan assessment using the saved Lending Club model and calibrator."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

from credit_risk import models as cr_models
from presentation import (
    drivers,
    economics_row,
    html,
    profit_sensitivity,
    risk_hero,
    shap_chart,
    stylesheet,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
OPTIMAL_THRESHOLD = 0.2807
LGD_FRACTION = 0.35
FRIENDLY_NAMES = {
    "int_rate": "Interest rate",
    "sub_grade": "Loan subgrade",
    "grade": "Loan grade",
    "dti": "Debt-to-income ratio",
    "annual_inc": "Annual income",
    "fico_range_low": "FICO score",
    "fico_range_high": "FICO score (upper)",
    "loan_amnt": "Loan amount",
    "term": "Loan term",
    "installment": "Monthly installment",
    "installment_to_income": "Installment / income",
    "emp_length_years": "Employment length",
    "purpose": "Loan purpose",
    "home_ownership": "Home ownership",
    "verification_status": "Income verification",
    "revol_util": "Revolving utilization",
    "revol_bal": "Revolving balance",
    "open_acc": "Open credit accounts",
    "total_acc": "Total credit accounts",
    "delinq_2yrs": "Delinquencies (2 years)",
    "inq_last_6mths": "Recent credit inquiries",
    "pub_rec": "Public records",
    "addr_state": "Borrower state",
    "credit_history_years": "Credit history",
}
st.set_page_config(
    page_title="Obligor Credit · Loan assessment", page_icon="↗", layout="wide"
)
html(stylesheet())
# Supply accessible labels for native controls in the pinned Streamlit release.
components.html(
    (ASSETS_DIR / "accessibility.html").read_text(), height=0, scrolling=False
)


@st.cache_resource
def load_model_bundle():
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm.txt"))
    iso = joblib.load(MODELS_DIR / "isotonic_lgb.joblib")
    return booster, iso, shap.TreeExplainer(booster)


@st.cache_data
def load_feature_defaults():
    return json.loads((ASSETS_DIR / "feature_defaults.json").read_text())


def index_of(options, value):
    return options.index(value) if value in options else 0


def format_input(name, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not recorded"
    if isinstance(value, (int, float, np.number)):
        if name in {"annual_inc", "loan_amnt", "installment", "revol_bal"}:
            return f"${value:,.0f}"
        if name in {"int_rate", "revol_util", "dti"}:
            return f"{value:.1f}%"
        if name == "installment_to_income":
            return f"{value:.1%}"
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value).strip().replace("_", " ")


try:
    booster, iso, explainer = load_model_bundle()
    snapshot = load_feature_defaults()
except (FileNotFoundError, OSError, ValueError, lgb.basic.LightGBMError) as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.info(
        "Restore the files in models/ and dashboard/assets/feature_defaults.json, then reload the app."
    )
    st.stop()

defaults, cat_options, feature_order = (
    snapshot["defaults"],
    snapshot["cat_options"],
    snapshot["feature_order"],
)
with st.sidebar:
    html(
        '<div class="brand"><div class="brand-mark">o<span>↗</span></div><div>Obligor Credit<small>RESEARCH WORKSPACE</small></div></div>'
    )
    html('<div class="side-heading">APPLICATION INPUTS</div>')
    st.caption("Change the terms, then select Assess loan.")
    with st.form("application"):
        html('<div class="side-section">Loan terms</div>')
        loan_amnt = st.number_input("Loan amount ($)", 1_000, 40_000, 15_000, step=500)
        term_months = st.radio("Term (months)", [36, 60], horizontal=True)
        int_rate = st.slider("Interest rate (%)", 5.0, 30.0, 13.0, 0.25)
        purpose = st.selectbox(
            "Purpose",
            cat_options["purpose"],
            index=index_of(cat_options["purpose"], "debt_consolidation"),
            format_func=lambda value: value.replace("_", " ").capitalize(),
        )
        html('<div class="side-section">Borrower</div>')
        annual_inc = st.number_input(
            "Annual income ($)", 10_000, 500_000, 65_000, step=1_000
        )
        dti = st.slider("Debt-to-income ratio (%)", 0.0, 50.0, 18.0, 0.5)
        fico = st.slider("FICO score (range low)", 660, 845, 690)
        with st.expander("Credit profile"):
            emp_length = st.slider("Employment length (years)", 0.0, 10.0, 5.0, 0.5)
            home_ownership = st.selectbox(
                "Home ownership",
                cat_options["home_ownership"],
                index=index_of(cat_options["home_ownership"], "MORTGAGE"),
                format_func=str.title,
            )
            verification = st.selectbox(
                "Income verification", cat_options["verification_status"]
            )
        st.form_submit_button("Assess loan ↗", use_container_width=True)
    html(
        '<div class="side-note">LENDING CLUB · USD<br>Historical loan data, 2007–2016.<br>Unspecified fields use training defaults.</div>'
    )

term_str = next(s for s in cat_options["term"] if str(term_months) in s)
r = int_rate / 100 / 12
installment = (
    loan_amnt * r / (1 - (1 + r) ** (-term_months)) if r else loan_amnt / term_months
)
features = dict(defaults)
features.update(
    {
        "loan_amnt": float(loan_amnt),
        "int_rate": float(int_rate),
        "installment": float(installment),
        "term": term_str,
        "purpose": purpose,
        "annual_inc": float(annual_inc),
        "dti": float(dti),
        "fico_range_low": float(fico),
        "fico_range_high": float(fico + 4),
        "emp_length_years": float(emp_length),
        "home_ownership": home_ownership,
        "verification_status": verification,
        "installment_to_income": float(installment * 12 / annual_inc),
    }
)
X_one = pd.DataFrame([features])[feature_order]
for col in X_one.columns:
    if not pd.api.types.is_numeric_dtype(X_one[col]):
        X_one[col] = X_one[col].astype(object)
X_lgb = cr_models.prepare_for_lgb(X_one)
p_raw = float(booster.predict(X_lgb)[0])
p_cal = float(iso.predict(np.array([p_raw]))[0])
approved = p_cal <= OPTIMAL_THRESHOLD
expected_interest = installment * term_months - loan_amnt
expected_loss = loan_amnt * LGD_FRACTION
weighted_interest = (1 - p_cal) * expected_interest
weighted_loss = p_cal * expected_loss
expected_profit = weighted_interest - weighted_loss

html(
    '<div class="topline"><span>Risk analytics<span class="divider">/</span>Loan assessment</span><span class="live-status"><i></i>Saved model loaded</span></div>'
)
title_col, export_col = st.columns([4, 1], vertical_alignment="center")
with title_col:
    html(
        '<p class="eyebrow">OBLIGOR CREDIT / CREDIT ASSESSMENT</p><h1 class="page-title">Loan assessment<span class="title-dot">.</span></h1><p class="page-description">Review default risk, return and the factors behind the estimate.</p>'
    )
with export_col:
    report = {
        "application": features,
        "default_probability": p_cal,
        "raw_probability": p_raw,
        "decision": "approve" if approved else "decline",
        "threshold": OPTIMAL_THRESHOLD,
        "expected_profit": expected_profit,
        "monthly_installment": installment,
        "loss_fraction": LGD_FRACTION,
        "model": "lightgbm.txt + isotonic_lgb.joblib",
        "scope": "Historical Lending Club model; threshold selected on the 2016 evaluation cohort.",
    }
    st.download_button(
        "Export review ↗",
        json.dumps(report, indent=2),
        "obligor-credit-review.json",
        "application/json",
        use_container_width=True,
    )
risk_hero(p_cal, OPTIMAL_THRESHOLD, loan_amnt, term_months)
economics_row(expected_profit, installment, weighted_interest, weighted_loss)

tab_review, tab_factors, tab_method = st.tabs(
    ["Risk & return", "Risk factors", "Model notes"]
)
with tab_review:
    chart_col, facts_col = st.columns([1.65, 1], gap="large")
    with chart_col:
        html(
            '<p class="section-kicker">PROBABILITY SENSITIVITY</p><h2 class="section-title">Return sensitivity</h2><p class="section-copy">Expected profit across default probabilities, holding this loan’s terms fixed. The dot marks this assessment; the dotted line marks the policy cutoff.</p>'
        )
        st.plotly_chart(
            profit_sensitivity(
                p_cal, expected_interest, expected_loss, OPTIMAL_THRESHOLD
            ),
            use_container_width=True,
            config={"displayModeBar": False, "scrollZoom": False},
        )
    with facts_col:
        html(
            f'<div class="fact-sheet"><p class="section-kicker">LOAN ECONOMICS</p><div class="fact-row"><span>Principal</span><strong>${loan_amnt:,.0f}</strong></div><div class="fact-row"><span>Annual interest rate</span><strong>{int_rate:.2f}%</strong></div><div class="fact-row"><span>Interest if repaid</span><strong>${expected_interest:,.0f}</strong></div><div class="fact-row"><span>Loss if defaulted</span><strong>${expected_loss:,.0f}</strong></div><div class="fact-row"><span>Break-even probability</span><strong>{expected_interest / (expected_interest + expected_loss):.1%}</strong></div></div>'
        )
    html(
        '<div class="research-note">The decision follows the portfolio cutoff. Expected profit uses this loan’s terms and a fixed loss assumption, so a declined loan can still have a positive expected return.</div>'
    )
    with st.expander("Review model inputs and training defaults"):
        st.caption(
            "Fields marked Training default were not entered in the application. Loan grade and subgrade remain at their training defaults."
        )
        entered = {
            "loan_amnt",
            "int_rate",
            "term",
            "purpose",
            "annual_inc",
            "dti",
            "fico_range_low",
            "emp_length_years",
            "home_ownership",
            "verification_status",
        }
        derived = {"installment", "fico_range_high", "installment_to_income"}
        rows = [
            {
                "Field": FRIENDLY_NAMES.get(name, name.replace("_", " ").capitalize()),
                "Value": format_input(name, value),
                "Source": "Application"
                if name in entered
                else "Calculated"
                if name in derived
                else "Training default",
            }
            for name, value in features.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
with tab_factors:
    html(
        '<p class="section-kicker">MODEL ATTRIBUTION</p><h2 class="section-title">Contributions to the estimate</h2><p class="section-copy">Feature contributions to the model’s default log-odds, before calibration. Blue lowers the raw score; rust raises it. These associations do not establish causes.</p>'
    )
    try:
        explanation = explainer(X_lgb)
        driver_col, attribution_col = st.columns([1, 1.6], gap="large")
        with driver_col:
            drivers(explanation, FRIENDLY_NAMES, format_input)
        with attribution_col:
            st.plotly_chart(
                shap_chart(explanation, FRIENDLY_NAMES),
                use_container_width=True,
                config={"displayModeBar": False, "scrollZoom": False},
            )
        st.caption(
            f"Raw model probability: {p_raw:.2%}. Calibrated probability: {p_cal:.2%}. Isotonic calibration maps ranges of raw scores to the same probability, so small input changes can leave the displayed result unchanged."
        )
    except (ValueError, IndexError, lgb.basic.LightGBMError):
        st.warning(
            "Feature contributions could not be calculated for this application. The probability and decision above are still available. Try another credit profile."
        )
with tab_method:
    html(f"""<p class="section-kicker">STUDY DESIGN</p><h2 class="section-title">Data, method and limitations</h2>
<div class="method-grid">
<div class="method-block"><h3>Data and model</h3><p>LightGBM trained on approximately 691,000 matured Lending Club loans issued from 2007–2015. A held-out 2015 fold fits the isotonic calibrator. The evaluation cohort contains matured 36-month loans issued in early 2016.</p></div>
<div class="method-block"><h3>The approval cutoff</h3><p>The {OPTIMAL_THRESHOLD:.2%} threshold maximizes realized profit on the 2016 evaluation cohort. Because the threshold was selected on those outcomes, the reported profit is a retrospective result, not an independent policy backtest.</p></div>
<div class="method-block"><h3>Return assumptions</h3><p>Interest assumes all scheduled installments are paid. Default loss is fixed at 35% of original principal. Expected profit is repayment-weighted interest minus default-weighted loss; it excludes funding costs, operating costs and prepayments.</p></div>
<div class="method-block"><h3>Limits of the sample</h3><p>The data contains approved loans only. Results may not transfer to declined applicants, other lending populations or later economic conditions. Inputs not shown in the form use saved training medians or modes.</p></div>
</div>""")
    st.caption(
        "Model development, calibration and the historical profit analysis are documented in notebooks/01–05. The underlying dataset is Lending Club’s 2007–2018 loan history."
    )
html(
    '<div class="footline"><span>Obligor Credit / Loan assessment</span><span>Lending Club · Historical model · USD</span></div>'
)
