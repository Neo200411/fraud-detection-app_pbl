import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st

st.set_page_config(
    page_title="Fraud Detection — Hybrid AI Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np

from config import apply_custom_css
from ui_sidebar import render_sidebar
from models import train_pipeline
from ui_tabs import render_overview, render_model_comparison, render_curves, render_shap, render_live_prediction, render_ablation

# Set global CSS
apply_custom_css()

# Header
st.markdown("""
<h1 style='color:#0d1b3e;margin-bottom:0'>🛡️ E-Commerce Fraud Detection</h1>
<p style='color:#64748b;font-size:1rem;margin-top:4px'>
A Novel Taxonomical Framework — Hybrid AI Demo &nbsp;|&nbsp; 
Manipal University Jaipur &nbsp;|&nbsp; DSE3270 PBL-4
</p>
""", unsafe_allow_html=True)
st.divider()

# Sidebar
sidebar_cfg = render_sidebar()

if "results" not in st.session_state or sidebar_cfg["train_btn"]:
    with st.spinner("Training pipeline — XGBoost + Isolation Forest + Autoencoder..."):
        st.session_state.results = train_pipeline(
            sidebar_cfg["n_samples"], sidebar_cfg["fraud_rate"], 
            sidebar_cfg["latent_dim"], sidebar_cfg["ae_epochs"],
            sidebar_cfg["if_n_estimators"], sidebar_cfg["xgb_n_estimators"], 
            sidebar_cfg["alpha"], sidebar_cfg["beta"]
        )

R = st.session_state.results
y_test      = R["y_test"]
final_score = R["final_score"]
p_xgb       = R["p_xgb"]
s_if_n      = R["s_if_n"]
rec_err_n   = R["rec_err_n"]

T = np.percentile(final_score, 100 * (1 - sidebar_cfg["top_pct"] / 100))
y_pred = (final_score >= T).astype(int)

# Tabs
tabs = st.tabs([
    "📊 Overview",
    "🔬 Model Comparison",
    "📈 ROC & PR Curves",
    "🔍 SHAP Explainability",
    "🎯 Live Prediction",
    "🧪 Ablation Study"
])

with tabs[0]:
    render_overview(y_test, final_score, p_xgb, y_pred, sidebar_cfg["top_pct"], T, 
                    sidebar_cfg["alpha"], sidebar_cfg["beta"], 
                    sidebar_cfg["n_samples"], sidebar_cfg["fraud_rate"])

with tabs[1]:
    render_model_comparison(y_test, final_score, p_xgb, s_if_n, rec_err_n, sidebar_cfg["top_pct"])

with tabs[2]:
    render_curves(y_test, final_score, p_xgb, s_if_n, rec_err_n, sidebar_cfg["top_pct"], T)

with tabs[3]:
    render_shap(R, y_test, p_xgb)

with tabs[4]:
    render_live_prediction(R, T, sidebar_cfg["alpha"], sidebar_cfg["beta"], s_if_n, rec_err_n)

with tabs[5]:
    render_ablation(y_test, p_xgb, s_if_n, rec_err_n, final_score, sidebar_cfg["top_pct"])

# Footer
st.divider()
st.markdown("""
<div style='text-align:center;color:#94a3b8;font-size:0.8rem;padding:0.5rem'>
DSE3270 PBL-4 &nbsp;|&nbsp; Manipal University Jaipur &nbsp;|&nbsp;
Anshul Ojha · Sujal Kumar · Neo Mishra &nbsp;|&nbsp; Guide: Mr. Deevesh Chaudhary<br>
Dataset: ULB Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
</div>
""", unsafe_allow_html=True)
