import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.image("https://img.shields.io/badge/Manipal%20University-Jaipur-0d9488?style=for-the-badge", use_container_width=True)
        st.markdown("## ⚙️ Pipeline Settings")

        st.markdown("**Dataset**")
        n_samples  = st.slider("Total transactions", 2000, 10000, 5000, 500)
        fraud_rate = st.slider("Fraud rate (%)", 0.5, 3.0, 1.7, 0.1) / 100

        st.markdown("**Model Parameters**")
        latent_dim       = st.selectbox("AE latent dimension", [8, 16, 32], index=1)
        ae_epochs        = st.slider("AE training epochs", 5, 40, 15, 5)
        if_n_estimators  = st.selectbox("IF n_estimators", [128, 256, 512], index=1)
        xgb_n_estimators = st.slider("XGB n_estimators", 50, 300, 100, 50)

        st.markdown("**Fusion Weights**")
        alpha = st.slider("α — IF vs AE weight", 0.0, 1.0, 0.4, 0.05,
                          help="α·IF + (1-α)·AE")
        beta  = st.slider("β — XGB vs Anomaly weight", 0.0, 1.0, 0.6, 0.05,
                          help="β·XGB + (1-β)·s_comb")

        st.markdown("**Alert Threshold**")
        top_pct = st.slider("Flag top X% as fraud", 1, 10, 3, 1)

        train_btn = st.button("🚀 Train Pipeline", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown("""
**How it works**
1. Data generated / loaded
2. Feature engineering (31 features)
3. SMOTE inside training only
4. XGBoost + Isolation Forest + Autoencoder trained
5. Scores fused with α, β weights
6. Top X% flagged as fraud
        """)
        
    return {
        "n_samples": n_samples,
        "fraud_rate": fraud_rate,
        "latent_dim": latent_dim,
        "ae_epochs": ae_epochs,
        "if_n_estimators": if_n_estimators,
        "xgb_n_estimators": xgb_n_estimators,
        "alpha": alpha,
        "beta": beta,
        "top_pct": top_pct,
        "train_btn": train_btn
    }
