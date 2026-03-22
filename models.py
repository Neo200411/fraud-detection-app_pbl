import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import torch
import torch.nn as nn

from data import generate_data, engineer_features

# ─────────────────────────────────────────────
# AUTOENCODER DEFINITION
# ─────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ─────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def train_pipeline(n_samples, fraud_rate, latent_dim, ae_epochs,
                    if_n_estimators, xgb_n_estimators, alpha, beta):
    df = generate_data(n_samples, fraud_rate)
    df, feat_cols = engineer_features(df)

    X = df[feat_cols].values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    # ── SMOTE ──
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # ── XGBoost ──
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_n_estimators, max_depth=5,
        learning_rate=0.05, use_label_encoder=False,
        eval_metric="auc", tree_method="hist",
        scale_pos_weight=int((y_res == 0).sum() / max((y_res == 1).sum(), 1)),
        verbosity=0, random_state=42, n_jobs=1
    )
    xgb_model.fit(X_res, y_res)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # ── Isolation Forest ──
    if_model = IsolationForest(
        n_estimators=if_n_estimators, contamination=float(fraud_rate),
        random_state=42, n_jobs=1
    )
    if_model.fit(X_train[y_train == 0])
    s_if = -if_model.decision_function(X_test)
    s_if_n = MinMaxScaler().fit_transform(s_if.reshape(-1, 1)).ravel()

    # ── Autoencoder ──
    device = torch.device("cpu")
    X_ae   = torch.tensor(X_train[y_train == 0], dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    ae = Autoencoder(X_train.shape[1], latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader  = torch.utils.data.DataLoader(X_ae, batch_size=256, shuffle=True)

    ae.train()
    for _ in range(ae_epochs):
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(ae(batch), batch)
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        rec = ae(X_test_t).numpy()
    rec_err   = np.mean((rec - X_test) ** 2, axis=1)
    rec_err_n = MinMaxScaler().fit_transform(rec_err.reshape(-1, 1)).ravel()

    # ── Hybrid Fusion ──
    s_comb      = alpha * s_if_n + (1 - alpha) * rec_err_n
    final_score = beta  * p_xgb  + (1 - beta)  * s_comb

    return {
        "X_test": X_test, "y_test": y_test,
        "p_xgb": p_xgb, "s_if_n": s_if_n,
        "rec_err_n": rec_err_n, "s_comb": s_comb,
        "final_score": final_score,
        "xgb_model": xgb_model, "if_model": if_model, "ae_model": ae,
        "scaler": scaler, "feat_cols": feat_cols,
        "alpha": alpha, "beta": beta,
        "df": df,
    }
