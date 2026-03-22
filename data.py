import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def generate_data(n_samples=5000, fraud_rate=0.017, seed=42):
    """Generate synthetic fraud data that mimics ULB dataset structure."""
    np.random.seed(seed)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions — tight PCA cluster
    legit = np.random.randn(n_legit, 28) * 0.8
    legit_amount = np.random.exponential(50, n_legit)
    legit_time   = np.random.uniform(0, 172800, n_legit)

    # Fraudulent transactions — shifted distribution
    fraud = np.random.randn(n_fraud, 28) * 1.4 + np.random.choice([-2, 2], size=(n_fraud, 28))
    fraud_amount = np.random.exponential(200, n_fraud)
    fraud_time   = np.random.uniform(0, 172800, n_fraud)

    V_cols = [f"V{i}" for i in range(1, 29)]
    df_legit = pd.DataFrame(legit, columns=V_cols)
    df_legit["Amount"] = legit_amount
    df_legit["Time"]   = legit_time
    df_legit["Class"]  = 0

    df_fraud = pd.DataFrame(fraud, columns=V_cols)
    df_fraud["Amount"] = fraud_amount
    df_fraud["Time"]   = fraud_time
    df_fraud["Class"]  = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True).sample(frac=1, random_state=seed)
    return df

def engineer_features(df):
    df = df.copy()
    df["Amount_log"]          = np.log1p(df["Amount"])
    df["hour_of_day"]         = ((df["Time"] % 86400) // 3600).astype(int)
    df["txn_count_prev_1000"] = (pd.Series(range(len(df)))
                                   .rolling(window=1000, min_periods=1)
                                   .count().shift(1).fillna(0).values)
    feat_cols = [c for c in df.columns if c.startswith("V")] + \
                ["Amount_log", "hour_of_day", "txn_count_prev_1000"]
    return df, feat_cols
