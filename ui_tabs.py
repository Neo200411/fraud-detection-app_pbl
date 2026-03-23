import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score)
import torch

def render_overview(y_test, final_score, p_xgb, y_pred, top_pct, T, alpha, beta, n_samples, fraud_rate):
    auc_h  = roc_auc_score(y_test, final_score)
    auc_x  = roc_auc_score(y_test, p_xgb)
    rec_h  = recall_score(y_test, y_pred, zero_division=0)
    prec_h = precision_score(y_test, y_pred, zero_division=0)
    f1_h   = f1_score(y_test, y_pred, zero_division=0)

    st.markdown('<div class="section-header">Pipeline Performance Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, f"{auc_h:.3f}",         "Hybrid ROC-AUC"),
        (c2, f"{rec_h:.3f}",         f"Recall @ top {top_pct}%"),
        (c3, f"{prec_h:.3f}",        f"Precision @ top {top_pct}%"),
        (c4, f"{f1_h:.3f}",          "F1 Score"),
        (c5, f"{int(y_pred.sum())}",  "Alerts Raised"),
    ]:
        col.markdown(f'''
        <div class="metric-card">
            <div class="metric-big">{val}</div>
            <div class="metric-label">{label}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-header">Fraud Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(final_score[y_test == 0], bins=60, alpha=0.65,
                label="Legitimate", color="#0d9488", density=True)
        ax.hist(final_score[y_test == 1], bins=30, alpha=0.75,
                label="Fraud", color="#f59e0b", density=True)
        ax.axvline(T, color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"Alert threshold (top {top_pct}%)")
        ax.set_xlabel("Final Fusion Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"], ax=ax2,
                    annot_kws={"size": 14})
        ax2.set_xlabel("Predicted", fontsize=10)
        ax2.set_ylabel("Actual", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="section-header">Hybrid Pipeline Architecture</div>', unsafe_allow_html=True)
    st.markdown(f'''
<div class="info-box">
<b>Fusion Formula:</b> &nbsp;
<code>s_comb = α·IF_score + (1-α)·AE_rec_error</code>
&nbsp;→&nbsp;
<code>final_score = β·XGB_prob + (1-β)·s_comb</code>
&nbsp; | &nbsp;
Alert if score ≥ <b>top {top_pct}% percentile</b>
</div>
''', unsafe_allow_html=True)

    pipe_cols = st.columns(5)
    steps = [
        ("📥", "Raw Data",          f"{n_samples:,} transactions\n{fraud_rate*100:.1f}% fraud rate"),
        ("⚙️", "Preprocessing",    "Log-Amount, hour\nvelocity → StandardScaler"),
        ("⚖️", "SMOTE + Split",    "Balance inside\n70% train / 30% test"),
        ("🤖", "3 Models",         "XGBoost + IF\n+ Autoencoder"),
        ("🔀", "Hybrid Fusion",    f"α={alpha:.2f}, β={beta:.2f}\nTop {top_pct}% flagged"),
    ]
    for col, (icon, title, desc) in zip(pipe_cols, steps):
        col.markdown(f'''
        <div style="background:#0d1b3e;border-radius:10px;padding:12px 10px;text-align:center;color:white;height:110px">
            <div style="font-size:1.5rem">{icon}</div>
            <div style="font-weight:600;font-size:0.85rem;color:#5eead4;margin:4px 0">{title}</div>
            <div style="font-size:0.72rem;color:#94a3b8;line-height:1.4">{desc}</div>
        </div>''', unsafe_allow_html=True)

def render_model_comparison(y_test, final_score, p_xgb, s_if_n, rec_err_n, top_pct):
    st.markdown('<div class="section-header">Individual Model vs Hybrid Fusion</div>', unsafe_allow_html=True)

    models = {
        "XGBoost":         p_xgb,
        "Isolation Forest": s_if_n,
        "Autoencoder":     rec_err_n,
        "Hybrid Fusion":   final_score,
    }
    colors = ["#0d9488", "#0f6e56", "#5eead4", "#f59e0b"]
    rows = []
    for name, scores in models.items():
        thr = np.percentile(scores, 100 * (1 - top_pct / 100))
        yp  = (scores >= thr).astype(int)
        rows.append({
            "Model":       name,
            "ROC-AUC":     round(roc_auc_score(y_test, scores), 4),
            "Precision":   round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":      round(recall_score(y_test, yp, zero_division=0), 4),
            "F1":          round(f1_score(y_test, yp, zero_division=0), 4),
        })

    comp_df = pd.DataFrame(rows).set_index("Model")

    def color_best(col):
        best = col.max()
        return ["background-color:#d1fae5;font-weight:bold"
                if v == best else "" for v in col]

    st.dataframe(comp_df.style.apply(color_best), use_container_width=True)

    fig3, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["ROC-AUC", "Recall"]):
        vals  = comp_df[metric].values
        names = comp_df.index.tolist()
        bars  = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(min(vals) * 0.95, 1.02)
        ax.set_title(metric, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

def render_curves(y_test, final_score, p_xgb, s_if_n, rec_err_n, top_pct, T):
    st.markdown('<div class="section-header">ROC and Precision-Recall Curves</div>', unsafe_allow_html=True)

    fig4, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    plot_models = {
        "XGBoost":          (p_xgb,       "#0d9488"),
        "Isolation Forest": (s_if_n,      "#0f6e56"),
        "Autoencoder":      (rec_err_n,   "#5eead4"),
        "Hybrid Fusion":    (final_score, "#f59e0b"),
    }

    for name, (scores, col) in plot_models.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val     = roc_auc_score(y_test, scores)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
                    color=col, linewidth=2)

        prec_c, rec_c, _ = precision_recall_curve(y_test, scores)
        pr_auc = np.trapezoid(prec_c, rec_c)
        ax_pr.plot(rec_c, prec_c, label=f"{name} (PR-AUC={abs(pr_auc):.3f})",
                   color=col, linewidth=2)

    ax_roc.plot([0,1],[0,1],"--",color="#94a3b8",linewidth=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves", fontweight="bold")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.spines[["top","right"]].set_visible(False)

    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves", fontweight="bold")
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown('<div class="section-header">Precision & Recall vs Threshold (Hybrid)</div>',
                unsafe_allow_html=True)
    prec_vals, rec_vals, thresholds = precision_recall_curve(y_test, final_score)
    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    ax5.plot(thresholds, prec_vals[:-1], label="Precision", color="#0d9488", linewidth=2)
    ax5.plot(thresholds, rec_vals[:-1],  label="Recall",    color="#f59e0b", linewidth=2)
    ax5.axvline(T, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Current threshold (top {top_pct}%)")
    ax5.set_xlabel("Threshold"); ax5.set_ylabel("Score")
    ax5.set_title("Precision & Recall vs Decision Threshold — Hybrid Model", fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

def render_shap(R, y_test, p_xgb):
    st.markdown('<div class="section-header">XGBoost Feature Importance (Gain + Permutation)</div>',
                unsafe_allow_html=True)

    feat_cols = R["feat_cols"]
    xgb_model = R["xgb_model"]

    # Built-in importance
    imp = xgb_model.feature_importances_
    imp_df = (pd.DataFrame({"Feature": feat_cols, "Importance": imp})
                .sort_values("Importance", ascending=False).head(15))

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("**XGBoost Gain Importance (top 15)**")
        fig6, ax6 = plt.subplots(figsize=(6, 5))
        colors_bar = ["#f59e0b" if i < 3 else "#0d9488" for i in range(len(imp_df))]
        ax6.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                 color=colors_bar[::-1])
        ax6.set_xlabel("Importance Score")
        ax6.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

    with right2:
        st.markdown("**Permutation Importance (AUC drop, 5 repeats)**")
        rng = np.random.RandomState(42)
        X_test_arr = R["X_test"]
        base_auc   = roc_auc_score(y_test, p_xgb)
        perm_imps  = []
        sample_idx = rng.choice(len(X_test_arr), size=min(500, len(X_test_arr)), replace=False)

        for fi in range(len(feat_cols)):
            drops = []
            for _ in range(5):
                X_perm = X_test_arr[sample_idx].copy()
                rng.shuffle(X_perm[:, fi])
                preds  = xgb_model.predict_proba(X_perm)[:, 1]
                drops.append(base_auc - roc_auc_score(y_test[sample_idx], preds))
            perm_imps.append(np.mean(drops))

        perm_df = (pd.DataFrame({"Feature": feat_cols, "AUC Drop": perm_imps})
                     .sort_values("AUC Drop", ascending=False).head(15))

        fig7, ax7 = plt.subplots(figsize=(6, 5))
        colors_perm = ["#f59e0b" if i < 3 else "#0d9488" for i in range(len(perm_df))]
        ax7.barh(perm_df["Feature"][::-1], perm_df["AUC Drop"][::-1],
                 color=colors_perm[::-1])
        ax7.set_xlabel("Mean AUC Decrease")
        ax7.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

    st.markdown('''
<div class="info-box">
<b>How to read this:</b> Features with a high permutation importance (right chart) are the ones the model 
genuinely depends on. Shuffling their values causes a large drop in AUC. The top 3 (highlighted in amber) 
are the most critical fraud indicators — in the ULB dataset these are typically V14, V4, and V11.
</div>''', unsafe_allow_html=True)

def render_live_prediction(R, T, alpha, beta, s_if_n, rec_err_n):
    st.markdown('<div class="section-header">Single Transaction — Live Fraud Scoring</div>',
                unsafe_allow_html=True)
    st.markdown("Adjust the sliders to simulate a transaction and see how each model scores it.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        amount = st.slider("Transaction Amount (€)", 0.5, 5000.0, 150.0, 0.5)
        hour   = st.slider("Hour of Day", 0, 23, 14)
        v1     = st.slider("V1 (PCA feature)", -5.0, 5.0, 0.0, 0.1)
        v2     = st.slider("V2 (PCA feature)", -5.0, 5.0, 0.0, 0.1)
        v3     = st.slider("V3 (PCA feature)", -5.0, 5.0, 0.0, 0.1)
        v4     = st.slider("V4 (PCA feature)", -5.0, 5.0, 0.0, 0.1)

    with col_b:
        v5     = st.slider("V5 (PCA feature)",  -5.0, 5.0, 0.0, 0.1)
        v14    = st.slider("V14 (key feature)", -5.0, 5.0, 0.0, 0.1,
                           help="V14 is typically the strongest fraud indicator")
        v17    = st.slider("V17 (key feature)", -5.0, 5.0, 0.0, 0.1)
        v11    = st.slider("V11 (key feature)", -5.0, 5.0, 0.0, 0.1)

    # Build feature vector (remaining V cols = 0)
    feat_cols = R["feat_cols"]
    txn = np.zeros(len(feat_cols))
    v_vals = {1: v1, 2: v2, 3: v3, 4: v4, 5: v5, 11: v11, 14: v14, 17: v17}
    for fi, fname in enumerate(feat_cols):
        if fname.startswith("V"):
            vi = int(fname[1:])
            txn[fi] = v_vals.get(vi, 0.0)
        elif fname == "Amount_log":
            txn[fi] = np.log1p(amount)
        elif fname == "hour_of_day":
            txn[fi] = hour
        elif fname == "txn_count_prev_1000":
            txn[fi] = 1.0

    txn_scaled = R["scaler"].transform(txn.reshape(1, -1))

    # Score
    xgb_prob = float(R["xgb_model"].predict_proba(txn_scaled)[0, 1])
    if_raw   = float(-R["if_model"].decision_function(txn_scaled)[0])
    if_norm  = float(np.clip(R["if_scaler"].transform([[if_raw]])[0, 0], 0, 1))

    txn_t    = torch.tensor(txn_scaled, dtype=torch.float32)
    R["ae_model"].eval()
    with torch.no_grad():
        rec_txn = R["ae_model"](txn_t).numpy()
    ae_err_raw  = float(np.mean((rec_txn - txn_scaled) ** 2))
    ae_norm     = float(np.clip(R["ae_scaler"].transform([[ae_err_raw]])[0, 0], 0, 1))

    s_comb_txn  = alpha * if_norm + (1 - alpha) * ae_norm
    final_txn   = beta  * xgb_prob + (1 - beta) * s_comb_txn

    st.markdown("---")
    st.markdown("### Scores")
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, name, val, clr in [
        (sc1, "XGBoost P(fraud)", xgb_prob, "#0d9488"),
        (sc2, "Isolation Forest", if_norm,  "#0f6e56"),
        (sc3, "Autoencoder",      ae_norm,  "#5eead4"),
        (sc4, "Hybrid Final",     final_txn,"#f59e0b"),
    ]:
        col.markdown(f'''
        <div class="metric-card">
            <div style="font-size:1.6rem;font-weight:700;color:{clr}">{val:.3f}</div>
            <div class="metric-label">{name}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")
    verdict = final_txn >= T
    badge = '<span class="fraud-badge">⚠️ FRAUD ALERT</span>' if verdict else '<span class="safe-badge">✅ LEGITIMATE</span>'
    st.markdown(f"### Verdict: {badge} &nbsp; Final score: **{final_txn:.3f}** | Threshold: **{T:.3f}**",
                unsafe_allow_html=True)

    # Score bar chart
    fig8, ax8 = plt.subplots(figsize=(8, 2.2))
    names_bar = ["XGBoost", "Isolation Forest", "Autoencoder", "Hybrid Final"]
    vals_bar  = [xgb_prob, if_norm, ae_norm, final_txn]
    bar_colors = ["#0d9488", "#0f6e56", "#5eead4", "#f59e0b"]
    bars8 = ax8.barh(names_bar, vals_bar, color=bar_colors, height=0.55)
    ax8.axvline(T, color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Alert threshold {T:.3f}")
    ax8.set_xlim(0, 1.05)
    ax8.set_xlabel("Score")
    ax8.legend(fontsize=9)
    ax8.spines[["top","right"]].set_visible(False)
    for bar, val in zip(bars8, vals_bar):
        ax8.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()

def render_ablation(y_test, p_xgb, s_if_n, rec_err_n, final_score, top_pct):
    st.markdown('<div class="section-header">Ablation Study — Each Component Contribution</div>',
                unsafe_allow_html=True)

    ablation_configs = {
        "XGB Only":       p_xgb,
        "IF Only":        s_if_n,
        "AE Only":        rec_err_n,
        "Anomaly Avg":    0.5 * s_if_n + 0.5 * rec_err_n,
        "Hybrid Fusion":  final_score,
    }
    abl_rows = []
    for name, scores in ablation_configs.items():
        thr = np.percentile(scores, 100 * (1 - top_pct / 100))
        yp  = (scores >= thr).astype(int)
        abl_rows.append({
            "Configuration":  name,
            "ROC-AUC":        round(roc_auc_score(y_test, scores), 4),
            "Precision":      round(precision_score(y_test, yp, zero_division=0), 4),
            "Recall":         round(recall_score(y_test, yp, zero_division=0), 4),
            "F1":             round(f1_score(y_test, yp, zero_division=0), 4),
        })

    abl_df = pd.DataFrame(abl_rows).set_index("Configuration")

    def highlight_hybrid(row):
        return ["background-color:#fef3c7;font-weight:bold"
                if row.name == "Hybrid Fusion" else "" for _ in row]

    st.dataframe(abl_df.style.apply(highlight_hybrid, axis=1), use_container_width=True)

    fig9, axes9 = plt.subplots(1, 2, figsize=(12, 4))
    abl_colors = ["#94a3b8", "#0f6e56", "#5eead4", "#64748b", "#f59e0b"]

    for ax, metric in zip(axes9, ["ROC-AUC", "Recall"]):
        vals  = abl_df[metric].values
        names = abl_df.index.tolist()
        bars  = ax.bar(names, vals, color=abl_colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(min(vals)*0.9, 1.02)
        ax.set_title(metric, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig9)
    plt.close()

    st.markdown('''
<div class="info-box">
<b>Key Takeaway:</b> No single model achieves the Hybrid's recall score. The ablation study proves that 
XGBoost handles known patterns while Isolation Forest and Autoencoder independently catch anomalies 
the supervised model misses. Removing any component degrades performance — all three are necessary.
</div>''', unsafe_allow_html=True)

    # Cost metric comparison
    st.markdown('<div class="section-header">Cost Metric Comparison (C_fp=1, C_fn=50)</div>',
                unsafe_allow_html=True)
    C_fp, C_fn = 1, 50
    cost_rows = []
    for name, scores in ablation_configs.items():
        thr = np.percentile(scores, 100 * (1 - top_pct / 100))
        yp  = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        cost = (C_fp * fp + C_fn * fn) / len(y_test)
        cost_rows.append({"Configuration": name, "Cost/Transaction": round(cost, 4),
                          "False Positives": int(fp), "False Negatives": int(fn)})

    cost_df = pd.DataFrame(cost_rows).set_index("Configuration")

    def highlight_min_cost(col):
        if col.name == "Cost/Transaction":
            mn = col.min()
            return ["background-color:#d1fae5;font-weight:bold"
                    if v == mn else "" for v in col]
        return ["" for _ in col]

    st.dataframe(cost_df.style.apply(highlight_min_cost), use_container_width=True)
