# Hybrid AI Fraud Detection — Component Architecture

This document explains the modular structure of the E-Commerce Fraud Detection application. The originally monolithic application has been refactored into distinct components to improve maintainability, readability, and separation of concerns.

## 📁 1. `streamlit_app.py` (Main Entry Point)
This is the primary script that Streamlit Community Cloud (and local Streamlit servers) executes to run the application. 
- **Role:** Orchestrates the entire application flow.
- **Functionality:** It sets the global page configuration (`st.set_page_config`), applies the custom CSS, renders the sidebar, triggers the training pipeline, and maps the output metrics to the respective UI tabs.

## 📁 2. `config.py` (Configuration & Styling)
- **Role:** Design system and global settings.
- **Functionality:** Contains the `apply_custom_css()` function which injects custom HTML/CSS into the Streamlit app. This includes styling for metric cards, info boxes, section headers, badges (Fraud Alert vs. Legitimate), and Streamlit component overrides (like adjusting padding and tab styling).

## 📁 3. `data.py` (Data Engineering)
- **Role:** Handles data simulation and feature engineering.
- **Functionality:** 
  - `generate_data()`: Simulates a highly imbalanced credit card fraud dataset based on the requested number of samples and fraud rate.
  - `engineer_features()`: Applies feature transformations (e.g., `np.log1p` on transaction amounts) and extracts relevant columns for machine learning.

## 📁 4. `models.py` (Machine Learning & Hybrid Fusion)
- **Role:** The core computational engine of the application.
- **Functionality:**
  - **Autoencoder (`nn.Module`):** Defines a PyTorch neural network to learn the localized manifold of legitimate transactions (unsupervised learning).
  - **`train_pipeline()`:** 
    - Scales the dataset (`StandardScaler`).
    - Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the data for supervised training.
    - Trains the supervised **XGBoost** model.
    - Trains the unsupervised **Isolation Forest** to detect statistical anomalies.
    - Trains the **Autoencoder** and calculates reconstruction errors.
    - Calculates the **Hybrid Fusion** mathematics (`s_comb = α * IF_score + (1 - α) * AE_error`, followed by `final = β * XGB_prob + (1 - β) * s_comb`).
    - Exposes trained models (`xgb_model`, `if_model`, `ae_model`) and their respective `MinMaxScaler` objects for live predictions down the line.

## 📁 5. `ui_sidebar.py` (User Input Controls)
- **Role:** Manages the parameters adjustable by the user.
- **Functionality:** Uses `render_sidebar()` to draw sliders inside the Streamlit sidebar (`st.sidebar`). It collects dataset parameters (N samples, fraud rate), hybrid weights (α, β), model hyperparameters (Autoencoder epochs, XGB/IF trees), and alerts rate (Top %). It returns these selections as a configuration dictionary.

## 📁 6. `ui_tabs.py` (Dashboard Visualizations)
- **Role:** Renders all metrics, charts, and interactive elements.
- **Functionality:** Broken down into specific functions for each tab of the app:
  - `render_overview()`: Shows ROC-AUC, F1, Alert density histograms, and Confusion Matrix.
  - `render_model_comparison()`: Generates a table and bar charts comparing strictly XGBoost vs Isolation Forest vs Autoencoder vs the Hybrid approach.
  - `render_curves()`: Plots ROC and Precision-Recall trajectories for all models.
  - `render_shap()`: Visualizes XGBoost feature importances (Gain) and Permutation Importance tests.
  - `render_live_prediction()`: Uses live slider inputs to build a transaction array, scales it using the saved `MinMaxScaler`s from `models.py`, and calculates real-time individual and fused fraud scores.
  - `render_ablation()`: Highlights the consequence of removing various models from the hybrid pipeline, including a financial Cost Metric analysis (`C_fp` vs `C_fn`).

## 📁 7. `requirements.txt` (Environment Dependencies)
- **Role:** Package manager file for deployment.
- **Functionality:** Instructs Streamlit Cloud to install necessary PyPI dependencies (e.g., `scikit-learn`, `xgboost`, `torch`, `imbalanced-learn`) before booting up `streamlit_app.py`.
