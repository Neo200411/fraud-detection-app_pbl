import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        .main { background-color: #f8fafc; }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            border: 1px solid #e2e8f0;
            text-align: center;
        }
        .metric-big { font-size: 2rem; font-weight: 700; color: #0d9488; }
        .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
        .fraud-badge {
            background: #fee2e2; color: #991b1b;
            padding: 4px 12px; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600;
        }
        .safe-badge {
            background: #dcfce7; color: #166534;
            padding: 4px 12px; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600;
        }
        .section-header {
            background: #0d1b3e; color: white;
            padding: 0.6rem 1rem; border-radius: 8px;
            margin: 1rem 0 0.5rem; font-weight: 600;
        }
        .info-box {
            background: #f0fdf9; border-left: 4px solid #0d9488;
            padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
            margin: 0.5rem 0; font-size: 0.9rem;
        }
        .stProgress > div > div { background-color: #0d9488 !important; }
    </style>
    """, unsafe_allow_html=True)
