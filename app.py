import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

# --- 1. System Configuration & Security ---
st.set_page_config(page_title="Intelligence Suite", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>

    /* GLOBAL APP BACKGROUND */
    .main {
        background: linear-gradient(135deg, #0c1117, #1a2330);
        color: #e2e8f0;
    }

    /* TITLES */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #00eaff !important;
        letter-spacing: 0.5px;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: #11151c !important;
        border-right: 1px solid #2a3342;
    }
    .sidebar-content {
        color: #cbd5e1;
    }

    /* METRICS CARDS */
    div[data-testid="stMetric"] {
        background: rgba(0,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        border: 2px solid rgba(0,255,255,0.15);
        transition: transform .2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: scale(1.05);
        border-color: #00eaff;
    }

    /* TABS */
    .stTabs [data-baseweb="tab"] {
        background-color: #141a24 !important;
        padding: 12px 20px;
        border-radius: 12px;
        color: #94a3b8;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1f2734 !important;
        color: #00eaff;
    }
    .stTabs [aria-selected="true"] {
        color: #00eaff !important;
        border-bottom: 3px solid #00eaff !important;
        background-color: transparent !important;
    }

    /* FILE UPLOAD */
    .stFileUploader {
        border: 2px dashed #2dd4bf !important;
        background: rgba(0,255,255,0.06);
        border-radius: 12px;
        padding: 10px;
    }

    /* DOWNLOAD BUTTON */
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: bold;
        background-color: #0ea5e9;
        border: none;
        padding: 10px 18px;
        transition: 0.25s;
    }
    .stDownloadButton button:hover {
        background-color: #38bdf8;
        transform: translateY(-2px);
    }

    /* SPINNER */
    .stSpinner > div {
        border-top-color: #00eaff !important;
    }

</style>
""", unsafe_allow_html=True)

def check_password():
    """Handles professional access control."""
    def password_entered():
        if st.session_state["password"] == "admin123":  # Change this to your preferred key
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ Enterprise Data Portal")
        st.text_input("Access Key", type="password", on_change=password_entered, key="password")
        st.info("Please enter the system access key to initialize the dashboard.")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🛡️ Enterprise Data Portal")
        st.text_input("Access Key", type="password", on_change=password_entered, key="password")
        st.error("🔒 Access Denied. Invalid Key.")
        return False
    else:
        return True

# --- 2. Application Logic ---
if check_password():
    
    # Custom CSS for a Premium "Software" Feel
    st.markdown("""
        <style>
        .main { background-color: #0b0d11; }
        div[data-testid="stMetricValue"] { font-size: 32px; color: #00d4ff; font-weight: 700; }
        .stMetric { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
        .stTabs [data-baseweb="tab-list"] { gap: 30px; }
        .stTabs [data-baseweb="tab"] { font-size: 16px; color: #94a3b8; }
        </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_engine():
        """Loads pre-trained model and scaler assets."""
        m_path, s_path = 'models/kmeans_model.pkl', 'models/scaler.pkl'
        if os.path.exists(m_path) and os.path.exists(s_path):
            return pickle.load(open(m_path, 'rb')), pickle.load(open(s_path, 'rb'))
        return None, None

    model, scaler = load_engine()

    # --- Sidebar: Control Panel ---
    with st.sidebar:
        st.title("⚙️ System Admin")
        st.markdown("---")
        data_file = st.file_uploader("Import Transaction Logs", type=['xlsx', 'csv'])
        
        st.markdown("---")
        if st.button("🔴 Terminate Session"):
            st.session_state["password_correct"] = False
            st.rerun()
        
        st.caption("Intelligence Engine v2.4.0")

    # --- Header ---
    st.title("💎 Revenue & Customer Intelligence Suite")
    
    if model is None:
        st.error("System Error: AI Model and Scaler not detected in /models/ directory.")
    elif data_file:
        with st.spinner("Analyzing Market Patterns..."):
            # Data Processing Pipeline
            if data_file.name.endswith('csv'):
                df_raw = pd.read_csv(data_file)
            else:
                df_raw = pd.read_excel(data_file, sheet_name='Year 2010-2011')
                
            df = df_raw.dropna(subset=['Customer ID'])
            df['TotalPrice'] = df['Quantity'] * df['Price']
            
            # RFM Engine
            latest = df['InvoiceDate'].max() + pd.Timedelta(days=1)
            rfm = df.groupby('Customer ID').agg({
                'InvoiceDate': lambda x: (latest - x.max()).days,
                'Invoice': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})
            rfm = rfm[rfm['Monetary'] > 0]

            # Machine Learning Inference
            rfm_scaled = scaler.transform(np.log1p(rfm))
            rfm['Cluster'] = model.predict(rfm_scaled)

            # --- Row 1: Executive KPI Metrics ---
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Clients", f"{len(rfm):,}")
            k2.metric("Gross Revenue", f"${rfm['Monetary'].sum():,.0f}")
            k3.metric("Avg Spend (LTV)", f"${rfm['Monetary'].mean():,.2f}")
            k4.metric("Retention Index", f"{int(rfm['Recency'].mean())}d")

            st.markdown("---")

            # --- Row 2: Analytics Tabs ---
            tab_vis, tab_strat = st.tabs(["📊 Visual Analytics", "🎯 Deployment Strategy"])

            with tab_vis:
                v_col1, v_col2 = st.columns([1, 1.5])
                with v_col1:
                    st.plotly_chart(px.pie(rfm, names='Cluster', hole=0.6, 
                                           title="Market Segment Distribution", 
                                           color_discrete_sequence=px.colors.sequential.Blues_r), use_container_width=True)
                with v_col2:
                    fig_3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', 
                                           color='Cluster', height=600, template="plotly_dark",
                                           color_continuous_scale=px.colors.sequential.Blues,
                                           title="3D Behavioral Projection")
                    fig_3d.update_traces(marker=dict(size=2))
                    st.plotly_chart(fig_3d, use_container_width=True)

            with tab_strat:
                st.subheader("Tactical Deployment Roadmap")
                
                # Professional Strategy Logic
                overall_means = rfm.mean()
                for i in range(model.n_clusters):
                    c_slice = rfm[rfm['Cluster'] == i]
                    avg_r, avg_m = c_slice['Recency'].mean(), c_slice['Monetary'].mean()
                    
                    if avg_r < overall_means['Recency'] and avg_m > overall_means['Monetary']:
                        tier, strategy = "Tier 1: High-Performance", "Primary revenue drivers. Apply exclusive retention perks."
                    elif avg_r > overall_means['Recency'] * 1.4:
                        tier, strategy = "Tier 4: At-Risk", "Significant churn risk. Trigger re-engagement campaign."
                    else:
                        tier, strategy = "Tier 2-3: Stable", "Steady activity. Target for cross-selling and upselling."

                    with st.expander(f"SEGMENT {i} | {tier}"):
                        st.markdown(f"**Recommended Action:** {strategy}")
                        st.write(f"**Group Size:** {len(c_slice)} Customers")
                        st.write(f"**Average Value:** ${avg_m:,.2f}")
                        st.download_button(f"Download Group {i} Registry", c_slice.to_csv().encode('utf-8'), f"segment_{i}.csv")
    else:
        st.info("Portal Active. Please upload transactional logs in the sidebar to generate analysis.")