import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import sqlite3
from datetime import datetime

# --- System Settings ---
st.set_page_config(page_title="Intelligence Suite", layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0c1117, #1a2330);
    color: #e2e8f0;
}
h1, h2, h3 {
    font-family: 'Segoe UI';
    color: #00eaff !important;
}
section[data-testid="stSidebar"] {
    background-color: #11151c !important;
    border-right: 1px solid #2a3342;
}
div[data-testid="stMetric"] {
    background: rgba(0,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    border: 2px solid rgba(0,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)

# ================= PASSWORD AUTH =================
def check_password():
    def password_entered():
        if st.session_state["password"] == "admin123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🛡️ Enterprise Data Portal")
        st.text_input("Access Key", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🛡️ Enterprise Data Portal")
        st.text_input("Access Key", type="password", on_change=password_entered, key="password")
        st.error("Wrong Password ❌")
        return False
    return True


# ================= DATABASE CONFIG =================
DB_PATH = "customer_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            data BLOB
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_upload(filename, df):
    df.to_pickle("temp.pkl")
    with open("temp.pkl", "rb") as f:
        blob = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO uploads (filename, timestamp, data) VALUES (?, ?, ?)",
                 (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), blob))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    records = conn.execute("SELECT id, filename, timestamp FROM uploads ORDER BY id DESC").fetchall()
    conn.close()
    return records

def load_df(upload_id):
    conn = sqlite3.connect(DB_PATH)
    blob = conn.execute("SELECT data FROM uploads WHERE id=?", (upload_id,)).fetchone()[0]
    conn.close()
    with open("temp.pkl", "wb") as f:
        f.write(blob)
    return pd.read_pickle("temp.pkl")


# ================= MODEL LOADING =================
@st.cache_resource
def load_engine():
    if os.path.exists("models/kmeans_model.pkl") and os.path.exists("models/scaler.pkl"):
        return (
            pickle.load(open("models/kmeans_model.pkl", "rb")),
            pickle.load(open("models/scaler.pkl", "rb"))
        )
    return None, None

model, scaler = load_engine()


# MAIN APP
if check_password():

    with st.sidebar:
        st.header("📂 Data Source")

        mode = st.radio("Select Source", ["Upload New Data", "Load From History"])

        file = None
        df_raw = None

        if mode == "Upload New Data":
            file = st.file_uploader("Upload File", type=["csv", "xlsx"])

        else:
            history = load_history()
            if history:
                names = [f"{r[1]} (ID:{r[0]})" for r in history]
                choice = st.selectbox("History", names)
                selected_id = int(choice.split("ID:")[1][:-1])
                df_raw = load_df(selected_id)
                st.success("Loaded from history ✔")
            else:
                st.info("No history available yet.")

        st.markdown("---")
        if st.button("🔴 Logout"):
            st.session_state["password_correct"] = False
            st.rerun()

        st.caption("Engine v2.4")

    st.title("💎 Revenue & Customer Intelligence Suite")

    if model is None:
        st.error("Model missing in `models/` folder")
    else:
        if file:
            if file.name.endswith("csv"):
                df_raw = pd.read_csv(file)
            else:
                df_raw = pd.read_excel(file, sheet_name=0)

            save_upload(file.name, df_raw)

        if df_raw is not None:
            with st.spinner("Processing..."):
                df = df_raw.dropna(subset=['Customer ID'])
                df['TotalPrice'] = df['Quantity'] * df['Price']

                latest = df['InvoiceDate'].max() + pd.Timedelta(days=1)

                rfm = df.groupby('Customer ID').agg({
                    'InvoiceDate': lambda x: (latest - x.max()).days,
                    'Invoice': 'nunique',
                    'TotalPrice': 'sum'
                }).rename(columns={'InvoiceDate':'Recency', 'Invoice':'Frequency', 'TotalPrice':'Monetary'})

                rfm = rfm[rfm['Monetary'] > 0]

                rfm_scaled = scaler.transform(np.log1p(rfm))
                rfm['Cluster'] = model.predict(rfm_scaled)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("📌 Total Clients", f"{len(rfm):,}")
            k2.metric("💰 Total Revenue", f"${rfm['Monetary'].sum():,.0f}")
            k3.metric("🧾 Avg Spend", f"${rfm['Monetary'].mean():,.2f}")
            k4.metric("📆 Avg Recency", f"{int(rfm['Recency'].mean())} days")

            st.markdown("---")

            tab1, tab2 = st.tabs(["📊 Visuals", "🎯 Strategy"])

            with tab1:
                colA, colB = st.columns([1,2])

                with colA:
                    fig_pie = px.pie(rfm, names='Cluster', hole=0.5, title="Customer Segments")
                    st.plotly_chart(fig_pie, use_container_width=True)

                with colB:
                    fig3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                                          color='Cluster', height=600)
                    fig3d.update_traces(marker=dict(size=3))
                    st.plotly_chart(fig3d, use_container_width=True)

            with tab2:
                st.subheader("AI Recommended Actions")

                mean_vals = rfm.mean()

                for i in range(model.n_clusters):
                    group = rfm[rfm['Cluster']==i]

                    avg_r = group['Recency'].mean()
                    avg_m = group['Monetary'].mean()

                    if avg_r < mean_vals['Recency'] and avg_m > mean_vals['Monetary']:
                        label = "🏆 High Value Customers"
                        plan = "Exclusive VIP loyalty perks"
                    elif avg_r > mean_vals['Recency'] * 1.4:
                        label = "⚠️ Churn Risk"
                        plan = "Urgent re-engagement campaigns"
                    else:
                        label = "📈 Growth Potential"
                        plan = "Upsell & Cross-sell strategy"

                    with st.expander(f"Segment {i} | {label}"):
                        st.write(f"Recommended Plan: **{plan}**")
                        st.write(f"Customers: {len(group)}")
                        st.download_button("Download List", group.to_csv().encode(),
                                           f"segment_{i}.csv")

        else:
            st.info("Please upload or select data to begin.")

