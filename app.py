import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os
import sqlite3
from datetime import datetime

# --- 1. System Settings & Custom UI ---
st.set_page_config(page_title="Intelligence Suite Pro", layout="wide")

st.markdown("""
<style>
.main { background: linear-gradient(135deg, #0c1117, #1a2330); color: #e2e8f0; }
h1, h2, h3 { font-family: 'Segoe UI'; color: #00eaff !important; }
div[data-testid="stMetric"] { 
    background: rgba(0,255,255,0.05); 
    border-radius: 15px; 
    padding: 20px; 
    border: 1px solid rgba(0,255,255,0.2); 
}
.ai-pulse { width: 100px; height: 20px; margin: 10px auto; display: flex; justify-content: space-between; }
.ai-pulse div { width: 15px; height: 15px; background-color: #00eaff; border-radius: 50%; animation: pulse 1s infinite ease-in-out; }
.ai-pulse div:nth-child(2) { animation-delay: 0.2s; }
.ai-pulse div:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse { 0%, 80%, 100% { transform: scale(0.2); opacity: 0.3;} 40% { transform: scale(1); opacity: 1;} }
</style>
""", unsafe_allow_html=True)

# --- 2. Database Intelligence ---
DB_PATH = "customer_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, timestamp TEXT, data BLOB)")
    conn.commit()
    conn.close()

def save_upload(filename, df):
    df.to_pickle("temp.pkl")
    with open("temp.pkl", "rb") as f:
        blob = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO uploads (filename, timestamp, data) VALUES (?, ?, ?)", (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), blob))
    conn.commit()
    conn.close()

init_db()

# --- 3. Asset Loading ---
@st.cache_resource
def load_engine():
    m_path, s_path = "models/kmeans_model.pkl", "models/scaler.pkl"
    if os.path.exists(m_path) and os.path.exists(s_path):
        return pickle.load(open(m_path, "rb")), pickle.load(open(s_path, "rb"))
    return None, None

model, scaler = load_engine()

# --- 4. Sidebar Control Panel ---
with st.sidebar:
    st.header("📂 Data Source")
    mode = st.radio("Mode", ["New Upload", "History"])
    df_raw = None
    
    if mode == "New Upload":
        file = st.file_uploader("Sync Logs", type=["csv", "xlsx"])
        if file:
            df_raw = pd.read_csv(file) if file.name.endswith('csv') else pd.read_excel(file)
            save_upload(file.name, df_raw)
    else:
        conn = sqlite3.connect(DB_PATH)
        history = conn.execute("SELECT id, filename FROM uploads ORDER BY id DESC").fetchall()
        conn.close()
        if history:
            choice = st.selectbox("Select Session", [f"{r[1]} (ID:{r[0]})" for r in history])
            s_id = choice.split("ID:")[1][:-1]
            conn = sqlite3.connect(DB_PATH)
            blob = conn.execute("SELECT data FROM uploads WHERE id=?", (s_id,)).fetchone()[0]
            conn.close()
            with open("temp.pkl", "wb") as f: f.write(blob)
            df_raw = pd.read_pickle("temp.pkl")

# --- 5. Main Processing & UI ---
st.title("💎 Customer Intelligence Command Center")

if model is None:
    st.error("System Core Missing: /models/ assets not found.")
elif df_raw is not None:
    st.markdown('<div class="ai-pulse"><div></div><div></div><div></div></div>', unsafe_allow_html=True)
    
    # Column Mapping UI
    cols = df_raw.columns.tolist()
    c1, c2, c3, c4 = st.columns(4)
    uid = c1.selectbox("Customer ID", cols)
    qty = c2.selectbox("Quantity", cols)
    prc = c3.selectbox("Unit Price", cols)
    dt_col = c4.selectbox("Invoice Date", cols)

    if st.button("🚀 Execute AI Segmentation"):
        with st.spinner("Analyzing Market Behaviors..."):
            # --- CRASH FIX START ---
            # Force conversion to numeric. If a user picked a Date column for Quantity, 
            # pd.to_numeric will turn it into NaN (Not a Number) rather than crashing.
            df_raw[qty] = pd.to_numeric(df_raw[qty], errors='coerce')
            df_raw[prc] = pd.to_numeric(df_raw[prc], errors='coerce')
            df_raw[dt_col] = pd.to_datetime(df_raw[dt_col], errors='coerce')
            
            # Drop rows where mapping was wrong or data is missing
            df = df_raw.dropna(subset=[uid, qty, prc, dt_col])
            
            # Now comparison is safe because everything is guaranteed to be numeric
            df = df[(df[qty] > 0) & (df[prc] > 0)]
            # --- CRASH FIX END ---

            df['Total'] = df[qty] * df[prc]

            # RFM Engine
            latest = df[dt_col].max() + pd.Timedelta(days=1)
            rfm = df.groupby(uid).agg({dt_col: lambda x: (latest - x.max()).days, 'Total': 'sum'})
            rfm.columns = ['Recency', 'Monetary']
            rfm['Frequency'] = df.groupby(uid)[dt_col].nunique()
            rfm = rfm[rfm['Monetary'] > 0]

            # Prediction
            scaled = scaler.transform(np.log1p(rfm[['Recency', 'Frequency', 'Monetary']]))
            rfm['Cluster'] = model.predict(scaled)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clients", f"{len(rfm):,}")
            m2.metric("Revenue", f"${rfm['Monetary'].sum():,.0f}")
            m3.metric("Avg Spend", f"${rfm['Monetary'].mean():,.2f}")
            m4.metric("Avg Recency", f"{int(rfm['Recency'].mean())}d")

            st.divider()

            # Visuals
            t1, t2 = st.tabs(["📊 Analytics", "🎯 Strategy"])
            with t1:
                v1, v2 = st.columns([1, 2])
                v1.plotly_chart(px.pie(rfm, names='Cluster', hole=0.5, template="plotly_dark", title="Segment Split"), use_container_width=True)
                v2.plotly_chart(px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', template="plotly_dark", title="Behavioral Projection"), use_container_width=True)

            with t2:
                # Dynamic Differentiated Labeling
                stats = rfm.groupby('Cluster').median()
                store_med_r = rfm['Recency'].median()
                store_med_m = rfm['Monetary'].median()
                
                for i in range(model.n_clusters):
                    group = rfm[rfm['Cluster'] == i]
                    if len(group) == 0: continue # Skip empty clusters
                    
                    med_r, med_f, med_m = stats.loc[i, 'Recency'], stats.loc[i, 'Frequency'], stats.loc[i, 'Monetary']
                    
                    # Business Tiers based on comparison to Store Median
                    if med_r < store_med_r and med_m > store_med_m:
                        label, plan, color = "🏆 Champions", "VIP loyalty program & early access.", "green"
                    elif med_r > rfm['Recency'].quantile(0.75):
                        label, plan, color = "⚠️ Churn Risk", "Re-engagement discount campaign.", "red"
                    elif med_f > rfm['Frequency'].median():
                        label, plan, color = "⭐ Loyalists", "Cross-sell new categories.", "blue"
                    else:
                        label, plan, color = "📈 Growth Potential", "Welcome coupons to drive 2nd purchase.", "orange"

                    with st.expander(f"Cluster {i} | {label}"):
                        st.markdown(f"**Strategy:** :{color}[{plan}]")
                        
                        char_data = pd.DataFrame({
                            "Metric": ["Recency (Days)", "Frequency (Orders)", "Monetary ($)"],
                            "Cluster Median": [f"{med_r:.0f}", f"{med_f:.0f}", f"${med_m:,.2f}"],
                            "Store Median": [f"{store_med_r:.0f}", f"{rfm['Frequency'].median():.0f}", f"${store_med_m:,.2f}"]
                        })
                        st.table(char_data)
                        st.download_button(f"Export Cluster {i}", group.to_csv().encode(), f"cluster_{i}.csv", key=f"dl_{i}")
else:
    st.info("Awaiting Data Stream. Please upload or select history from the sidebar.")