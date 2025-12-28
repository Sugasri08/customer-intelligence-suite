import pandas as pd
import numpy as np
import datetime as dt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. We need the function definition inside this file too
def load_and_prep(file_path):
    df = pd.read_excel(file_path, sheet_name='Year 2010-2011')
    df = df.dropna(subset=['Customer ID'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})
    
    rfm_log = np.log1p(rfm[rfm['Monetary'] > 0])
    
    # CRITICAL: We define the scaler here so we can save it later
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(rfm_log)
    
    df_scaled = pd.DataFrame(scaled_values, index=rfm_log.index, columns=rfm_log.columns)
    return df_scaled, scaler # Return both the data AND the scaler object

# --- EXECUTION ---
try:
    k_optimal = 4 
    # Update variables to receive both outputs
    rfm_scaled_df, scaler = load_and_prep('data/raw/online_retail_II.xlsx')

    # Fit Model
    model = KMeans(n_clusters=k_optimal, init='k-means++', n_init=10, random_state=42)
    model.fit(rfm_scaled_df)

    # Save Objects
    os.makedirs('models', exist_ok=True)

    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Success! Model and Scaler saved in the 'models/' folder.")

except Exception as e:
    print(f"❌ Error during training: {e}")