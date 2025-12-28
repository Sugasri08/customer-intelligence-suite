import pandas as pd
import datetime as dt
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    # Load data
    df = pd.read_excel(file_path, sheet_name='Year 2010-2011')
    
    # Cleaning
    df = df.dropna(subset=['Customer ID'])
    df['Invoice'] = df['Invoice'].astype(str) # Ensure string for 'C' check
    df = df[~df['Invoice'].str.contains('C', na=False)]
    df['TotalPrice'] = df['Quantity'] * df['Price']
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    return df

def create_rfm_table(df):
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    return rfm[rfm['Monetary'] > 0]

# --- EXECUTION ORDER ---
# 1. Load data first
raw_data = load_and_clean_data('data/raw/online_retail_II.xlsx')

# 2. Transform to RFM
rfm_df = create_rfm_table(raw_data)

# 3. Handle Skewness (Internship Pro-Tip: Use Log Transform)
# Real-world data has long tails; log makes it "Normal" for K-Means
rfm_log = np.log1p(rfm_df) # log1p handles log(0) safely

# 4. Scaling
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
rfm_scaled_df = pd.DataFrame(rfm_scaled, index=rfm_df.index, columns=rfm_df.columns)

print("Preprocessed Data for ML:")
print(rfm_scaled_df.head())