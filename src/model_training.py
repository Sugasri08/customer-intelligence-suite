import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load and Clean
def load_and_prep(file_path):
    # Load the specific sheet
    df = pd.read_excel(file_path, sheet_name='Year 2010-2011')
    
    # Cleaning
    df = df.dropna(subset=['Customer ID'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # RFM Aggregation
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'})
    
    # Preprocessing (Log Transform + Scaling)
    # We use log1p to handle the high variance in Monetary/Frequency
    rfm_log = np.log1p(rfm[rfm['Monetary'] > 0])
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(rfm_log)
    
    # Return as DataFrame for easier handling
    return pd.DataFrame(scaled_values, index=rfm_log.index, columns=rfm_log.columns)

# 2. Optimal K Function
def find_optimal_k(df_scaled):
    inertia = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# 3. FIXED EXECUTION
file_path = 'data/raw/online_retail_II.xlsx' # Ensure this path is correct!

try:
    # TYPO FIXED: Changed load_prep to load_and_prep
    rfm_scaled_df = load_and_prep(file_path)
    print("Data loaded and scaled successfully.")
    
    # Run the visualization
    find_optimal_k(rfm_scaled_df)
    
except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}. Please check the folder structure.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")