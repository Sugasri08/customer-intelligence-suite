import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_messy_data():
    base_date = datetime(2024, 1, 1)
    
    # 1. Global Tech Store (Decimal Quantities)
    df1 = pd.DataFrame({
        'User_ID': np.random.randint(1000, 1050, 100),
        'Qty': np.random.uniform(0.5, 10.0, 100),
        'Unit_Cost': np.random.uniform(10, 500, 100),
        'Order_Date': [base_date + timedelta(days=x) for x in range(100)]
    })
    df1.to_csv("global_tech_store.csv", index=False)

    # 2. UK Boutique (Negative Values/Returns)
    df2 = pd.DataFrame({
        'CustomerID': np.random.randint(2000, 2050, 100),
        'Quantity': np.random.choice([5, 10, -1, 2], 100), # Includes -1 for returns
        'Price': np.random.uniform(5, 50, 100),
        'InvoiceDate': [base_date + timedelta(days=x) for x in range(100)]
    })
    df2.to_csv("uk_boutique.csv", index=False)

    # 3. Apparel Co. (Missing Values)
    df3 = pd.DataFrame({
        'ID': np.random.randint(3000, 3050, 100),
        'Items': np.random.randint(1, 5, 100),
        'Amount': [np.nan if i % 10 == 0 else np.random.uniform(20, 100) for i in range(100)],
        'Timestamp': [base_date + timedelta(days=x) for x in range(100)]
    })
    df3.to_csv("apparel_co.csv", index=False)

    print("✅ 5 Messy datasets generated successfully!")

generate_messy_data()