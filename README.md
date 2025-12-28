💎 Customer Intelligence Suite

AI-powered dashboard to analyze customer behavior, predict churn, and uncover revenue-driving segments.

Upload your transactional data → Get instant insights → Make strategic decisions.
Perfect for retail, marketing, and business intelligence applications.

🔹 About

This project combines RFM (Recency, Frequency, Monetary) analysis with K-Means clustering to segment customers and provide actionable insights:

Identify high-value customers

Detect churn-risk customers

Recommend cross-sell/upsell opportunities

Track uploaded datasets with persistent history

🛠 Tech Stack

Frontend & UI: Streamlit

Data Processing: Pandas, NumPy

Machine Learning: Scikit-Learn (K-Means, Scaler)

Visualizations: Plotly

Database: SQLite (persistent storage for uploads)

Deployment: Streamlit Cloud

⚙️ Workflow

Secure Login: Password-protected dashboard access

Data Upload / Load History: CSV/XLSX files or previous uploads

Data Cleaning & RFM Feature Engineering: Computes Recency, Frequency, Monetary

Machine Learning Inference: Clustering with K-Means

Dashboard Metrics & Visuals:

Total customers, revenue, average spend, recency

Pie chart & 3D cluster visualization

Actionable Strategy Recommendations: Tiered suggestions per cluster

Export & History: Download customer segments & revisit uploaded datasets anytime

💻 Installation
# Clone the repository
git clone https://github.com/Sugasri08/customer-intelligence-suite.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
