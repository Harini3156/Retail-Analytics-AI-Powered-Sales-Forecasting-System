
# 🛍️ Retail Sales Analytics & AI Forecasting System

An end-to-end interactive Business Intelligence and Predictive Analytics platform built with **Python**, **Streamlit**, and **Machine Learning**. This application processes multi-store, multi-category sales data to deliver actionable insights, identify seasonal demand trends, forecast future revenue, and cluster retail locations based on performance metrics.

---

## ✨ Key Features

* **📊 Executive Dashboard:** High-level KPIs including Total Revenue, Units Sold, Average Order Value (AOV), and overall monthly growth trends.
* **📈 Multi-Store & Category Performance:** Interactive visualizations breaking down revenue contribution across store locations and product segments.
* **🌊 Seasonality & Trend Identification:** Analysis of daily purchase cycles, peak sales periods, and store-category cross-heatmaps.
* **🔮 Time-Series Revenue Forecasting:** Predictive modeling powered by **Holt-Winters Exponential Smoothing** (Statsmodels) with configurable forecast horizons (7 to 90 days).
* **🎯 Performance Segmentation:** Unsupervised **K-Means Clustering** (Scikit-Learn) categorizing store performance into actionable behavioral tiers.

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python 3.9+
* **Frontend / UI Framework:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `plotly`
* **Machine Learning & Time-Series:** `scikit-learn`, `statsmodels`

---

## 🚀 Quick Start Guide
Set Up Project Folder:
Open terminal and navigate into a new project directory:

Bash
mkdir retail_analytics_system
cd retail_analytics_system
Create & Activate Virtual Environment:

Windows:

Bash
python -m venv venv
venv\Scripts\activate
macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
Install Dependencies:

Bash
pip install streamlit pandas numpy plotly scikit-learn statsmodels
Add Code:
Create a file named app.py in the folder and paste the full Python script into it.

Run the Dashboard:

Bash
streamlit run app.py
(The app will open automatically in your browser at http://localhost:8501)
