import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# 1. PAGE CONFIGURATION & LAYOUT SETUP
# ==========================================
st.set_page_config(
    page_title="Retail Sales Analytics & AI Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ==========================================
# 2. DATA LOAD / SYNTHETIC DATA GENERATOR
# ==========================================
@st.cache_data
def load_sample_data():
    """Generates synthetic multi-store, multi-category retail sales dataset."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")
    stores = [f"Store_{i}" for i in range(1, 6)]
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Groceries"]
    
    data = []
    for date in dates:
        # Yearly seasonality (Q4 boost)
        seasonality = 1.0 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365) + (0.5 if date.month == 12 else 0)
        # Weekend boost
        weekend = 1.25 if date.weekday() >= 5 else 1.0
        
        for store in stores:
            store_multiplier = float(store.split("_")[1]) * 0.3 + 0.7
            for category in categories:
                base_sales = np.random.uniform(200, 1000)
                sales = base_sales * seasonality * weekend * store_multiplier
                units = int(sales / np.random.uniform(15, 50))
                
                data.append({
                    "Date": date,
                    "Store": store,
                    "Category": category,
                    "Sales": round(sales, 2),
                    "Units_Sold": max(1, units)
                })
                
    df = pd.DataFrame(data)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["DayOfWeek"] = df["Date"].dt.day_name()
    return df

# Initialize Data Engine
df = load_sample_data()

# ==========================================
# 3. GLOBAL CONTROLS & SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Global Dashboard Filters")
selected_stores = st.sidebar.multiselect(
    "Select Stores", 
    options=df["Store"].unique(), 
    default=df["Store"].unique()
)
selected_categories = st.sidebar.multiselect(
    "Select Product Categories", 
    options=df["Category"].unique(), 
    default=df["Category"].unique()
)

# Filter Dataframe based on UI selections
filtered_df = df[(df["Store"].isin(selected_stores)) & (df["Category"].isin(selected_categories))]

# ==========================================
# 4. DASHBOARD HEADER & TAB NAVIGATION
# ==========================================
st.title("🛍️ Retail Sales Analytics & AI Forecasting System")
st.markdown("An end-to-end suite for Multi-Store Analytics, Seasonal Pattern Recognition, Time-Series Forecasting, and Performance Clustering.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary",
    "📈 Multi-Store & Category Analysis",
    "🌊 Seasonality & Patterns",
    "🔮 Time-Series Revenue Forecast",
    "🎯 Store Performance Segmentation"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.subheader("High-Level Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = filtered_df["Sales"].sum()
    total_units = filtered_df["Units_Sold"].sum()
    avg_order_value = filtered_df["Sales"].mean()
    active_stores = filtered_df["Store"].nunique()
    
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Units Sold", f"{total_units:,}")
    col3.metric("Avg Order Value", f"${avg_order_value:.2f}")
    col4.metric("Active Stores Selected", active_stores)
    
    st.markdown("---")
    st.subheader("Monthly Sales Growth Trajectory")
    monthly_sales = filtered_df.groupby("Month")["Sales"].sum().reset_index()
    fig_monthly = px.line(monthly_sales, x="Month", y="Sales", markers=True, title="Monthly Aggregate Sales Trend")
    st.plotly_chart(fig_monthly, use_container_width=True)

# ==========================================
# TAB 2: MULTI-STORE & CATEGORY ANALYSIS
# ==========================================
with tab2:
    st.subheader("Objective 1: Multi-Store & Product Category Breakdowns")
    col_a, col_b = st.columns(2)
    
    with col_a:
        store_sales = filtered_df.groupby("Store")["Sales"].sum().reset_index()
        fig_store = px.bar(store_sales, x="Store", y="Sales", color="Store", title="Revenue Contribution by Store")
        st.plotly_chart(fig_store, use_container_width=True)
        
    with col_b:
        cat_sales = filtered_df.groupby("Category")["Sales"].sum().reset_index()
        fig_cat = px.pie(cat_sales, names="Category", values="Sales", title="Revenue Share by Category", hole=0.4)
        st.plotly_chart(fig_cat, use_container_width=True)

# ==========================================
# TAB 3: SEASONALITY & TREND IDENTIFICATION
# ==========================================
with tab3:
    st.subheader("Objective 2: Key Sales Patterns, Growth Drivers & Seasonality")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        # Day of Week Demand Distribution
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_sales = filtered_df.groupby("DayOfWeek")["Sales"].mean().reindex(day_order).reset_index()
        fig_dow = px.bar(dow_sales, x="DayOfWeek", y="Sales", title="Average Sales by Day of Week", color_discrete_sequence=["#2E86C1"])
        st.plotly_chart(fig_dow, use_container_width=True)
        
    with col_s2:
        # Cross-Heatmap: Store vs Category
        pivot_df = filtered_df.pivot_table(index="Store", columns="Category", values="Sales", aggfunc="sum")
        fig_heat = px.imshow(pivot_df, text_auto=".2s", aspect="auto", title="Revenue Heatmap (Store vs Category)")
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 4: TIME-SERIES REVENUE FORECASTING
# ==========================================
with tab4:
    st.subheader("Objective 3: Revenue Forecasting via Holt-Winters Model")
    
    forecast_days = st.slider("Select Forecast Horizon (Days):", min_value=7, max_value=90, value=30)
    
    # Resample daily revenue series
    daily_ts = filtered_df.groupby("Date")["Sales"].sum().asfreq("D", fill_value=0)
    
    # Train Exponential Smoothing Time-Series Model
    model = ExponentialSmoothing(daily_ts, trend="add", seasonal="add", seasonal_periods=7).fit()
    forecast = model.forecast(forecast_days)
    
    # Structure Forecast Data
    historical_df = pd.DataFrame({"Date": daily_ts.index, "Sales": daily_ts.values, "Type": "Historical"})
    forecast_dates = pd.date_range(start=daily_ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Sales": forecast.values, "Type": "Forecast"})
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    fig_forecast = px.line(
        combined_df, 
        x="Date", 
        y="Sales", 
        color="Type", 
        title=f"Next {forecast_days}-Day Sales Prediction",
        color_discrete_map={"Historical": "blue", "Forecast": "red"}
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ==========================================
# TAB 5: STORE SEGMENTATION (K-MEANS)
# ==========================================
with tab5:
    st.subheader("Objective 4: Store Performance & Behavioral Clustering")
    
    # Aggregate store performance metrics
    store_metrics = df.groupby("Store").agg(
        Total_Revenue=("Sales", "sum"),
        Total_Units=("Units_Sold", "sum"),
        Avg_Transaction=("Sales", "mean")
    ).reset_index()
    
    # Scale Features
    features = ["Total_Revenue", "Total_Units", "Avg_Transaction"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(store_metrics[features])
    
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    store_metrics["Cluster"] = kmeans.fit_predict(scaled_features)
    store_metrics["Segment"] = store_metrics["Cluster"].map({
        0: "Moderate Performers",
        1: "Top Performers",
        2: "Low-Volume Stores"
    })
    
    col_cl1, col_cl2 = st.columns([1, 2])
    
    with col_cl1:
        st.write("### Store Segments")
        st.dataframe(store_metrics[["Store", "Total_Revenue", "Avg_Transaction", "Segment"]])
        
    with col_cl2:
        fig_scatter = px.scatter(
            store_metrics, 
            x="Total_Revenue", 
            y="Avg_Transaction", 
            color="Segment", 
            size="Total_Units", 
            hover_name="Store",
            title="Store Clusters: Total Revenue vs Average Transaction"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
