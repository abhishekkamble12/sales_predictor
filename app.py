import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Get the absolute path of the directory containing the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page Configuration
st.set_page_config(
    page_title="Online Sales Analytics & Forecast",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------------------------------------
# 1. LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, 'final_sales_model.pkl')
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please ensure 'final_sales_model.pkl' is in the app directory.")
        return None

model = load_model()

# ---------------------------------------------------------
# 2. SIDEBAR & NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Business Dashboard", "🤖 Sales Predictor"])

st.sidebar.markdown("---")
st.sidebar.info("Project: Online Sales Dataset Curation & Predictive Analytics")

st.title("Online Sales Analytics & Forecast")

# ---------------------------------------------------------
# 3. PAGE: BUSINESS DASHBOARD (Visualizations)
# ---------------------------------------------------------
if page == "📊 Business Dashboard":
    st.header("📊 Executive Sales Dashboard")
    st.markdown("Overview of historical performance and key metrics based on the merged retail dataset.")

    # --- Load Actual Data ---
    @st.cache_data
    def load_data():
        data_path = os.path.join(BASE_DIR, 'Merged_Retail_Dataset.xlsx')
        if not os.path.exists(data_path):
            st.error(f"⚠️ Dataset not found at {data_path}. Please place 'Merged_Retail_Dataset.xlsx' in the app directory.")
            st.info("For testing, you can generate a sample dataset using the provided script.")
            return pd.DataFrame()
        
        try:
            df = pd.read_excel(data_path, engine='openpyxl')
            # Normalize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Handle date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                st.warning("No 'date' column found. Dashboard features may be limited.")
                return df
            
            # Filter completed transactions
            if 'status' in df.columns:
                df = df[df['status'].str.lower() == 'completed']
            
            # Calculate total revenue per transaction (use 'amount' directly)
            if 'amount' in df.columns:
                df['total_price'] = df['amount']
            else:
                st.warning("No 'amount' column found. Using default values.")
                df['total_price'] = 0
            
            df['month'] = df['date'].dt.month_name()
            df['year_month'] = df['date'].dt.to_period('M')
            df['day_of_week'] = df['date'].dt.day_name()
            
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()

    df = load_data()

    if not df.empty:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_revenue = df['total_price'].sum()
        avg_transaction = df['total_price'].mean()
        total_transactions = len(df)
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 'N/A'

        with col1:
            st.metric("💰 Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            st.metric("🛒 Avg Transaction Value", f"${avg_transaction:,.2f}")
        with col3:
            st.metric("📦 Total Transactions", f"{total_transactions:,}")
        with col4:
            st.metric("⭐ Avg Rating", f"{avg_rating:.2f}" if isinstance(avg_rating, float) else avg_rating)

        st.markdown("---")

        # ROW 1: Revenue Trend & Sales by Store
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("📈 Monthly Revenue Trend")
            if 'year_month' in df.columns:
                monthly_sales = df.groupby('year_month')['total_price'].sum()
                st.line_chart(monthly_sales)
                st.caption("Aggregated monthly sales revenue")
            else:
                st.warning("Unable to generate monthly trend.")

        with col_right:
            st.subheader("🏬 Sales by Store")
            if 'storeid' in df.columns:
                store_sales = df.groupby('storeid')['total_price'].sum().nlargest(10)
                fig, ax = plt.subplots()
                ax.pie(store_sales, labels=store_sales.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                st.caption("Top stores by revenue distribution")
            else:
                st.warning("No 'storeid' column found.")

        # ROW 2: Top Products & Sales by Day of Week
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Top Products by Revenue")
            if 'productid' in df.columns:
                top_products = df.groupby('productid')['total_price'].sum().nlargest(10)
                st.bar_chart(top_products)
                st.caption("Top 10 products based on ProductID")
            else:
                st.warning("No 'productid' column found.")

        with col2:
            st.subheader("📅 Sales by Day of Week")
            weekly_sales = df.groupby('day_of_week')['total_price'].sum()
            ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_sales = weekly_sales.reindex(ordered_days)
            st.bar_chart(weekly_sales)
            st.caption("Revenue distribution across days")

        # ROW 3: Additional Insights
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💳 Payment Mode Distribution")
            if 'paymentmode' in df.columns:
                payment_dist = df['paymentmode'].value_counts()
                fig, ax = plt.subplots()
                ax.pie(payment_dist, labels=payment_dist.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.warning("No 'paymentmode' column found.")

        with col2:
            st.subheader("📊 Rating Distribution")
            if 'rating' in df.columns:
                fig, ax = plt.subplots()
                sns.histplot(df['rating'], bins=5, kde=False, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No 'rating' column found.")

# ---------------------------------------------------------
# 4. PAGE: SALES PREDICTOR (Model Interface)
# ---------------------------------------------------------
elif page == "🤖 Sales Predictor":
    st.header("🤖 Daily Sales Forecaster")
    st.markdown("Use the trained ensemble model to predict future daily revenue. Inputs based on historical lags and trends.")

    if model is None:
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📅 Date Parameters")
        selected_date = st.date_input("Select Date for Prediction", value=datetime.now())
        
        day_of_week = selected_date.weekday()
        month = selected_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        st.info(f"""
        **Derived Features:**
        - Day of Week: {day_of_week} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]})
        - Month: {month}
        - Weekend: {'Yes' if is_weekend else 'No'}
        """)

    with col2:
        st.subheader("📉 Historical Context")
        lag_1 = st.number_input("Sales Yesterday ($)", min_value=0.0, value=200.0)
        lag_7 = st.number_input("Sales 7 Days Ago ($)", min_value=0.0, value=190.0)
        rolling_mean_7 = st.number_input("7-Day Rolling Mean ($)", min_value=0.0, value=195.0)

    if st.button("🚀 Predict Revenue"):
        input_data = pd.DataFrame([[day_of_week, month, is_weekend, lag_1, lag_7, rolling_mean_7]],
                                  columns=['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7'])
        
        prediction = model.predict(input_data)[0]
        
        st.success(f"Predicted Daily Revenue: ${prediction:,.2f}")
        
        if prediction > rolling_mean_7:
            st.info("📈 Uptrend detected compared to weekly average.")
        else:
            st.warning("📉 Downtrend detected compared to weekly average.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Powered by Streamlit  | Retail Sales Analytics v1.0</p>", unsafe_allow_html=True)