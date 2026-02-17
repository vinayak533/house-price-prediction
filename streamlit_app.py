import streamlit as st
import requests
import plotly.graph_objects as go

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="House Price Intelligence",
    page_icon="🏠",
    layout="wide"
)

# FastAPI Endpoint
API_URL = "http://localhost:8000/predict"

# ======================
# CUSTOM CSS (FinTech Dark Theme)
# ======================
st.markdown("""
<style>

.main {
    background: #020617;
    color: white;
}

.kpi-card {
    background: #1e293b;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border: 1px solid #334155;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.result-card {
    background: linear-gradient(135deg,#2563eb,#1d4ed8);
    padding:35px;
    border-radius:18px;
    text-align:center;
    color:white;
    font-size:30px;
    font-weight:bold;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.4);
    animation: fadeIn 0.7s ease-in-out;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(10px);}
    to {opacity:1; transform: translateY(0);}
}

.sidebar .sidebar-content {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "About Model"])

st.sidebar.markdown("---")
st.sidebar.info("Built by Vinayak K V")

# ======================
# DASHBOARD
# ======================
if page == "Dashboard":

    st.title("🏠 House Price Prediction Dashboard")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="kpi-card"><h3>📊 Avg Market Price</h3><h2>$350,000</h2></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '<div class="kpi-card"><h3>📈 Model Accuracy</h3><h2>92%</h2></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            '<div class="kpi-card"><h3>⚡ Prediction Speed</h3><h2>0.8s</h2></div>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Real-Time Prediction",
        "📊 Market Insights",
        "🤖 Model Performance"
    ])

    # ======================
    # TAB 1 — PREDICTION
    # ======================
    with tab1:

        st.subheader("Enter Property Details")

        c1, c2 = st.columns(2)

        with c1:
            median_income = st.number_input("💰 Median Income", value=5.0)
            housing_age = st.number_input("🏗 Housing Median Age", value=20.0)
            total_rooms = st.number_input("🛋 Total Rooms", value=2000.0)
            total_bedrooms = st.number_input("🛏 Total Bedrooms", value=400.0)

        with c2:
            population = st.number_input("👥 Population", value=1000.0)
            households = st.number_input("🏠 Households", value=300.0)
            latitude = st.number_input("📍 Latitude", value=34.0)
            longitude = st.number_input("📍 Longitude", value=-118.0)

        ocean = st.selectbox(
            "🌊 Ocean Proximity",
            ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
        )

        predict_btn = st.button("🚀 Predict Price")

        if predict_btn:

            with st.spinner("Valuating property..."):

                payload = {
                    "longitude": longitude,
                    "latitude": latitude,
                    "housing_median_age": housing_age,
                    "total_rooms": total_rooms,
                    "total_bedrooms": total_bedrooms,
                    "population": population,
                    "households": households,
                    "median_income": median_income,
                    "ocean_proximity": ocean
                }

                try:
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:

                        prediction = response.json()["predicted_price"]

                        st.markdown(
                            f'<div class="result-card">Estimated Value<br>${prediction:,.2f}</div>',
                            unsafe_allow_html=True
                        )

                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            title={'text': "Price Position"},
                            gauge={
                                'axis': {'range': [0, 500000]},
                                'bar': {'color': "#2563eb"},
                                'steps': [
                                    {'range': [0, 200000], 'color': "#1e293b"},
                                    {'range': [200000, 350000], 'color': "#334155"},
                                    {'range': [350000, 500000], 'color': "#475569"}
                                ]
                            }
                        ))

                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error("API Error")

                except Exception as e:
                    st.error(f"Could not connect to FastAPI: {e}")

    # ======================
    # TAB 2 — MARKET INSIGHTS
    # ======================
    with tab2:

        st.subheader("Market Insights")

        st.info("This section can include price distribution, location trends, etc.")

        # Example chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["INLAND", "NEAR BAY", "NEAR OCEAN"],
            y=[250000, 400000, 450000]
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ======================
    # TAB 3 — MODEL PERFORMANCE
    # ======================
    with tab3:

        st.subheader("Model Performance")

        st.write("Model: HistGradientBoostingRegressor")
        st.write("R² Score: 0.92")
        st.write("RMSE: 48,000")

        st.success("Production-ready ML pipeline deployed with FastAPI")

# ======================
# ABOUT PAGE
# ======================
if page == "About Model":

    st.title("🤖 About the Model")

    st.write("""
This application predicts house prices using a Machine Learning model trained on housing-related features 
such as income, population, location, and housing characteristics.
             
🏗️ Architecture
- Scikit-Learn Pipeline for preprocessing and prediction
- FastAPI backend for serving the model
- Streamlit frontend for interactive user experience
- Plotly visualizations for insights

🤖 Models Used
- Linear Regression
- Ridge & Lasso Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor

The final model was selected based on cross-validation performance and evaluation metrics.

 🎯 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

This application is designed to simulate a production-ready Machine Learning system with real-time prediction capability.

👨‍💻 Developer: **Vinayak K V**
""")
