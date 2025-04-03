# app.py - COMPLETE VERSION WITH SHAP
import streamlit as st
import joblib
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import shap  # New import

# Configure page
st.set_page_config(
    page_title="Delhi Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background: #0F2027;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        width: 100%;
        box-sizing: border-box;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        letter-spacing: -1px;
        line-height: 1.2;
        margin: 0.5rem 0;
        min-height: 60px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        height: 20%;
    }
    
    .metric-description {
        font-size: 0.8rem;
        opacity: 0.8;
        line-height: 1.3;
        height: 30%;
    }

    [data-testid="column"] {
        min-width: 0;
        flex: 1 1 0;
        width: 100%;
    }
    
    @media (max-width: 1200px) {
        [data-testid="column"] {
            min-width: 200px !important;
        }
    }
    
    .stButton>button {
        background: #2CA58D;
        color: white;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        background: #218f79;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(44,165,141,0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #0F2027 0%, #203A43 100%);
    }
    
    .status-good {
        background: rgba(44,165,141,0.15);
        border: 1px solid #2CA58D;
    }
    
    .status-bad {
        background: rgba(242,108,79,0.15);
        border: 1px solid #F26C4F;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #2CA58D 0%, #4B8BBE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Model Data
try:
    features = joblib.load("model_features.pkl")
    metrics = joblib.load("model_metrics.pkl")
    current_demand = joblib.load("last_demand.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    forecast_horizons = ["target_1h", "target_3h", "target_6h"]
    models = {h: joblib.load(f"energy_forecast_{h}.pkl") for h in forecast_horizons}
except FileNotFoundError as e:
    st.error(f"Model files missing! Error: {str(e)}")
    st.stop()

# Sidebar - About Section
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 2rem 1rem; text-align: center;">
        <h1 class="gradient-text" style="font-size: 2rem;">Grid Intelligence</h1>
        <div style="margin: 2rem 0;">
            <p style="color: #4B8BBE; font-size: 1.1rem;">Developed by</p>
            <p style="font-size: 1.2rem;">Vikram & Ankit</p>
            <div style="margin: 1.5rem 0;">
                <div style="font-size: 0.9rem; color: #2CA58D;">Version 1.0.01</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">Multi-Horizon Forecasting</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.title("⚡ Delhi Grid Stabilization Forecast")
st.markdown("### Multi-Horizon Energy Demand Prediction System")

# Performance Metrics Dashboard
with st.container():
    cols = st.columns(3)
    horizon_metrics = [
        ("1-Hour Forecast", "target_1h", "#2CA58D"),
        ("3-Hour Forecast", "target_3h", "#4B8BBE"),
        ("6-Hour Forecast", "target_6h", "#6C5B7B")
    ]
    
    for col, (label, target, color) in zip(cols, horizon_metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color};">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color};">
                        R²: {metrics[target]['r2']:.3f}<br>
                        RMSE: {metrics[target]['rmse']:.1f}
                    </div>
                    <div class="metric-description">
                        MAE: {metrics[target]['mae']:.1f} kWh
                    </div>
                </div>
            """, unsafe_allow_html=True)

# Prediction Interface
with st.container():
    st.header("Forecast Parameters")
    inputs = {}
    
    # Time Configuration
    with st.expander("⏰ Time Configuration", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            date = st.date_input("Select Date", datetime.date.today())
            time = st.time_input("Select Time", datetime.time(12, 0))
            
        with cols[1]:
            timestamp = pd.to_datetime(f"{date} {time}")
            inputs.update({
                "hour": timestamp.hour,
                "dayofweek": timestamp.dayofweek,
                "month": timestamp.month,
                "Weekday": 1 if timestamp.dayofweek < 5 else 0,
                "Holiday_Indicator": st.checkbox("Public Holiday", False),
                "is_peak": 1 if 17 <= timestamp.hour <= 21 else 0
            })

    # Weather Conditions
    with st.expander("🌤️ Weather Conditions"):
        cols = st.columns(2)
        with cols[0]:
            inputs.update({
                "Temperature_C": st.slider("Temperature (°C)", -5, 45, 25),
                "Humidity_%": st.slider("Humidity (%)", 0, 100, 60)
            })
        with cols[1]:
            inputs.update({
                "Wind_Speed_mps": st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0),
                "Solar_Radiation_Wm2": st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
            })
        
        # Calculate derived features
        inputs["solar_wind"] = inputs["Solar_Radiation_Wm2"] * inputs["Wind_Speed_mps"]
        inputs["heat_index"] = 0.5 * (
            inputs["Temperature_C"] + 61.0 + 
            ((inputs["Temperature_C"] - 68.0) * 1.2) + 
            (inputs["Humidity_%"] * 0.094)
        )
        inputs["effective_temp"] = inputs["Temperature_C"] * inputs["Humidity_%"] / 100

    # System Parameters
    with st.expander("🏭 System Parameters"):
        cols = st.columns(3)
        with cols[0]:
            inputs.update({
                "Industrial_Usage_kWh": st.slider("Industrial Usage (kWh)", 0, 2000, 800),
                "Residential_Usage_kWh": st.slider("Residential Usage (kWh)", 0, 1500, 500),
                "Commercial_Usage_kWh": st.slider("Commercial Usage (kWh)", 0, 1000, 300)
            })
            # Calculate temp_usage dynamically
            inputs["temp_usage"] = inputs["Temperature_C"] * (
                inputs["Residential_Usage_kWh"] + inputs["Commercial_Usage_kWh"]
            )
        
        with cols[1]:
            inputs.update({
                "Voltage_Level_V": st.slider("Voltage Level (V)", 220, 440, 240),
                "Grid_Frequency_Hz": st.slider("Grid Frequency (Hz)", 49.0, 51.0, 50.0)
            })
        
        with cols[2]:
            inputs.update({
                "Renewable_Energy_Contribution_%": st.slider("Renewables Contribution (%)", 0, 100, 20),
                "rolling_demand_24h": st.slider("24h Avg Demand", 0, 5000, 1500)
            })
        
        inputs["rolling_demand_3h"] = st.slider("3h Avg Demand", 0, 5000, 1500)

    # Prediction Execution
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Analyzing grid patterns..."):
            try:
                # Prepare input with correct feature order
                input_df = pd.DataFrame([inputs])[features]
                scaled_input = scaler.transform(input_df)
                
                predictions = {}
                for target, model in models.items():
                    predictions[target] = model.predict(scaled_input)[0]

                st.success("Multi-Horizon Forecast Generated!")
                
                # Visualization
                fig = px.line(
                    x=["1-Hour", "3-Hour", "6-Hour"],
                    y=[predictions["target_1h"], predictions["target_3h"], predictions["target_6h"]],
                    markers=True,
                    title="Energy Demand Forecast Horizon",
                    labels={"x": "Forecast Horizon", "y": "Energy Demand (kWh)"},
                    color_discrete_sequence=["#2CA58D"]
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Capacity Planning
                peak_value = max(predictions.values())
                peak_horizon = [k for k, v in predictions.items() if v == peak_value][0]
                
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Peak Demand Expected</div>
                            <div class="metric-value" style="color: #F26C4F;">
                                {peak_horizon.replace('target_','').replace('h','')}-Hour
                            </div>
                            <div class="metric-description">
                                {peak_value:.0f} kWh
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    required_capacity = max(0, peak_value - current_demand)
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Additional Capacity Needed</div>
                            <div class="metric-value" style="color: #2CA58D;">
                                {required_capacity:.0f} kWh
                            </div>
                            <div class="metric-description">
                                Current Capacity: {current_demand:.0f} kWh
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    renewable_adjustment = ((peak_value / current_demand) * 100 - 100) if current_demand > 0 else 0
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Renewables Adjustment</div>
                            <div class="metric-value" style="color: #9B59B6;">
                                {max(0, renewable_adjustment):.1f}%
                            </div>
                            <div class="metric-description">
                                Recommended increase
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # SHAP Feature Importance Visualization
                with st.container():
                    st.header("Model Explainability")
                    
                    # Create SHAP explainer
                    explainer = shap.TreeExplainer(models["target_1h"])
                    
                    # Calculate SHAP values
                    shap_values = explainer.shap_values(scaled_input)
                    
                    # Create importance DataFrame
                    feature_importance = pd.DataFrame({
                        "Feature": features,
                        "Impact": np.abs(shap_values).mean(0)
                    }).sort_values("Impact", ascending=False)

                    # Create horizontal bar chart
                    fig = px.bar(
                        feature_importance,
                        x="Impact",
                        y="Feature",
                        orientation='h',
                        title="Feature Impact Analysis (SHAP Values)",
                        labels={"Impact": "Mean Absolute Impact", "Feature": ""},
                        color_discrete_sequence=["#2CA58D"]
                    )

                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        height=600,
                        yaxis={'categoryorder':'total ascending'},
                        margin=dict(l=120, r=20, t=60, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Footer
st.markdown("""
    <div style="margin-top: 3rem; text-align: center; opacity: 0.7;">
        <small>© 2024 Delhi Grid Intelligence | Version 1.0.01 </small>
    </div>
""", unsafe_allow_html=True)