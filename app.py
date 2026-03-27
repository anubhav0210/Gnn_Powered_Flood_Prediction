"""
Godavari Basin Water Level Predictor - Main Streamlit App
AI-powered 7-day forecasting with GNN for flood warning
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from pathlib import Path

import config
from utils import flood_warning, map_generator, data_loader, prediction_engine, visualization

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .map-container {
        margin: 0px;
        padding: 0px;
    }
    iframe {
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.markdown("## ⚙️ Settings")

# Get available date range
date_range_info = data_loader.get_date_range_info()
available_dates = data_loader.get_available_dates()

# Date selection
st.sidebar.markdown("### 📅 Date Selection")
st.sidebar.info(f"Data available from {date_range_info['start_date'].strftime('%Y-%m-%d')} to {date_range_info['end_date'].strftime('%Y-%m-%d')} ({date_range_info['num_days']} days)")

# Selection mode
selection_mode = st.sidebar.radio(
    "Choose date selection method:",
    ("Slider", "Date Picker", "Date Number", "Latest Date"),
    horizontal=False
)

if selection_mode == "Slider":
    # Slider to select the END date for the 30-day window
    selected_date_idx = st.sidebar.slider(
        "Select end date for 30-day prediction window:",
        min_value=29,  # Need at least 30 days of history
        max_value=len(available_dates) - 1,
        value=len(available_dates) - 1,  # Default to latest date
        step=1
    )
    selected_end_date = available_dates[selected_date_idx]
    
elif selection_mode == "Date Picker":
    # Date picker for direct date selection
    selected_date_obj = st.sidebar.date_input(
        "Select date for prediction:",
        value=available_dates[-1],
        min_value=available_dates[29],
        max_value=available_dates[-1]
    )
    # Find the closest available date
    selected_end_date = pd.to_datetime(selected_date_obj)
    if selected_end_date not in available_dates:
        # Find closest date
        closest_idx = min(range(len(available_dates)), key=lambda i: abs(available_dates[i] - selected_end_date))
        selected_end_date = available_dates[closest_idx]
        st.sidebar.warning(f"📌 Closest available date: {selected_end_date.strftime('%Y-%m-%d')}")
    
elif selection_mode == "Date Number":
    # Input by date number (like option 3 in predict_and_plot.py)
    st.sidebar.write("**Enter date number (1-{}):**".format(len(available_dates)))
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        date_number = st.number_input(
            "Date number:",
            min_value=30,
            max_value=len(available_dates),
            value=len(available_dates),
            step=1,
            label_visibility="collapsed"
        )
    selected_end_date = available_dates[date_number - 1]
    
    # Show a preview table of dates around the selection
    if st.sidebar.checkbox("📋 Show date list", value=False):
        preview_start = max(0, date_number - 6)
        preview_end = min(len(available_dates), date_number + 5)
        
        preview_data = {
            "Number": range(preview_start + 1, preview_end + 1),
            "Date": [d.strftime('%Y-%m-%d') for d in available_dates[preview_start:preview_end]],
            "Selected": ["✓" if i + 1 == date_number else "" for i in range(preview_start, preview_end)]
        }
        st.sidebar.dataframe(pd.DataFrame(preview_data), use_container_width=True)

else:  # Latest Date
    selected_end_date = available_dates[-1]
    st.sidebar.success("✓ Using latest available date")

selected_end_date_str = selected_end_date.strftime('%Y-%m-%d')
st.sidebar.caption(f"**Selected date:** {selected_end_date_str}")
st.sidebar.caption(f"**Will use:** 30 days of data ending on {selected_end_date_str}")

# Lake selection
st.sidebar.markdown("### 🏞️ Lake Selection")
selected_lakes = st.sidebar.multiselect(
    "Select lakes to monitor:",
    config.LAKES,
    default=['Adhala', 'Indravati']
)

# Flood threshold textbox
st.sidebar.markdown("### ⚠️ Flood Threshold")
flood_threshold_level = st.sidebar.slider(
    "Set flood threshold :",
    min_value=0.0,
    max_value=2.0,
    value=2.0,
    step=0.1,
    format="%.1f",
    help="Alert when water level exceeds this threshold"
)

# Get lake capacity for conversion
lake_capacities = config.LAKE_CAPACITIES  # In units
max_capacity = max(lake_capacities.values()) if lake_capacities else 100.0

# Convert absolute water level to percentage for internal calculations
flood_threshold_decimal = (flood_threshold_level / max_capacity) if max_capacity > 0 else 0.9

# Other settings
st.sidebar.markdown("### 🎯 Display Options")
show_confidence = st.sidebar.checkbox("Show Confidence Bands", value=True)

# ===== MAIN CONTENT =====

# Header
st.markdown('<div class="main-header">🌊 Godavari Basin Water Level Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered 7-day forecasting with Graph Neural Networks</div>', unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_assets():
    """Load model and metrics (cached)"""
    # Try to load actual model, fallback to dummy
    model = prediction_engine.load_model()
    if model is None:
        st.warning("⚠️ Trained model not found. Using dummy model for demonstration.")
        model = prediction_engine.create_dummy_model()
    
    metrics = data_loader.load_lake_metrics()
    training_history = data_loader.load_training_history()
    
    return model, metrics, training_history

model, lake_metrics, training_history = load_assets()

# ===== FLOOD WARNING MAP (TOP PRIORITY) =====
st.markdown("## 📍 Flood Warning Map", unsafe_allow_html=True)

# Load real model and make actual predictions using SELECTED DATE
model = prediction_engine.load_model()

# Load 30-day historical sequences for all lakes UP TO THE SELECTED DATE
lake_sequences = data_loader.load_30day_sequence(config.LAKES, config.SEQUENCE_LENGTH, end_date=selected_end_date_str)

# Prepare input tensor for model (converts to (1, 30, 6, 9) tensor)
input_tensor = data_loader.prepare_input_sequence(lake_sequences, config.LAKES, config.SEQUENCE_LENGTH)

# Create edge indices for graph
edge_index_array = data_loader.create_graph_edges()
edge_index = torch.as_tensor(edge_index_array, dtype=torch.long).t().contiguous()

# Get actual predictions from model
try:
    predictions_array = prediction_engine.make_predictions(model, input_tensor, edge_index)
    # predictions_array shape: (num_nodes=6, forecast_horizon=7)
    
    # Convert to dictionary format
    predictions_dict = {}
    for lake_idx, lake_name in enumerate(config.LAKES):
        predictions_dict[lake_name] = predictions_array[lake_idx]
    
except Exception as e:
    st.error(f"Model prediction failed: {str(e)}")
    # Fallback to dummy predictions
    predictions_dict = data_loader.create_dummy_predictions(config.LAKES, config.FORECAST_HORIZON)

# Get latest water levels from the SELECTED DATE
current_levels = {}
for lake_name in config.LAKES:
    df = data_loader.load_lake_data(lake_name)
    if df is not None and len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = df['timestamp'] <= pd.to_datetime(selected_end_date_str)
        df_filtered = df[mask]
        if len(df_filtered) > 0:
            current_levels[lake_name] = df_filtered.iloc[-1]['water_level_m']
        else:
            current_levels[lake_name] = np.random.uniform(1.5, 4.0)
    else:
        current_levels[lake_name] = np.random.uniform(1.5, 4.0)

# Calculate flood status for all lakes
flood_status_dict = {}
for lake_name in config.LAKES:
    current_level = current_levels[lake_name]
    max_capacity = config.LAKE_CAPACITIES[lake_name]
    predictions = predictions_dict[lake_name]
    
    flood_status_dict[lake_name] = flood_warning.get_flood_status_for_lake(
        current_level=current_level,
        max_capacity=max_capacity,
        predictions=predictions,
        threshold=flood_threshold_decimal
    )

# Create and display map
try:
    from streamlit_folium import st_folium
    
    # Add explicit styling for map container
    st.markdown("""
        <style>
            .streamlit-container { max-width: 1400px; }
            .st-ae { width: 100%; }
            iframe { height: 800px !important; }
        </style>
    """, unsafe_allow_html=True)
    
    flood_map = map_generator.create_flood_warning_map(
        lake_data=current_levels,
        flood_status_dict=flood_status_dict
    )
    st_folium(flood_map, width=1400, height=800)
    
except ImportError:
    st.error("Streamlit-Folium not properly installed. Please reinstall dependencies.")

# ===== KEY METRICS =====
st.markdown("## 📊 Key Metrics")

num_flooded = sum(1 for status in flood_status_dict.values() if status['is_flood'])
highest_risk = max(flood_status_dict.items(), 
                   key=lambda x: x[1]['current_percent'])

visualization.plot_metrics_cards(
    num_flooded=num_flooded,
    total_lakes=len(config.LAKES),
    highest_risk_lake=highest_risk[0],
    highest_risk_percent=highest_risk[1]['current_percent']
)

# ===== FLOOD STATUS TABLE =====
st.markdown("## ⚠️ Flood Status")
visualization.plot_lake_status_table(flood_status_dict, lake_metrics)

# ===== FORECAST CHARTS (FOR SELECTED LAKES) =====
st.markdown("## 📈 7-Day Forecast")

# Load real historical data from CSV for the SELECTED DATE (30 days back)
historical_data = {}
historical_timestamps = {}
for lake_name in config.LAKES:
    # Get 30-day historical data ending at selected_end_date
    start_date_obj = selected_end_date - timedelta(days=29)  # 30 days including end date
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    timestamps, levels = data_loader.load_historical_data_for_date_range(
        lake_name, 
        start_date=start_date_str, 
        end_date=selected_end_date_str
    )
    historical_timestamps[lake_name] = timestamps
    historical_data[lake_name] = levels

# Create tabs for each selected lake
if selected_lakes:
    tabs = st.tabs([f"📊 {lake}" for lake in selected_lakes])
    
    for tab, lake_name in zip(tabs, selected_lakes):
        with tab:
            # Get forecast data
            predictions = predictions_dict[lake_name]
            current_level = current_levels[lake_name]
            
            # Get historical data for this lake
            hist_data = historical_data.get(lake_name, [])
            hist_timestamps = historical_timestamps.get(lake_name, [])
            
            # If historical data is empty, show warning but continue
            if not hist_data:
                st.warning(f"No historical data available for {lake_name} on {selected_end_date_str}")
                # Create synthetic data with proper date strings (not integers)
                hist_data = np.linspace(2.0, current_level, 30)
                start_date_for_fallback = selected_end_date - timedelta(days=29)
                hist_timestamps = [(start_date_for_fallback + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
            
            # Calculate confidence bands (simple: ±10%)
            confidence_upper = predictions * 1.1
            confidence_lower = predictions * 0.9
            
            # Plot with real historical data from selected date range
            fig = visualization.plot_forecast_chart(
                lake_name=lake_name,
                historical_data=hist_data,
                predictions=predictions,
                historical_dates=hist_timestamps,
                confidence_bands=(confidence_upper.tolist(), confidence_lower.tolist()) if show_confidence else None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Status info
            status = flood_status_dict[lake_name]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", status['status_text'], delta=None)
            with col2:
                if status['days_to_flood']:
                    st.metric("Days to Flood", f"{status['days_to_flood']} days")
                else:
                    st.metric("Days to Flood", "Safe")
else:
    st.info("👈 Select lakes from the sidebar to view detailed forecasts")

# ===== FOOTER =====
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔧 Model: Spatio-Temporal GNN (GAT + LSTM + Attention)")
with col2:
    st.caption(f"📅 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col3:
    st.caption("✅ Status: Ready")
