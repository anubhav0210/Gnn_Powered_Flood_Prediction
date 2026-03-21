"""
Visualization components for the Streamlit app
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import config


def plot_forecast_chart(lake_name, historical_data, predictions, historical_dates=None, confidence_bands=None):
    """
    Create interactive forecast chart using Plotly.
    
    Args:
        lake_name (str): Name of the lake
        historical_data (list or np.array): Historical water levels (last 30 days)
        predictions (list or np.array): Predicted water levels (next 7 days)
        historical_dates (list): Dates for historical data (if None, uses day numbers)
        confidence_bands (tuple): (upper, lower) confidence bands
    
    Returns:
        plotly.graph_objects.Figure: Interactive chart
    """
    # Create date range using provided dates or fall back to day numbers
    if historical_dates is None or len(historical_dates) == 0:
        # Fallback: use day numbers if no dates provided
        historical_dates = [f"Day {i-len(historical_data)+1}" for i in range(len(historical_data))]
        forecast_dates = [f"Day {i+1}" for i in range(len(predictions))]
    else:
        # Convert to datetime if they're strings
        try:
            hist_datetime = []
            for d in historical_dates:
                # Skip if it's an integer (fallback to day numbers)
                if isinstance(d, int):
                    raise ValueError(f"Got integer timestamp {d}, expected string or datetime")
                if isinstance(d, str):
                    hist_datetime.append(pd.to_datetime(d))
                elif hasattr(d, 'strftime'):  # datetime object
                    hist_datetime.append(d)
                else:
                    raise ValueError(f"Unsupported date type: {type(d)}")
            
            # Get the last historical date and add days for forecast
            last_date = hist_datetime[-1]
            forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
            historical_dates = [d.strftime('%Y-%m-%d') for d in hist_datetime]
        except Exception as e:
            # If date conversion fails, use day numbers
            print(f"Warning: Could not parse dates: {e}")
            historical_dates = [f"Day {i-len(historical_data)+1}" for i in range(len(historical_data))]
            forecast_dates = [f"Day {i+1}" for i in range(len(predictions))]
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_data,
        mode='lines',
        name='Historical',
        line=dict(color='#7f7f7f', width=2),
        hovertemplate='<b>Historical</b><br>Date: %{x}<br>Level: %{y:.2f}m<extra></extra>'
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color=config.LAKE_COLORS.get(lake_name, '#1f77b4'), width=3),
        marker=dict(size=8),
        hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Level: %{y:.2f}m<extra></extra>'
    ))
    
    # Add confidence bands if provided
    if confidence_bands is not None:
        upper, lower = confidence_bands
        fig.add_trace(go.Scatter(
            x=forecast_dates + forecast_dates[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 100, 200, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            showlegend=True,
            name='Confidence Band',
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"7-Day Water Level Forecast: {lake_name}",
        xaxis_title="Date",
        yaxis_title="Water Level (meters)",
        hovermode='x unified',
        height=400,
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def plot_metrics_cards(num_flooded, total_lakes, highest_risk_lake, highest_risk_percent):
    """
    Display KPI metric cards.
    
    Args:
        num_flooded (int): Number of lakes in flood state
        total_lakes (int): Total number of lakes
        highest_risk_lake (str): Lake name with highest risk
        highest_risk_percent (float): Highest risk percentage
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "In Flood",
            f"{num_flooded}/{total_lakes} Lakes",
            delta=None,
            help="Number of lakes currently above flood threshold"
        )
    
    with col2:
        safe_count = total_lakes - num_flooded
        st.metric(
            "Safe",
            f"{safe_count}/{total_lakes} Lakes",
            delta=None,
            help="Number of lakes currently below flood threshold"
        )
    
    with col3:
        st.metric(
            "Highest Risk",
            f"{highest_risk_lake}",
            f"{highest_risk_percent:.1f}%",
            help="Lake with highest water level relative to capacity"
        )


def plot_lake_status_table(flood_status_dict, lake_metrics):
    """
    Display table with flood status for all lakes.
    
    Args:
        flood_status_dict (dict): Flood status for each lake
        lake_metrics (pd.DataFrame): Lake performance metrics
    """
    data = []
    
    for lake_name in config.LAKES:
        status = flood_status_dict.get(lake_name, {})
        
        # Get metrics - handle both old and new column names
        lake_metric = lake_metrics[lake_metrics['Lake'].str.lower() == lake_name.lower()]
        if len(lake_metric) > 0:
            # Use 'r2' column from actual CSV (lowercase)
            r2_score = lake_metric['r2'].values[0]
        else:
            r2_score = 'N/A'
        
        data.append({
            'Lake': lake_name,
            'Status': status.get('status_text', '🟢 SAFE'),
            'Current Level': f"{status.get('current_percent', 0):.1f}%",
            'Threshold': f"{status.get('threshold_percent', 90):.1f}%",
            'Days to Flood': status.get('days_to_flood', '-') if status.get('days_to_flood') else '-',
            'Model R²': f"{r2_score:.3f}" if isinstance(r2_score, (int, float)) else r2_score
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def plot_model_performance_table(lake_metrics):
    """
    Display model performance metrics table.
    
    Args:
        lake_metrics (pd.DataFrame): Lake performance metrics
    """
    # Create a clean display version with proper column names
    display_metrics = lake_metrics[['Lake', 'r2', 'rmse', 'mae']].copy()
    display_metrics.columns = ['Lake', 'R² Score', 'RMSE', 'MAE']
    display_metrics['R² Score'] = display_metrics['R² Score'].apply(lambda x: f"{x:.4f}")
    display_metrics['RMSE'] = display_metrics['RMSE'].apply(lambda x: f"{x:.4f}")
    display_metrics['MAE'] = display_metrics['MAE'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)


def plot_training_history(training_history):
    """
    Plot training and validation loss over epochs.
    
    Args:
        training_history (pd.DataFrame): Training history data
    """
    if training_history is None or training_history.empty:
        st.info("Training history not available")
        return
    
    fig = go.Figure()
    
    # Add training loss
    fig.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['train_loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='#1f77b4')
    ))
    
    # Add validation loss
    fig.add_trace(go.Scatter(
        x=training_history['epoch'],
        y=training_history['val_loss'],
        mode='lines',
        name='Val Loss',
        line=dict(color='#ff7f0e')
    ))
    
    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
