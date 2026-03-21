"""
Water Level Prediction Script
Predicts water levels for a user-selected date using the trained GNN model
and plots historical + predicted data with option to save plots
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import app modules
APP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(APP_ROOT))

# Change working directory to app root so relative paths work
os.chdir(APP_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime as dt

import config
from utils import data_loader, prediction_engine, visualization


def display_available_dates():
    """Display all available dates in the dataset."""
    dates = data_loader.get_available_dates()
    date_range = data_loader.get_date_range_info()
    
    print("\n" + "="*70)
    print("AVAILABLE DATA")
    print("="*70)
    print(f"Total days available: {date_range['num_days']}")
    print(f"Date range: {date_range['start_date'].strftime('%Y-%m-%d')} to {date_range['end_date'].strftime('%Y-%m-%d')}")
    print(f"\nFirst 10 dates:")
    for i, date in enumerate(dates[:10], 1):
        print(f"  {i:2}. {date.strftime('%Y-%m-%d')}")
    print(f"...")
    print(f"Last 10 dates:")
    for i, date in enumerate(dates[-10:], len(dates)-9):
        print(f"  {i:2}. {date.strftime('%Y-%m-%d')}")
    print()
    
    return dates


def get_user_date_selection(available_dates):
    """Get date selection from user via command line."""
    print("="*70)
    print("SELECT DATE FOR PREDICTION")
    print("="*70)
    print("\nOptions:")
    print("  1. Enter specific date (YYYY-MM-DD)")
    print("  2. Use latest available date")
    print("  3. Enter date number from available dates")
    
    while True:
        choice = input("\nSelect option (1/2/3): ").strip()
        
        if choice == "1":
            # User enters specific date
            while True:
                date_str = input("Enter date (YYYY-MM-DD): ").strip()
                try:
                    selected_date = pd.to_datetime(date_str)
                    # Check if date is in dataset
                    if selected_date in available_dates:
                        return selected_date
                    else:
                        print(f"✗ Date {date_str} not in dataset. Please try another date.")
                        print(f"  Available range: {available_dates[0].strftime('%Y-%m-%d')} to {available_dates[-1].strftime('%Y-%m-%d')}")
                except Exception as e:
                    print(f"✗ Invalid date format: {e}")
        
        elif choice == "2":
            # Use latest date
            selected_date = available_dates[-1]
            print(f"✓ Selected latest date: {selected_date.strftime('%Y-%m-%d')}")
            return selected_date
        
        elif choice == "3":
            # User enters date number
            try:
                idx = int(input(f"Enter date number (1-{len(available_dates)}): ").strip())
                if 1 <= idx <= len(available_dates):
                    selected_date = available_dates[idx - 1]
                    print(f"✓ Selected date: {selected_date.strftime('%Y-%m-%d')}")
                    return selected_date
                else:
                    print(f"✗ Number must be between 1 and {len(available_dates)}")
            except ValueError:
                print("✗ Invalid number")
        else:
            print("✗ Invalid option. Choose 1, 2, or 3.")


def load_and_predict(selected_date):
    """Load data and make predictions for selected date."""
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    
    print("\n" + "="*70)
    print(f"LOADING DATA FOR {selected_date_str}")
    print("="*70)
    
    # Load 30-day sequences for all lakes
    print("Loading 30-day historical sequences...")
    lake_sequences = data_loader.load_30day_sequence(
        config.LAKES, 
        config.SEQUENCE_LENGTH, 
        end_date=selected_date_str
    )
    
    # Check if data loaded successfully
    lakes_loaded = sum(1 for s in lake_sequences.values() if s is not None)
    print(f"✓ Loaded data for {lakes_loaded}/{len(config.LAKES)} lakes")
    
    # Prepare input tensor
    print("Preparing input tensor (1, 30, 6, 9)...")
    input_tensor = data_loader.prepare_input_sequence(
        lake_sequences, 
        config.LAKES, 
        config.SEQUENCE_LENGTH
    )
    print(f"✓ Tensor shape: {input_tensor.shape}")
    
    # Create edge indices
    print("Creating graph structure...")
    edge_index_array = data_loader.create_graph_edges()
    edge_index = torch.as_tensor(edge_index_array, dtype=torch.long).t().contiguous()
    print(f"✓ Graph edges: {edge_index.shape}")
    
    # Load model
    print("Loading trained GNN model...")
    model = prediction_engine.load_model()
    print("✓ Model loaded")
    
    # Make predictions
    print("Running inference...")
    try:
        predictions_array = prediction_engine.make_predictions(
            model, 
            input_tensor, 
            edge_index
        )
        print(f"✓ Predictions generated: shape {predictions_array.shape}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return None, None, None
    
    # Convert to dictionary
    predictions_dict = {
        lake: predictions_array[idx] 
        for idx, lake in enumerate(config.LAKES)
    }
    
    # Get current water levels (at selected date)
    print("Extracting current water levels...")
    current_levels = {}
    for lake_name in config.LAKES:
        df = data_loader.load_lake_data(lake_name)
        if df is not None and len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = df['timestamp'] <= selected_date
            df_filtered = df[mask]
            if len(df_filtered) > 0:
                current_levels[lake_name] = df_filtered.iloc[-1]['water_level_m']
            else:
                current_levels[lake_name] = np.nan
        else:
            current_levels[lake_name] = np.nan
    
    print("✓ Current water levels retrieved")
    
    # Get historical data for charts
    print("Loading historical data for visualization...")
    historical_data = {}
    historical_timestamps = {}
    
    start_date_obj = selected_date - timedelta(days=29)
    start_date_str = start_date_obj.strftime('%Y-%m-%d')
    
    for lake_name in config.LAKES:
        timestamps, levels = data_loader.load_historical_data_for_date_range(
            lake_name,
            start_date=start_date_str,
            end_date=selected_date_str
        )
        historical_timestamps[lake_name] = timestamps
        historical_data[lake_name] = levels
    
    print("✓ Historical data loaded")
    
    return predictions_dict, current_levels, (historical_data, historical_timestamps)


def get_actual_forecast_data(lake_name, forecast_start_date, num_days=7):
    """Check if actual data exists for the forecast period and return it.
    
    Args:
        lake_name: Name of the lake
        forecast_start_date: Date when forecast starts (first day after selected date)
        num_days: Number of days to fetch (default 7)
    
    Returns:
        Dict with 'dates' and 'levels' if data exists, else empty dict
    """
    try:
        df = data_loader.load_lake_data(lake_name)
        if df is None or len(df) == 0:
            return {}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get data for forecast period
        forecast_end_date = forecast_start_date + timedelta(days=num_days-1)
        
        mask = (df['timestamp'] >= forecast_start_date) & (df['timestamp'] <= forecast_end_date)
        df_forecast = df[mask].sort_values('timestamp')
        
        if len(df_forecast) == 0:
            return {}
        
        return {
            'dates': df_forecast['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
            'levels': df_forecast['water_level_m'].values.tolist()
        }
    
    except Exception as e:
        print(f"  ⚠ Could not fetch actual forecast data: {e}")
        return {}


def plot_predictions(lake_name, historical_data, predictions, 
                     historical_dates, current_level, save_dir=None, actual_forecast_data=None):
    """Create a plot for a lake showing historical + predicted water levels.
    
    Args:
        actual_forecast_data: Dict with keys 'dates' and 'levels' for actual data within forecast range
    """
    
    # Denormalize predictions and current level
    # Note: Using approximate scaling factors from data
    pred_mean, pred_std = 0, 1  # Normalized data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot historical data
    if historical_data and len(historical_data) > 0:
        ax.plot(
            range(len(historical_data)),
            historical_data,
            'o-',
            linewidth=2,
            markersize=6,
            label='Historical Data',
            color='#2ca02c'
        )
    
    # Plot current level
    if not np.isnan(current_level):
        ax.axhline(
            y=current_level,
            color='#ff7f0e',
            linestyle='--',
            linewidth=2,
            label=f'Current Level: {current_level:.3f}m',
            alpha=0.7
        )
    
    # Plot predictions and/or actual data
    hist_len = len(historical_data) if historical_data else 30
    pred_indices = range(hist_len, hist_len + len(predictions))
    
    # Check if we have actual data for the forecast period
    has_actual_data = actual_forecast_data and len(actual_forecast_data.get('levels', [])) > 0
    
    if has_actual_data:
        # Plot predictions first
        ax.plot(
            pred_indices,
            predictions,
            'D--',
            linewidth=2.5,
            markersize=8,
            label='Model Prediction (7-day)',
            color='#d62728',
            zorder=2
        )
        
        # Add confidence band (±10%)
        pred_upper = predictions * 1.1
        pred_lower = predictions * 0.9
        ax.fill_between(
            pred_indices,
            pred_lower,
            pred_upper,
            alpha=0.15,
            color='#d62728',
            label='Confidence Band (±10%)',
            zorder=1
        )
        
        # Plot actual data on top for comparison
        actual_levels = actual_forecast_data['levels']
        actual_dates = actual_forecast_data['dates']
        
        # Map actual data to corresponding prediction days
        actual_x_indices = range(hist_len, hist_len + len(actual_levels))
        
        ax.plot(
            actual_x_indices,
            actual_levels,
            'o-',
            linewidth=2.5,
            markersize=8,
            label=f'Actual Data ({len(actual_levels)} days available)',
            color='#1f77b4',
            zorder=3
        )
    else:
        # Only plot predictions (no actual data available)
        ax.plot(
            pred_indices,
            predictions,
            's-',
            linewidth=2,
            markersize=8,
            label='7-Day Forecast',
            color='#d62728'
        )
        
        # Add confidence band (±10%)
        pred_upper = predictions * 1.1
        pred_lower = predictions * 0.9
        ax.fill_between(
            pred_indices,
            pred_lower,
            pred_upper,
            alpha=0.2,
            color='#d62728',
            label='Confidence Band (±10%)'
        )
    
    # Formatting
    ax.set_xlabel('Days', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Level (meters)', fontsize=12, fontweight='bold')
    title_suffix = " (with Actual Data Validation)" if has_actual_data else ""
    ax.set_title(f'{lake_name} - Water Level: Historical + 7-Day Forecast{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # X-axis labels
    all_labels = []
    if historical_dates and len(historical_dates) > 0:
        # Use actual dates for historical
        for i, date_str in enumerate(historical_dates):
            if i % 5 == 0:  # Show every 5th date to avoid crowding
                all_labels.append(date_str)
            else:
                all_labels.append('')
    
    # Add forecast/actual dates
    if historical_dates and len(historical_dates) > 0:
        last_date = pd.to_datetime(historical_dates[-1])
        for i in range(len(predictions)):
            forecast_date = (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            all_labels.append(forecast_date)
    
    if all_labels:
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        save_path = Path(save_dir) / f"{lake_name}_prediction.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    return fig


def save_predictions_to_csv(selected_date, predictions_dict, current_levels, save_dir):
    """Save predictions to CSV file."""
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    
    # Prepare data
    data = {'Lake': config.LAKES}
    data['Current Level (m)'] = [current_levels.get(lake, np.nan) for lake in config.LAKES]
    
    for day in range(1, 8):
        data[f'Day +{day}'] = [predictions_dict[lake][day-1] for lake in config.LAKES]
    
    df = pd.DataFrame(data)
    
    # Save
    save_path = Path(save_dir) / f"predictions_{selected_date_str}.csv"
    df.to_csv(save_path, index=False)
    print(f"✓ Saved predictions to: {save_path}")
    
    # Display
    print("\nPredictions Summary:")
    print(df.to_string(index=False))
    
    return df


def main():
    """Main function."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  GODAVARI BASIN - WATER LEVEL PREDICTION SCRIPT".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # Display available dates
    available_dates = display_available_dates()
    
    # Get user date selection
    selected_date = get_user_date_selection(available_dates)
    
    # Load data and make predictions
    predictions_dict, current_levels, historical_info = load_and_predict(selected_date)
    
    if predictions_dict is None:
        print("\n✗ Prediction failed. Exiting.")
        return
    
    historical_data, historical_timestamps = historical_info
    selected_date_str = selected_date.strftime('%Y-%m-%d')
    
    # Create output directory
    output_dir = Path(__file__).parent / f"results_{selected_date_str}"
    output_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir}")
    
    # Plot predictions for each lake
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    for lake_idx, lake_name in enumerate(config.LAKES):
        print(f"\nPlotting {lake_name}...")
        
        hist_data = historical_data.get(lake_name, [])
        hist_dates = historical_timestamps.get(lake_name, [])
        predictions = predictions_dict[lake_name]
        current_level = current_levels.get(lake_name, np.nan)
        
        # Try to get actual data for forecast period
        forecast_start_date = selected_date + timedelta(days=1)
        actual_forecast_data = get_actual_forecast_data(lake_name, forecast_start_date, num_days=7)
        
        if actual_forecast_data:
            print(f"  ℹ Found actual data for {len(actual_forecast_data['dates'])} of 7 forecast days")
        
        fig = plot_predictions(
            lake_name,
            hist_data,
            predictions,
            hist_dates,
            current_level,
            save_dir=output_dir,
            actual_forecast_data=actual_forecast_data
        )
        plt.close(fig)
    
    # Save predictions to CSV
    print("\n" + "="*70)
    print("SAVING PREDICTIONS")
    print("="*70)
    print()
    save_predictions_to_csv(selected_date, predictions_dict, current_levels, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Selected date: {selected_date_str}")
    print(f"Lakes predicted: {len(config.LAKES)}")
    print(f"Forecast horizon: {len(predictions_dict[config.LAKES[0]])} days")
    print(f"Output directory: {output_dir}")
    print(f"Files saved: {len(list(output_dir.glob('*')))}")
    print("\n✓ Prediction complete!")
    print(f"\nYou can view the results in: {output_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Cancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
