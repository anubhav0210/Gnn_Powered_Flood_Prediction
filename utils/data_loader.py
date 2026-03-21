"""
Data loading utilities for GNN model inference
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import config
import json

# Paths to real processed data
PROCESSED_DATA_DIR = Path(__file__).parent.parent / 'proccessed _data' / 'processed'
DATASET_INFO_PATH = PROCESSED_DATA_DIR / 'dataset_info.json'

def load_lake_data(lake_name):
    """Load processed data for a specific lake."""
    lake_file = PROCESSED_DATA_DIR / f'{lake_name.lower()}_cleaned.csv'
    if lake_file.exists():
        return pd.read_csv(lake_file)
    return None

def load_dataset_info():
    """Load dataset information."""
    if DATASET_INFO_PATH.exists():
        with open(DATASET_INFO_PATH, 'r') as f:
            return json.load(f)
    return {"sequence_length": 30, "forecast_horizon": 7}

def get_latest_water_level(lake_name):
    """Get the latest water level for a lake."""
    df = load_lake_data(lake_name)
    if df is not None and len(df) > 0:
        # Get the latest denormalized water level
        latest = df.iloc[-1]
        # The water_level_m is already denormalized in the CSV
        return latest['water_level_m']
    return np.random.uniform(1.5, 4.0)  # Fallback


def load_lake_metrics():
    """
    Load lake performance metrics from CSV.
    
    Returns:
        pd.DataFrame: Lake metrics with R² scores, RMSE, etc.
    """
    try:
        metrics_df = pd.read_csv(config.METRICS_PATH, index_col=0)
        # Reset index to make lake names a column
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'Lake'}, inplace=True)
        # Capitalize lake names to match config
        metrics_df['Lake'] = metrics_df['Lake'].str.capitalize()
        return metrics_df
    except FileNotFoundError:
        print(f"Warning: Metrics file not found at {config.METRICS_PATH}")
        return create_dummy_metrics()


def create_dummy_metrics():
    """
    Create dummy metrics for testing (when actual data not available).
    
    Returns:
        pd.DataFrame: Dummy metrics
    """
    return pd.DataFrame({
        'Lake': [lake.capitalize() for lake in config.LAKES],
        'mse': [0.338, 0.164, 0.131, 1.179, 0.303, 0.216],
        'mae': [0.278, 0.054, 0.215, 0.212, 0.235, 0.234],
        'rmse': [0.581, 0.405, 0.362, 1.086, 0.550, 0.465],
        'r2': [0.761, -0.001, 0.873, 0.018, 0.736, 0.819]
    })


def load_training_history():
    """
    Load training history from CSV.
    
    Returns:
        pd.DataFrame: Training metrics over epochs
    """
    try:
        history_df = pd.read_csv(config.TRAINING_HISTORY_PATH)
        return history_df
    except FileNotFoundError:
        print(f"Warning: Training history file not found at {config.TRAINING_HISTORY_PATH}")
        return None


def create_dummy_predictions(lake_names, forecast_horizon=7):
    """
    Create dummy predictions for testing.
    
    Args:
        lake_names (list): List of lake names
        forecast_horizon (int): Number of days to forecast
    
    Returns:
        dict: Dummy predictions for each lake
    """
    predictions = {}
    
    for lake in lake_names:
        # Random predictions between 1 and 4 meters
        base_level = np.random.uniform(1.5, 3.5)
        trend = np.random.uniform(-0.1, 0.1)
        noise = np.random.normal(0, 0.05, forecast_horizon)
        
        pred = [base_level + trend * (i + 1) + noise[i] for i in range(forecast_horizon)]
        predictions[lake] = np.array(pred)
    
    return predictions


def create_dummy_current_levels(lake_names):
    """
    Create dummy current water levels for testing.
    
    Args:
        lake_names (list): List of lake names
    
    Returns:
        dict: Current water level for each lake (in meters)
    """
    return {
        'Adhala': 2.85,
        'Girija': 2.10,
        'Indravati': 3.25,
        'Manjira': 2.60,
        'Valamuru': 2.40,
        'Sabari': 2.75
    }


def normalize_data(data, scaler=None):
    """
    Normalize data using provided scaler or create new one.
    
    Args:
        data (np.array): Data to normalize
        scaler: Pre-fitted StandardScaler or None to fit new one
    
    Returns:
        tuple: (normalized_data, scaler)
    """
    from sklearn.preprocessing import StandardScaler
    
    if scaler is None:
        scaler = StandardScaler()
        normalized = scaler.fit_transform(data)
    else:
        normalized = scaler.transform(data)
    
    return normalized, scaler


def denormalize_data(normalized_data, scaler):
    """
    Denormalize data using fitted scaler.
    
    Args:
        normalized_data (np.array): Normalized data
        scaler: Pre-fitted StandardScaler
    
    Returns:
        np.array: Denormalized data
    """
    return scaler.inverse_transform(normalized_data)


def load_30day_sequence(lake_names, sequence_length=30, end_date=None):
    """
    Load 30-day historical sequences from processed CSV files for all lakes.
    
    Args:
        lake_names (list): List of lake names to load
        sequence_length (int): Length of historical sequence to load (default 30 days)
        end_date (str or datetime): End date for the sequence (default: last date in data)
                                    Format: 'YYYY-MM-DD' or datetime object
    
    Returns:
        dict: Dictionary mapping lake_name -> DataFrame with 30 days of data ending at end_date
    """
    sequences = {}
    
    # Convert end_date to datetime if string
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    for lake_name in lake_names:
        df = load_lake_data(lake_name)
        if df is not None and len(df) > 0:
            # Ensure timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if end_date is None:
                # Use last date in data
                end_date_to_use = df['timestamp'].max()
            else:
                end_date_to_use = end_date
            
            # Find rows up to end_date
            mask = df['timestamp'] <= end_date_to_use
            df_up_to_date = df[mask]
            
            if len(df_up_to_date) >= sequence_length:
                # Get last sequence_length rows from filtered data
                sequences[lake_name] = df_up_to_date.iloc[-sequence_length:].reset_index(drop=True)
            else:
                print(f"Warning: Not enough data for {lake_name} up to {end_date_to_use}")
                sequences[lake_name] = None
        else:
            sequences[lake_name] = None
    
    return sequences


def prepare_input_sequence(lake_sequences, lake_names, sequence_length=30):
    """
    Prepare input sequence for GNN model using real data from CSV.
    
    This function converts 30 days of historical data from all 6 lakes into a single
    tensor that the Graph Neural Network expects as input.
    
    DATA FLOW EXPLANATION:
    =====================
    Input: 
    - lake_sequences: dict with 6 DataFrames, each 30 rows × 16 columns
      Example: {
        'Adhala': DataFrame(30 rows, columns=[timestamp, rainfall_mm, humidity_pct, ...]),
        'Girija': DataFrame(30 rows, ...),
        ... (4 more lakes)
      }
    
    Output Tensor Shape: (1, 30, 6, 9)
    - Dimension 1 (batch_size): = 1 (always 1 for inference, we predict once at a time)
    - Dimension 2 (sequence_length): = 30 (days of historical data)
    - Dimension 3 (num_nodes): = 6 (Adhala, Girija, Indravati, Manjira, Valamuru, Sabari)
    - Dimension 4 (num_features): = 9 (selected features per lake)
    
    The 9 extracted features per lake per day are:
    1. rainfall_mm - Rainfall in millimeters (normalized)
    2. humidity_pct - Humidity percentage (normalized)
    3. water_level_m - Water level in meters (normalized)
    4. discharge_m3s - Discharge in cubic meters per second (normalized)
    5. day_of_year - Day number in year (1-366, normalized)
    6. month - Month number (1-12, normalized)
    7. year - Year value (normalized)
    8. day_sin - Sine of day of year for cyclical encoding
    9. day_cos - Cosine of day of year for cyclical encoding
    
    Note: ALL values are already NORMALIZED in the CSV files (standardized to mean≈0, std≈1)
    
    Tensor Memory Layout:
    tensor[0, day_idx, lake_idx, feature_idx] = value for that day/lake/feature
    
    Example:
    tensor[0, 0, 0, 0] = rainfall on day 1 at Adhala lake
    tensor[0, 1, 2, 3] = discharge on day 2 at Indravati lake
    tensor[0, 29, 5, 2] = water level on day 30 at Sabari lake
    
    Args:
        lake_sequences (dict): Dictionary of lake_name -> DataFrame with historical data
        lake_names (list): List of lake names in order [Adhala, Girija, Indravati, Manjira, Valamuru, Sabari]
        sequence_length (int): Length of historical sequence (default 30 days)
    
    Returns:
        torch.Tensor: Input tensor for model with shape (1, sequence_length, num_nodes, num_features)
    """
    batch_size = 1
    num_nodes = len(lake_names)
    num_features = config.NUM_FEATURES  # Should be 9
    
    # Initialize sequence array with zeros: shape (1, 30, 6, 9)
    sequence = np.zeros((batch_size, sequence_length, num_nodes, num_features), dtype=np.float32)
    
    # Feature columns to extract from CSV (in order)
    # These 9 features are what the model was trained on
    feature_cols = [
        'rainfall_mm',          # Feature 0
        'humidity_pct',         # Feature 1
        'water_level_m',        # Feature 2
        'discharge_m3s',        # Feature 3
        'day_of_year',          # Feature 4
        'month',                # Feature 5
        'year',                 # Feature 6
        'day_sin',              # Feature 7
        'day_cos',              # Feature 8
    ]
    # Verify we have exactly num_features
    assert len(feature_cols) == num_features, f"Expected {num_features} features, got {len(feature_cols)}"
    
    # Fill in data for each lake
    for node_idx, lake_name in enumerate(lake_names):
        df = lake_sequences.get(lake_name)
        if df is not None and len(df) >= sequence_length:
            # Extract the required feature columns for this lake
            # Returns shape: (30, 9) for 30 days × 9 features
            data_slice = df[feature_cols].tail(sequence_length).values
            # Assign to tensor: sequence[batch=0, time=0..29, node=lake_idx, features=0..8]
            sequence[0, :, node_idx, :] = data_slice
        else:
            # Fallback: use zeros if insufficient data
            # This means the model gets all zeros for this lake, which may affect predictions
            sequence[0, :, node_idx, :] = 0.0
    
    # Convert numpy array to PyTorch tensor
    return torch.from_numpy(sequence)


def get_available_dates(lake_name=None):
    """
    Get all available dates in the processed data.
    
    Args:
        lake_name (str): Specific lake to check dates for (default: use first lake)
    
    Returns:
        list: List of datetime objects representing available dates
    """
    if lake_name is None:
        lake_name = config.LAKES[0]  # Use first lake as reference
    
    df = load_lake_data(lake_name)
    if df is not None and len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return sorted(df['timestamp'].unique().tolist())
    return []


def get_date_range_info():
    """
    Get the date range available in the dataset.
    
    Returns:
        dict: Contains 'start_date', 'end_date', and 'num_days'
    """
    dates = get_available_dates()
    if len(dates) > 0:
        return {
            'start_date': dates[0],
            'end_date': dates[-1],
            'num_days': len(dates)
        }
    return {'start_date': None, 'end_date': None, 'num_days': 0}


def load_historical_data_for_date_range(lake_name, start_date, end_date):
    """
    Load historical water level data for a specific date range.
    Useful for visualization/comparison.
    
    Args:
        lake_name (str): Lake name
        start_date (str or datetime): Start date (format: 'YYYY-MM-DD')
        end_date (str or datetime): End date (format: 'YYYY-MM-DD')
    
    Returns:
        tuple: (timestamps, water_levels) lists
    """
    df = load_lake_data(lake_name)
    if df is None or len(df) == 0:
        return [], []
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        return [], []
    
    timestamps = filtered_df['timestamp'].dt.strftime('%Y-%m-%d').tolist()
    water_levels = filtered_df['water_level_m'].tolist()
    
    return timestamps, water_levels





def create_graph_edges(num_nodes=6):
    """
    Create graph edges for Godavari basin (predefined water flow).
    
    Args:
        num_nodes (int): Number of nodes (lakes)
    
    Returns:
        torch.Tensor: Edge indices for PyG graph
    """
    # Godavari basin structure:
    # Upper (0,1) -> Middle (2,3) -> Lower (4,5)
    edges = [
        (0, 1), (1, 0),  # Adhala <-> Girija
        (0, 2), (2, 0),  # Adhala -> Indravati
        (1, 3), (3, 1),  # Girija -> Manjira
        (2, 4), (4, 2),  # Indravati -> Valamuru
        (3, 5), (5, 3),  # Manjira -> Sabari
        (4, 5), (5, 4),  # Valamuru <-> Sabari
    ]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
