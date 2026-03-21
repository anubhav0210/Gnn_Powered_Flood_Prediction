"""
Configuration settings for Godavari GNN Water Level Predictor
"""

import torch

# Lakes in the basin
LAKES = ['Adhala', 'Girija', 'Indravati', 'Manjira', 'Valamuru', 'Sabari']

# Model parameters
SEQUENCE_LENGTH = 30  # Historical days for input
FORECAST_HORIZON = 7  # Days to forecast
NUM_FEATURES = 9  # rainfall, humidity, water_level, discharge, day_of_year, month, year, day_sin, day_cos + more
NUM_NODES = 6  # Number of lakes

# File paths
MODEL_PATH = 'models/final_gnn_model.pth'
METRICS_PATH = 'data/lake_metrics.csv'
TRAINING_HISTORY_PATH = 'data/training_history.csv'

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Lake coordinates for mapping (latitude, longitude)
LAKE_COORDINATES = {
    'Adhala': (19.2183, 77.7499),
    'Girija': (19.3500, 77.8500),
    'Indravati': (19.8500, 82.0500),
    'Manjira': (19.0500, 77.8500),
    'Valamuru': (18.4000, 79.1500),
    'Sabari': (17.6000, 82.5000)
}

# Default flood threshold (% of capacity)
DEFAULT_FLOOD_THRESHOLD = 0.90  # 90%

# Lake maximum capacities (in meter³ - example values)
LAKE_CAPACITIES = {
    'Adhala': 1500e6,
    'Girija': 2000e6,
    'Indravati': 5000e6,
    'Manjira': 2800e6,
    'Valamuru': 1200e6,
    'Sabari': 3200e6
}

# Colors for lakes
LAKE_COLORS = {
    'Adhala': '#1f77b4',        # Blue
    'Girija': '#ff7f0e',        # Orange
    'Indravati': '#2ca02c',     # Green
    'Manjira': '#d62728',       # Red
    'Valamuru': '#9467bd',      # Purple
    'Sabari': '#8c564b'         # Brown
}

# Flood colors
FLOOD_COLOR = '#d62728'  # Red for flood
SAFE_COLOR = '#2ca02c'   # Green for safe

# Map settings
MAP_CENTER = [19.5, 79.5]  # Center of Godavari Basin
MAP_ZOOM_START = 8

# UI Settings
PAGE_TITLE = "🌊 Godavari Basin Water Level Predictor"
PAGE_ICON = "🌊"
LAYOUT = "wide"

# Model architecture parameters
MODEL_HIDDEN_DIM = 128
MODEL_NUM_LAYERS = 3
MODEL_DROPOUT = 0.3
MODEL_NUM_HEADS = 8
