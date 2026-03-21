# 🌊 Godavari Basin Water Level Predictor

**AI-powered 7-day water level forecasting with Graph Neural Networks & Flood Warning System**

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)](http://localhost:8501)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)

---

## 🎯 Current Status

### ✅ What's Working Now (MVP - Phase 1)
- ✅ Interactive Streamlit web app deployed locally
- ✅ Flood warning map with color-coded markers (red/green)
- ✅ User-adjustable flood threshold (water level units textbox)
- ✅ 7-day forecast charts for selected lakes with real data comparison
- ✅ Interactive date selection (Slider, Date Picker, Date Number, Latest)
- ✅ Flood status dashboard with KPI cards
- ✅ Multi-lake selection with full 1035-day dataset
- ✅ Professional UI with responsive design

### 🔄 Demo Mode (Phase 1)
The app uses **realistic demo predictions** to showcase functionality:
- Water levels generated from your actual data distribution
- Predictions follow realistic patterns from your model's performance metrics
- All UI components fully functional and interactive
- Perfect for testing, demos, and validation

### 📋 Phase 2: Real Data Integration (Next)
To use actual real-time predictions:
1. **Get NWIS data** - Real water level measurements from NWIS API
2. **Prepare 30-day sequences** - Create input tensors from actual data
3. **Run model inference** - Generate predictions with trained model
4. **Update data pipeline** - Replace dummy data loader with real data source

---

## 🎯 Overview

This project implements a sophisticated water level prediction system for the Godavari Basin using **Spatio-Temporal Graph Neural Networks (GNN)**. The system predicts water levels 7 days in advance and provides real-time **flood warnings** with an interactive map visualization.

### Key Features
- 🗺️ **Interactive Flood Warning Map** - Real-time flood status visualization with color-coded markers
- 📊 **7-Day Forecast Charts** - Interactive predictions for each lake
- ⚙️ **User-Adjustable Thresholds** - Simple slider to control flood alert levels
- 📈 **Model Performance Metrics** - R² scores and accuracy per lake
- 🔍 **Real-time Status Dashboard** - KPI cards showing flood counts and risk levels
- 🧠 **Advanced GNN Model** - GAT layers + LSTM + Multi-head attention

---

## 🏗️ Architecture

### Model: Spatio-Temporal GNN
```
Input (30 days) → GAT Layers → LSTM (Temporal) → Multi-head Attention → Output (7-day forecast)
   ↓
   Spatial aggregation between 6 lakes + Temporal learning
   ↓
Water level predictions for each lake
```

### System Components
- **Flood Warning:** Binary classification (FLOOD/SAFE) based on user threshold
- **Map:** Folium-based interactive geographic visualization
- **UI:** Streamlit web application with responsive design
- **Data:** Spatio-temporal sequences (30-day history → 7-day forecast)

---

## 📦 Installation

### Quick Setup (with `uv`)

```bash
# 1. Clone/navigate to project
cd /home/hrishi/Documents/sem8/major_project/app

# 2. Create virtual environment
uv venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

### Detailed Setup
See **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** for comprehensive setup instructions.

---

## 🚀 Quick Start

### Run the Application
```bash
# Make sure venv is activated
source venv/bin/activate

# Run Streamlit app
streamlit run app.py
```

### Using the App

1. **Select a Date**
   - Choose from 4 selection methods: Slider, Date Picker, Date Number, Latest
   - Date range: 2020-01-01 to 2022-10-31 (1035 days)
   - Shows 30-day historical data for context

2. **Adjust Flood Threshold**
   - Enter threshold as water level units (textbox)
   - Default: 90.0 units
   - Map updates instantly

3. **View Flood Warning Map**
   - 🔴 Red markers = FLOOD (above threshold)
   - 🟢 Green markers = SAFE (below threshold)
   - Click markers for lake details

4. **Select Lakes to Monitor**
   - Multi-select from sidebar
   - Choose which lakes to see forecasts for

5. **View 7-Day Forecasts**
   - Click on lake tabs
   - Interactive Plotly charts with:
     - 30-day historical data (actual from dataset)
     - 7-day predictions (model output)
     - Confidence bands (optional toggle)
   - Status metrics: Current Status & Days to Flood

---

## 📊 Project Structure

```
app/
├── 📄 app.py                      # Main Streamlit application
├── ⚙️ config.py                    # Configuration & constants
├── 📋 requirements.txt             # Python dependencies
├── 🔧 .streamlit/config.toml       # Streamlit theming
│
├── 🧠 models/
│   └── final_gnn_model.pth        # Trained model weights
│
├── 📚 data/
│   ├── lake_metrics.csv           # Performance metrics
│   ├── training_history.csv       # Training logs
│   └── lake_coordinates.json      # Geographic coordinates
│
├── 🛠️ utils/
│   ├── flood_warning.py           # Binary flood logic
│   ├── map_generator.py           # Folium map creation
│   ├── data_loader.py             # Data utilities
│   ├── prediction_engine.py       # GNN inference
│   └── visualization.py           # Plotly charts
│
├── 📖 INSTALLATION_GUIDE.md        # Setup instructions
├── 📋 IMPLEMENTATION_SUMMARY.md    # What's implemented
├── 📊 PROJECT_SPEC.md             # Detailed specifications
└── 📚 README.md                   # This file
```

---

## 📊 Model Performance

| Lake | R² Score | RMSE | Status |
|------|----------|------|--------|
| Indravati | 0.873 | 0.362 | ✅ Excellent |
| Sabari | 0.819 | 0.465 | ✅ Good |
| Valamuru | 0.736 | 0.550 | ✅ Good |
| Adhala | 0.761 | 0.581 | ✅ Good |
| Manjira | 0.018 | 1.086 | ⚠️ Poor |
| Girija | -0.001 | 0.405 | ⚠️ Poor |

**Overall:** R² = 0.67, MAE = 0.166 meters

---

## 🔧 Configuration

Key configuration parameters in `config.py`:

```python
# Flood Settings
DEFAULT_FLOOD_THRESHOLD = 0.90  # 90% capacity
FLOOD_COLOR = '#d62728'         # Red
SAFE_COLOR = '#2ca02c'          # Green

# Lake Coordinates (for mapping)
LAKE_COORDINATES = {
    'Adhala': (19.2183, 77.7499),
    'Girija': (19.3500, 77.8500),
    'Indravati': (19.8500, 82.0500),
    'Manjira': (19.0500, 77.8500),
    'Valamuru': (18.4000, 79.1500),
    'Sabari': (17.6000, 82.5000)
}

# Model Parameters
SEQUENCE_LENGTH = 30      # Historical days
FORECAST_HORIZON = 7      # Forecast days
```

---

## 🔬 Core Modules

### `flood_warning.py`
Binary flood classification system:
- `is_flood()` - Check if water level exceeds threshold
- `get_status_text()` - Returns "FLOOD" or "SAFE"
- `get_status_color()` - Red/Green color coding
- `days_until_threshold()` - Days until flood risk

### `map_generator.py`
Interactive map visualization:
- `create_base_map()` - Folium map setup
- `add_lake_markers()` - Color-coded markers
- `add_legend()` - Status legend
- `create_flood_warning_map()` - Complete map

### `prediction_engine.py`
GNN inference:
- `SpatioTemporalGNN` - Full model architecture
- `load_model()` - Load trained weights
- `make_predictions()` - 7-day forecast

### `data_loader.py`
Data utilities:
- Load metrics and training history
- Prepare input sequences
- Normalize/denormalize data
- Create graph edges

### `visualization.py`
Interactive charts:
- `plot_forecast_chart()` - Plotly time series
- `plot_metrics_cards()` - KPI displays
- `plot_lake_status_table()` - Status table

---

## 📋 Dependencies

- **Deep Learning:** `torch==2.2.0`, `torch-geometric==2.4.0`
- **Web App:** `streamlit==1.28.0`, `streamlit-folium==0.6.0`
- **Visualization:** `plotly==5.18.0`, `folium==0.14.0`
- **Data:** `pandas==2.2.0`, `numpy>=1.26.0`, `scikit-learn==1.3.2`

Full list: See `requirements.txt`

---

## 🌐 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect GitHub repo
4. Select `app/app.py` as main file
5. Deploy!

Public URL: `https://share.streamlit.io/your-repo/app/app.py`

---

## 🧪 Testing

### Verify Installation
```bash
python -c "import streamlit; import torch; import folium; print('✅ All imports working!')"
```

### Run App
```bash
streamlit run app.py
```

### Test Flood Map
- Adjust threshold slider → Map markers should change colors
- Click on marker → Popup with lake details should appear

### Test Forecasts
- Select lakes → Forecast tabs should populate
- Charts should render with historical + predicted data

---

## 📚 Documentation

- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Detailed setup instructions
- **[PROJECT_SPEC.md](PROJECT_SPEC.md)** - Complete technical specification
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What's implemented
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Original setup guide

---

## 🎨 UI Features

### Sidebar Controls
- 📅 **Date Selection** - 4 methods: Slider, Date Picker, Date Number, Latest Date
- 🏞️ **Lake Selection** - Multi-select dropdown
- ⚠️ **Flood Threshold** - Water level units textbox (default: 90.0)
- 🎯 **Display Options** - Show/hide confidence bands

### Main Dashboard
1. **Flood Warning Map** - Geographic flood visualization with status indicators
2. **KPI Metrics** - Lakes in flood, safe count, highest risk
3. **Flood Status Table** - Current levels, thresholds, days to flood
4. **7-Day Forecasts** - Tabbed interactive charts per lake
   - Historical data from dataset (30 days)
   - Model predictions (7 days)
   - Real vs Predicted visualization

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Virtual env not found | Run `uv venv venv` |
| Module not found | Activate venv: `source venv/bin/activate` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Model not found | App uses dummy model for demo (shows warning) |
| Map not displaying | Check `streamlit-folium` installation |

---

## 🔮 Future Enhancements

- [ ] Real-time data integration from USGS API
- [ ] Model retraining pipeline
- [ ] SMS/Email alerts for flood warnings
- [ ] Historical flood event analysis
- [ ] Weather integration for rainfall predictions
- [ ] Ensemble predictions from multiple models
- [ ] Mobile app (React Native)

---

## 📈 Workflow

```
User Input (Threshold, Lakes)
    ↓
Load Predictions (7-day forecast)
    ↓
Calculate Flood Status (threshold comparison)
    ↓
Update Map (color markers)
    ↓
Render Charts (Plotly forecast)
    ↓
Display Dashboard (KPI metrics)
```

---

## 📝 License

This project is part of a research initiative for water resource management.

---

## 👤 Author

**Hrishi** - Godavari Basin Water Level Prediction System  
**Date:** March 18, 2026  
**Status:** ✅ Phase 1 MVP Complete

---

## 📞 Support

For issues or questions:
1. Check **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** for setup help
2. Review **[PROJECT_SPEC.md](PROJECT_SPEC.md)** for technical details
3. See **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** for what's implemented

---

**Made with ❤️ for Water Resource Management**
