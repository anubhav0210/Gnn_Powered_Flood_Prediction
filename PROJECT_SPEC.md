# Godavari Basin Water Level Prediction - Streamlit App
## Project Specification Document

---

## 📊 Project Status

### ✅ Phase 1: COMPLETE (MVP - Deployed)
- Interactive web app with all UI components
- Flood warning map with real-time controls
- 7-day forecast charts
- Performance metrics dashboard
- Demo data using realistic patterns

### 🔄 Phase 2: PENDING (Real Data Integration)
- NWIS API data source connection
- Real model inference pipeline
- Live water level updates
- Production deployment

---

## 1. PROJECT OVERVIEW

**Project Name:** Godavari GNN Water Level Predictor  
**Purpose:** Interactive web application for real-time water level predictions across 6 lakes in the Godavari Basin using Graph Neural Networks  
**Technology Stack:** Streamlit, PyTorch, Graph Neural Networks (GNN)  
**Target Audience:** Water resource managers, hydrologists, researchers, government stakeholders  
**Deployment:** Streamlit Cloud (free hosting)  

---

## 2. WHAT'S IMPLEMENTED (Phase 1)

### ✅ 2.0 FLOOD WARNING SYSTEM ON MAP (COMPLETE)
- **Status:** ✅ Fully implemented and working
- **Features:**
  - Interactive Folium map centered on Godavari Basin
  - 6 lake markers with real coordinates
  - Color-coded status: 🔴 Red (FLOOD) / 🟢 Green (SAFE)
  - User-defined threshold slider (0-100%, default 90%)
  - Click popups showing lake details
  - Real-time map updates when threshold changes

### ✅ 2.1 Real-Time Predictions (DEMO MODE)
- **Status:** ✅ Working with realistic demo data
- **Current:** Generates 7-day predictions using your model's accuracy patterns
- **Future:** Will use actual model inference in Phase 2

### ✅ 2.2 Interactive Lake Selection (COMPLETE)
- **Status:** ✅ Fully implemented
- **Features:**
  - Multi-select widget for choosing lakes
  - Real metrics from your training_history.csv
  - Per-lake R² scores displayed

### ✅ 2.3 7-Day Forecast Visualizations (COMPLETE)
- **Status:** ✅ Interactive Plotly charts working
- **Features:**
  - Historical data + predictions in one view
  - Hover tooltips with exact values
  - Tabbed interface for multiple lakes
  - Export ready

### ✅ 2.4 Model Performance Metrics (COMPLETE)
- **Status:** ✅ Real metrics from your CSV
- **Display:**
  - Loads from lake_metrics.csv (R², RMSE, MAE)
  - Per-lake performance table
  - Model accuracy indicators

---

## 3. WHAT NEEDS PHASE 2 (Real Data)

### 🔄 Real Model Inference
- Replace demo predictions with actual model output
- Load trained model from final_gnn_model.pth
- Feed 30-day historical data to model
- Get true 7-day predictions

### 🔄 Real Water Level Data
- Connect to NWIS API (US Geological Survey)
- Get actual current water levels
- Retrieve 30-day historical sequences
- Update every 6 hours (or hourly)

### 🔄 Flood Alert System (Advanced)
- SMS notifications when threshold exceeded
- Email alerts for stakeholders
- Alert history and logs
- Advanced risk levels (currently just FLOOD/SAFE)

---

## CURRENT PHASE 1: CORE FEATURES

### 2.0 ⭐ FLOOD WARNING SYSTEM ON MAP (✅ WORKING)
- **Feature:** Interactive geographic map with binary flood status
- **Components:**
  - ✅ Folium/Streamlit Map centered on Godavari Basin
  - ✅ Lake locations pinned with color-coded markers
  - ✅ Binary flood classification: 🔴 FLOOD or 🟢 NO FLOOD
  - ✅ User-defined flood threshold (0-100% of capacity)
  - ✅ Interactive map with click-to-view details
  
- **Flood Classification (Simple Binary):**
  - 🔴 **FLOOD:** Water level ≥ User-defined threshold
  - 🟢 **NO FLOOD:** Water level < User-defined threshold
  - Example: Threshold = 90% → 90%+ shows red (FLOOD), <90% shows green (SAFE)

- **Interactive Map Features:**
  - Click on lake markers → View current level + status + 7-day forecast
  - Color-coded markers (red = flood risk, green = safe)
  - Mobile-responsive map view
  - Export map view as image

- **Threshold Control:**
  - User-adjustable threshold slider in sidebar (0-100%)
  - Default threshold: 90%
  - Threshold applies to all lakes uniformly

- **Display Metrics:**
  - Current water level (% of capacity)
  - Flood status: FLOOD or NO FLOOD
  - Days until threshold exceeded (based on 7-day forecast)

### 2.1 Real-Time Predictions
- **Feature:** Generate 7-day water level forecasts for selected lakes
- **Input:** Latest historical sequence data (30 days)
- **Output:** Predicted water levels for 7 days ahead
- **Visualization:** Line charts with confidence bands
- **Performance Metric:** Display model accuracy (MAE: 0.166, R²: 0.67)

### 2.2 Interactive Lake Selection
- **Feature:** Multi-select widget for lake monitoring
- **Lakes:** Adhala, Girija, Indravati, Manjira, Valamuru, Sabari
- **Capability:** View predictions for single or multiple lakes simultaneously
- **Color Coding:** Each lake has distinct color in visualizations
- **Status Indicator:** Show per-lake model accuracy (R² score)
- **Map Integration:** Selecting lake highlights it on flood warning map

### 2.3 7-Day Forecast Visualizations
- **Line Charts:** Historical (30 days) + Predicted (7 days) in one view
- **Shaded Areas:** Confidence intervals for predictions
- **Data Points:** Show exact values on hover
- **Time Axis:** Display date/time for each point
- **Comparison View:** Side-by-side charts for multiple lakes
- **Export:** Download predictions as CSV

### 2.4 Model Performance Metrics
- **Dashboard Metrics:**
  - Overall Model Accuracy (R² Score: 0.67)
  - Average Prediction Error (MAE: 0.166)
  - Best Performing Lake: Indravati (R² = 0.873)
  - Worst Performing Lake: Girija (R² = -0.001)
  
- **Per-Lake Metrics Table:**
  - Lake Name
  - R² Score
  - RMSE
  - Training Status
  - Last Updated

### 2.5 Advanced Features
- **Forecast Confidence:** Show uncertainty in predictions
- **Trend Analysis:** Display 7-day trend (↑ increasing, ↓ decreasing, → stable)
- **Historical Comparison:** Compare current year vs. previous year patterns
- **Data Quality Indicator:** Show data availability and freshness
- **Download Reports:** Generate PDF/CSV with predictions and analysis

---

## 3. USER INTERFACE DESIGN

### 3.1 Page Layout

```
┌─────────────────────────────────────────────────────────┐
│         🌊 Godavari Basin Water Level Predictor        │
│   AI-powered 7-day forecasting with Graph Neural Nets  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ SIDEBAR (Control Panel)                                 │
├─────────────────────────────────────────────────────────┤
│ 🏞️ Lake Selection (Multi-select)                       │
│ ├─ ☑ Adhala                                            │
│ ├─ ☑ Girija                                            │
│ ├─ ☑ Indravati (default)                              │
│ └─ ...                                                   │
│                                                          │
│ 📅 Forecast Settings                                   │
│ ├─ Days Ahead: 7 (fixed)                              │
│ └─ Confidence Level: 95%                              │
│                                                          │
│ 🎯 Display Options                                     │
│ ├─ Show Historical Data: ☑                            │
│ ├─ Show Confidence Bands: ☑                           │
│ └─ Units: Meters ▼                                     │
│                                                          │
│ ⚠️ FLOOD THRESHOLD SETTINGS                            │
│ ├─ Threshold (% capacity): 90% [===●====]            │
│ └─ Apply Threshold: ☑                                │
│                                                          │
│ [🔮 Generate Forecast]                                 │
│ [📥 Download Report]                                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ MAIN CONTENT AREA                                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ⚠️ FLOOD WARNING MAP (TOP SECTION) - PRIMARY FEATURE   │
│  ┌─────────────────────────────────────────────────────┐│
│  │  🗺️ Godavari Basin Flood Warning Map                ││
│  │  ┌───────────────────────────────────────────────┐  ││
│  │  │                                               │  ││
│  │  │     🔴 Manjira (FLOOD)                        │  ││
│  │  │         ↳ Level: 92% (Threshold: 90%)        │  ││
│  │  │                                               │  ││
│  │  │  🟢 Adhala (SAFE)  🟢 Indravati (SAFE)       │  ││
│  │  │                                               │  ││
│  │  │  🟢 Valamuru (SAFE)   🟢 Girija (SAFE)       │  ││
│  │  │                                               │  ││
│  │  │           🔴 Sabari (FLOOD)                   │  ││
│  │  │                                               │  ││
│  │  │  [Satellite] [Reset Zoom]                     │  ││
│  │  └───────────────────────────────────────────────┘  ││
│  │                                                      ││
│  │  Legend: 🔴 FLOOD    🟢 SAFE                       ││
│  │                                                      ││
│  │  Click on markers for details                       ││
│  └─────────────────────────────────────────────────────┘│
│                                                          │
│  📊 KEY METRICS (KPI Cards)                            │
│  ┌──────────────┬──────────────┬──────────────┐         │
│  │ In Flood     │ Above Thresh │ Days to Risk │         │
│  │ 2 Lakes      │ Manjira 92%  │ 3 Days       │         │
│  └──────────────┴──────────────┴──────────────┘         │
│                                                          │
│  📈 7-DAY FORECAST CHART (Interactive)                 │
│  ┌─────────────────────────────────────────┐           │
│  │  Water Level (m) + Flood Threshold      │           │
│  │  3.5 ┤           ─ ─ ─ ─ (Flood Threshold)         │
│  │      │                    ╱╲ ╱           │           │
│  │  3.0 ├────────────────────╱─────╲──     │           │
│  │      │             ╱  🔴CRITICAL ╲    │           │
│  │  2.5 ├───────────╱────────────────╲──  │           │
│  │      └─────────────────────────────────┘│           │
│  │       Mon  Tue  Wed  Thu  Fri  Sat  Sun │           │
│  │       ─── Historical ─ Predicted ─ Alert│           │
│  └─────────────────────────────────────────┘           │
│                                                          │
│  ⚠️ FLOOD STATUS TABLE                                 │
│  ┌─────────────────────────────────────┐               │
│  │ Lake         Status    Level  Days  │               │
│  ├─────────────────────────────────────┤               │
│  │ Manjira      🔴 FLOOD   92%    3d   │               │
│  │ Sabari       🔴 FLOOD   91%    5d   │               │
│  │ Adhala       🟢 SAFE    88%    -    │               │
│  │ Valamuru     🟢 SAFE    82%    -    │               │
│  │ Indravati    🟢 SAFE    65%    -    │               │
│  │ Girija       🟢 SAFE    58%    -    │               │
│  └─────────────────────────────────────┘               │
│                                                          │
│  📋 PREDICTIONS TABLE                                  │
│  ┌─────────────┬──────┬──────┬──────┬───────┐          │
│  │ Date        │ Pred │ Conf │ Risk │ Alert │          │
│  ├─────────────┼──────┼──────┼──────┼───────┤          │
│  │ 2026-03-19  │ 2.85 │ 0.92 │ HIGH │ ⚠️    │          │
│  │ 2026-03-20  │ 2.88 │ 0.90 │ CRIT │ 🔴   │          │
│  └─────────────┴──────┴──────┴──────┴───────┘          │
│                                                          │
│  📊 MODEL PERFORMANCE                                  │
│  ┌─────────────────────────────────────┐               │
│  │ Lake         R²      RMSE   Status  │               │
│  ├─────────────────────────────────────┤               │
│  │ Indravati    0.873   0.362  ✅ Exc  │               │
│  │ Sabari       0.819   0.465  ✅ Good │               │
│  │ Valamuru     0.736   0.550  ✅ Good │               │
│  │ Adhala       0.761   0.581  ✅ Good │               │
│  │ Manjira      0.018   1.086  ⚠️ Poor │               │
│  │ Girija      -0.001   0.405  ⚠️ Poor │               │
│  └─────────────────────────────────────┘               │
│                                                          │
│  ℹ️ Model Information                                   │
│  Architecture: Spatio-Temporal GNN (GAT + LSTM + Attn) │
│  Parameters: 500K | Training Epochs: 150               │
│  Last Update: 2026-03-18 | Status: ✅ Ready            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Color Scheme
- **Primary:** Teal/Blue (#1f77b4) - Water theme
- **Secondary:** Orange (#ff7f0e) - Warnings
- **Success:** Green (#2ca02c) - Good predictions
- **Alert:** Red (#d62728) - Poor predictions
- **Neutral:** Gray (#7f7f7f) - Historical data
- **Lake Colors:**
  - Adhala: #1f77b4 (Blue)
  - Girija: #ff7f0e (Orange)
  - Indravati: #2ca02c (Green)
  - Manjira: #d62728 (Red)
  - Valamuru: #9467bd (Purple)
  - Sabari: #8c564b (Brown)

---

## 4. TECHNICAL ARCHITECTURE

### 4.1 File Structure
```
app/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings
├── models/
│   ├── gnn_model.py               # GNN architecture definition
│   └── final_gnn_model.pth        # Trained model weights
├── utils/
│   ├── data_loader.py             # Data loading utilities
│   ├── prediction_engine.py       # Inference logic
│   ├── visualization.py           # Chart generation
│   ├── flood_warning.py           # Flood risk calculation & alerts
│   ├── map_generator.py           # Folium map creation
│   └── metrics.py                 # Metric calculations
├── data/
│   ├── lake_metrics.csv           # Per-lake performance metrics
│   ├── training_history.csv       # Training logs
│   ├── lake_coordinates.json      # Lake lat/lon for mapping
│   └── flood_thresholds.csv       # Historical flood capacity levels
└── requirements.txt               # Python dependencies
```

### 4.2 Data Flow

```
┌──────────────────────────────────┐
│  Input: Selected Lakes + Settings │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ Load Historical Data (30 days)   │
│ - Normalized features ready      │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│ Prepare Graph Structure          │
│ - 6 nodes (lakes)                │
│ - Predefined edges (flow)        │
└──────────────────┬───────────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
┌──────────────────────────┐ ┌──────────────────────────┐
│ Run GNN Inference        │ │ Calculate Flood Risk     │
│ - Input: (1, 30, 6, ...) │ │ - Current % capacity     │
│ - Output: (1, 6, 7)      │ │ - 7-day projection       │
└──────────────────┬───────┘ │ - Risk level classification
                   │         │ - Days until flood warning
                   │         └──────────┬────────────────
                   │                    │
                   └────────────┬───────┘
                                ▼
                   ┌──────────────────────────────┐
                   │ Post-Process Predictions     │
                   │ - Denormalize values         │
                   │ - Calculate confidence bands │
                   │ - Compute trend indicators   │
                   │ - Generate flood alerts      │
                   └──────────────────┬───────────┘
                                      │
                   ┌──────────────────┴───────────────┐
                   ▼                                  ▼
        ┌────────────────────────┐  ┌─────────────────────────┐
        │ Generate Flood Map     │  │ Display Results         │
        │ - Folium map           │  │ - Interactive charts    │
        │ - Color-coded markers  │  │ - Metrics tables        │
        │ - Risk heatmap         │  │ - Flood alerts          │
        │ - Interactive layers   │  │ - Export options        │
        └────────────────────────┘  └─────────────────────────┘
```

### 4.3 Key Components

#### A. Data Loader Module (`data_loader.py`)
**Purpose:** Load and prepare data for inference  
**Functions:**
- `load_historical_data(lake_name, days=30)` → DataFrame
- `normalize_features(data, scaler)` → Normalized array
- `create_graph_structure()` → PyG Data object

#### B. Model Module (`gnn_model.py`)
**Purpose:** Define GNN architecture  
**Classes:**
- `SpatioTemporalGNN` - Complete model definition
- Must match training architecture exactly

#### C. Prediction Engine (`prediction_engine.py`)
**Purpose:** Run inference and manage predictions  
**Functions:**
- `load_model(model_path)` → Trained model
- `predict(model, input_data, edge_index)` → Predictions
- `calculate_confidence(predictions, historical_std)` → Confidence bands
- `denormalize_predictions(pred, scaler)` → Real values

#### D. Visualization Module (`visualization.py`)
**Purpose:** Create interactive charts  
**Functions:**
- `plot_forecast_chart(historical, predictions, lakes)` → Plotly figure
- `plot_metrics_table(lake_metrics)` → Streamlit table
- `plot_confidence_bands(predictions, confidence)` → Area chart
- `plot_trend_indicators(predictions)` → Trend symbols

#### E. Flood Warning Module (`flood_warning.py`) ⭐ NEW
**Purpose:** Calculate binary flood status (Flood/No Flood)  
**Functions:**
- `is_flood(current_level, capacity, threshold=0.90)` → Boolean (True = FLOOD, False = SAFE)
- `get_status_color(is_flood)` → Color code (Red for flood, Green for safe)
- `get_status_text(is_flood)` → Status text ("FLOOD" or "SAFE")
- `days_until_threshold(predictions, threshold, capacity)` → Days until flood (or None if safe)
- `calculate_capacity_percentage(water_level, max_capacity)` → % capacity

**Simple Logic:**
```
if current_water_level >= (threshold * max_capacity):
    status = FLOOD (🔴 Red)
else:
    status = SAFE (🟢 Green)
```

#### F. Map Generator Module (`map_generator.py`) ⭐ NEW
**Purpose:** Create interactive Folium maps with simple flood warning  
**Functions:**
- `create_base_map()` → Folium Map centered on Godavari Basin
- `add_lake_markers(map, lake_data, is_flood_status)` → Add color-coded markers (red/green only)
- `add_legend(map)` → Simple 2-color legend (FLOOD/SAFE)
- `customize_map_controls(map)` → Add basic controls

**Map Features:**
- Red markers (🔴) for lakes in FLOOD state
- Green markers (🟢) for lakes in SAFE state
- Click marker → Show: Lake name, current %, threshold %, days to flood
- Export as image option

#### G. Configuration (`config.py`)
**Purpose:** Centralized settings  
**Content:**
```python
LAKES = ['Adhala', 'Girija', 'Indravati', 'Manjira', 'Valamuru', 'Sabari']
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 7
MODEL_PATH = 'models/final_gnn_model.pth'
METRICS_PATH = 'data/lake_metrics.csv'
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

# Lake maximum capacities (in meter³)
LAKE_CAPACITIES = {
    'Adhala': 1500e6,
    'Girija': 2000e6,
    'Indravati': 5000e6,
    'Manjira': 2800e6,
    'Valamuru': 1200e6,
    'Sabari': 3200e6
}
```

---

## 5. FEATURE SPECIFICATIONS

### 5.0 Feature 0: Flood Warning System with Interactive Map ⭐ TOP PRIORITY

**User Flow:**
1. App loads → Flood warning map displays at top with simple binary status
2. Map shows all 6 lakes with markers:
   - 🔴 Red = FLOOD (water level ≥ threshold)
   - 🟢 Green = SAFE (water level < threshold)
3. User adjusts threshold slider in sidebar (default: 90%)
   - Map updates instantly to reflect new threshold
4. User clicks on a lake marker → Popup shows:
   - Lake name
   - Current water level (% of capacity)
   - User-defined threshold
   - Status: FLOOD or SAFE
   - Days until threshold exceeded (if applicable)
5. User can export map as image

**Implementation Flow:**
```python
# Main app flow - SIMPLE
1. Load threshold from sidebar slider (default 90%)
2. Load predictions for all lakes
3. Calculate: is_flood = (current_level >= threshold)
4. Color code: red if flood, green if safe
5. Display map with colored markers
6. Show simple flood status table
7. Display forecast charts
```

**Map Components (Simple):**
- **Lake Markers:** Click-responsive, only 2 colors (red/green)
- **Legend:** 2-item legend (FLOOD / SAFE)
- **Popup:** Click shows lake name, %, status, days
- **Threshold Control:** Sidebar slider (0-100%)
- **Export:** Download map as PNG

**Flood Status Logic:**
```
Input: Predicted water level + User threshold
1. Calculate % capacity: (predicted_level / max_capacity) * 100
2. Compare: if % >= threshold → FLOOD (🔴)
             else → SAFE (🟢)
3. Color marker accordingly
4. Calculate days until threshold exceeded
```

**Flood Status Features:**

| Component | Details |
|-----------|---------|
| **Markers** | 2 colors: Red (🔴) for flood, Green (🟢) for safe |
| **Threshold** | User-adjustable slider in sidebar, 0-100% |
| **Default** | 90% capacity |
| **Updates** | Map refreshes when threshold slider changes |
| **Popup** | Click marker → Current level, threshold, status, days to flood |
| **Export** | Download map as PNG screenshot |
| **No Alerts** | Simple visual indication only (no notifications) |

### 5.1 Feature 1: Interactive Forecast Generation

**User Flow:**
1. User opens app → Default lakes pre-selected (Adhala, Indravati)
2. User can select/deselect lakes from multi-select widget
3. User clicks "🔮 Generate Forecast" button
4. App displays loading spinner with progress message
5. Predictions appear with visualization

**Implementation Details:**
```python
# Pseudo-code flow
@st.cache_resource
def load_model():
    # Load once, cache across sessions
    
with st.spinner("🔄 Generating forecast..."):
    for lake in selected_lakes:
        pred = model(input_data, edge_index)
        predictions[lake] = denormalize(pred)
    
st.success("✅ Forecast ready!")
```

**Error Handling:**
- Model loading fails → Display error message, suggest troubleshooting
- Data loading fails → Show "Data unavailable" with timestamp
- Prediction fails → Show "Inference error" with details

### 5.2 Feature 2: Real-Time Metrics Dashboard

**Displayed Metrics:**
- **Card 1:** Model Accuracy (R² = 0.67 overall)
- **Card 2:** Avg Prediction Error (MAE = 0.166)
- **Card 3:** Confidence Score (82% average)
- **Card 4:** Best Lake (Indravati)

**Update Logic:**
- Recalculate based on selected lakes
- Show average of selected lake metrics
- Display trend indicators (↑ improving, ↓ declining)

### 5.3 Feature 3: Interactive Visualizations

**Chart Types:**

1. **Line Chart (Primary Forecast)**
   - X-axis: Date/Time (30 days history + 7 days forecast)
   - Y-axis: Water Level (meters)
   - Lines: Historical (solid gray), Predicted (colored per lake)
   - Area: Shaded confidence band (light color)
   - Hover: Show exact values, lake name, date

2. **Comparison Chart (Multiple Lakes)**
   - Subplots if >2 lakes selected
   - Same scale for easier comparison
   - Color-coded per lake

3. **Performance Table**
   - Columns: Lake, R², RMSE, Status, Last Update
   - Sortable columns
   - Color-coded status (green=good, red=poor)

### 5.4 Feature 4: Download & Export

**Export Options:**
1. **CSV Download:**
   - Columns: Date, Lake, Predicted_Level, Confidence, Trend
   - Filter by selected lakes
   - Include headers and metadata

2. **PDF Report:**
   - Title page with project info
   - Summary statistics
   - Charts as images
   - Performance table
   - Timestamp of generation

---

## 6. IMPLEMENTATION PHASES

### Phase 1: MVP (Minimum Viable Product) - Week 1
**Goals:** Core prediction functionality + basic flood warning map

**Deliverables:**
- [ ] `app.py` - Main Streamlit app
- [ ] `config.py` - Configuration
- [ ] `flood_warning.py` - Binary flood status calculation
- [ ] `map_generator.py` - Basic Folium map with 2-color markers
- [ ] Model loading and inference
- [ ] Basic chart visualization
- [ ] Lake selection widget
- [ ] Flood threshold slider (0-100%)
- [ ] Flood warning map displays on top with red/green markers

**Success Criteria:**
- App runs without errors
- Flood map displays correctly with all 6 lakes
- Red/green markers based on threshold comparison
- Threshold slider works and updates map
- Can select lake and generate forecast
- Predictions display in chart
- Model loads successfully

### Phase 2: Enhanced UI & Features - Week 2
**Goals:** Complete flood warning system + advanced features

**Deliverables:**
- [ ] Flood warning map with interactive popups
- [ ] Simple red/green marker system
- [ ] Click-on-marker detailed information
- [ ] Current flood status table (showing FLOOD/SAFE)
- [ ] Multi-lake simultaneous forecasting
- [ ] Metrics cards (KPI dashboard) - count of FLOOD vs SAFE
- [ ] Performance table
- [ ] Confidence bands on charts
- [ ] Trend indicators
- [ ] Better styling with custom CSS

**Success Criteria:**
- Flood map is fully interactive
- Clicking markers shows forecast data
- Multiple lakes display correctly on map
- Metrics auto-calculate per selection
- Charts are interactive and responsive
- UI is professional-looking

### Phase 3: Advanced Features - Week 3
**Goals:** Polish and additional functionality

**Deliverables:**
- [ ] CSV/PDF export functionality
- [ ] Export map as image
- [ ] Historical comparison view (optional)
- [ ] Advanced filters (optional)
- [ ] Data refresh mechanism
- [ ] Error handling & logging
- [ ] Documentation

**Success Criteria:**
- Exports work correctly (map, data, reports)
- Error messages are helpful
- App handles edge cases gracefully
- Code is well-documented

### Phase 4: Deployment & Testing - Week 4
**Goals:** Deploy to Streamlit Cloud

**Deliverables:**
- [ ] Deploy to Streamlit Cloud
- [ ] Create `.streamlit/config.toml`
- [ ] Create `requirements.txt`
- [ ] Test on multiple devices
- [ ] Performance optimization
- [ ] Final documentation

**Success Criteria:**
- App deployed and publicly accessible
- Works on desktop and mobile
- Load times < 3 seconds
- Ready for presentation

---

## 7. DEPENDENCIES & REQUIREMENTS

### Python Packages
```
streamlit==1.28.0
torch==2.2.0
torch-geometric==2.4.0
pandas==2.0.0
numpy==1.24.0
plotly==5.14.0
scikit-learn==1.3.0
matplotlib==3.7.0
pytz==2023.3
kaleido==0.2.1
folium==0.14.0
streamlit-folium==0.6.0
```

### System Requirements
- Python 3.8+
- RAM: 4GB minimum, 8GB recommended
- GPU: Optional (CUDA support for faster inference)
- Storage: 500MB for model + data

### Installation with UV Package Manager

**Why UV?** Lightning-fast Python package manager (5-10x faster than pip)

**Quick Setup:**

```bash
# 1. Create virtual environment
uv venv venv

# 2. Activate
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Verify
python -c "import streamlit; import torch; import folium; print('✅ Ready!')"

# 5. Run app
streamlit run app.py
```

**Full Setup Guide:** See `SETUP_GUIDE.md` for detailed instructions, troubleshooting, and best practices.

**Python Version:** PyTorch 2.2.0 (instead of 2.0.0) for Python 3.12 compatibility.

---

## 8. TESTING STRATEGY

### 8.1 Unit Tests
- Model loading tests
- Data normalization tests
- Prediction output shape verification
- Confidence calculation tests

### 8.2 Integration Tests
- Full pipeline end-to-end
- Multiple lake selection handling
- Export functionality
- Error scenarios

### 8.3 UI/UX Tests
- Responsive design on mobile/desktop
- Chart interactivity
- Loading states
- Error message clarity

### 8.4 Performance Tests
- Model inference time: < 2 seconds
- App load time: < 3 seconds
- Chart rendering: < 1 second
- Memory usage: < 2GB

---

## 9. DEPLOYMENT

### 9.1 Streamlit Cloud Deployment
1. Push code to GitHub
2. Connect GitHub repo to Streamlit Cloud
3. Create `.streamlit/config.toml` for customization
4. Deploy with one-click

### 9.2 Configuration File (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "viewer"

[logger]
level = "info"
```

### 9.3 GitHub Repository Structure
```
godavari-gnn-predictor/
├── app.py
├── config.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   └── final_gnn_model.pth
├── utils/
├── data/
└── README.md
```

---

## 10. SUCCESS METRICS

**For Users:**
- ✅ Can generate forecast in <2 clicks
- ✅ Flood warnings are immediately visible on map (red/green)
- ✅ Predictions are visually clear
- ✅ Understand model accuracy per lake
- ✅ Can export results easily
- ✅ Understands flood status clearly (Flood or Safe)
- ✅ Can adjust threshold easily with slider

**For System:**
- ✅ Model inference time < 2 seconds
- ✅ Map renders in < 2 seconds
- ✅ App load time < 3 seconds
- ✅ Zero runtime errors
- ✅ Works on mobile devices
- ✅ Flood status is 100% accurate (based on threshold)

**For Project:**
- ✅ Deployed to Streamlit Cloud
- ✅ Presentation-ready with simple flood warning map
- ✅ Code well-documented
- ✅ Easily maintainable
- ✅ Easy to understand (no complex classifications)

---

## 11. FUTURE ENHANCEMENTS

- [ ] Real-time data integration from USGS API
- [ ] Model retraining pipeline with new data
- [ ] Historical backtesting dashboard
- [ ] Ensemble predictions from multiple models
- [ ] Anomaly detection for unusual patterns
- [ ] Integration with government dashboards
- [ ] Mobile app (React Native)
- [ ] Advanced flood risk classifications (if needed later)
- [ ] Weather integration for rainfall predictions

---

## 12. RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Model not in correct format | Critical | Test model loading before app development |
| Data loading fails | High | Create mock data for testing |
| GPU not available | Medium | Ensure CPU fallback works |
| Streamlit performance | Medium | Profile and optimize bottlenecks |
| Map rendering slow | Medium | Cache map, optimize Folium |
| Incorrect threshold logic | Medium | Test with known values |
| Model outdated | Low | Plan retraining schedule |
| Map not responsive on mobile | Medium | Use responsive Folium config |

---

**Version:** 2.1 (Simplified Flood Warning - Binary Only)  
**Last Updated:** March 18, 2026  
**Owner:** Your Name  
**Status:** Ready for Implementation  
**Top Priority Feature:** ⭐ Simple Flood Warning Map (Red/Green Binary System)
