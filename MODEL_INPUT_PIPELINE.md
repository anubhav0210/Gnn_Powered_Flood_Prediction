# Model Input Pipeline - Complete Explanation

## Problem You Identified

You noticed the app was:
1. ❌ Always using the LAST 30 days of data (most recent)
2. ❌ Never letting YOU choose which date range to analyze
3. ❌ Not clear what data was actually being fed into the model

**Now Fixed:** You can select ANY date from 2020-01-01 to 2022-10-31 and make predictions!

---

## Data Available

Your dataset:
- **Files:** 6 lake CSV files in `/proccessed _data/processed/`
- **Lakes:** Adhala, Girija, Indravati, Manjira, Valamuru, Sabari
- **Date Range:** 2020-01-01 to 2022-10-31 (1035 days total)
- **Columns per CSV:** 16 columns including timestamp, rainfall, humidity, water level, discharge, and temporal features

Each CSV file contains normalized values (already standardized to mean≈0, std≈1).

---

## How Data Flows into the Model

### Step 1: You Select a Date (in Streamlit UI)
```
Sidebar → "📅 Date Selection" → Select end date for 30-day window
Default: 2022-10-31 (latest date)
But you can choose ANY date from 2020-01-31 to 2022-10-31
```

### Step 2: Load 30-Day Data for All 6 Lakes (Your Selected Date)
**Function:** `data_loader.load_30day_sequence(lake_names, end_date="2022-01-15")`

This loads the most recent 30 days OF DATA ENDING AT YOUR SELECTED DATE:
- Gets all data up to your selected date
- Takes the last 30 rows
- Returns a dictionary with 6 DataFrames

Example:
```python
selected_date = "2022-01-15"
lake_sequences = {
    'Adhala':     DataFrame with 30 rows from 2021-12-17 to 2022-01-15,
    'Girija':     DataFrame with 30 rows from 2021-12-17 to 2022-01-15,
    'Indravati':  DataFrame with 30 rows from 2021-12-17 to 2022-01-15,
    ... (3 more lakes)
}
```

### Step 3: Extract 9 Features for Each Lake, Each Day
**Function:** `data_loader.prepare_input_sequence(lake_sequences, lake_names)`

For each lake and each day, extract these 9 normalized features:
1. `rainfall_mm` - Rainfall in millimeters
2. `humidity_pct` - Humidity percentage  
3. `water_level_m` - Water level in meters ⭐ (This is what we predict!)
4. `discharge_m3s` - Discharge (flow rate)
5. `day_of_year` - Day number (1-366)
6. `month` - Month (1-12)
7. `year` - Year
8. `day_sin` - Cyclical encoding of day (sine)
9. `day_cos` - Cyclical encoding of day (cosine)

### Step 4: Create Tensor with Shape (1, 30, 6, 9)

The CSV data is reshaped into a 4D tensor:

```
Tensor shape: (1, 30, 6, 9)
   ↓    ↓   ↓  ↓
   |    |   |  └─ 9 features per lake per day
   |    |   └──── 6 lakes (nodes in graph)
   |    └──────── 30 days (temporal sequence)
   └──────────── 1 batch (inference is done one at a time)
```

**Visual example of tensor layout:**

```python
tensor[0, day_idx, lake_idx, feature_idx]

tensor[0, 0, 0, 0]   = Rainfall on day 1 at Adhala
tensor[0, 0, 0, 2]   = Water level on day 1 at Adhala
tensor[0, 15, 3, 2]  = Water level on day 16 at Manjira
tensor[0, 29, 5, 3]  = Discharge on day 30 at Sabari
```

**How lakes are indexed:**
```
node_idx=0: Adhala
node_idx=1: Girija
node_idx=2: Indravati
node_idx=3: Manjira
node_idx=4: Valamuru
node_idx=5: Sabari
```

### Step 5: Create Graph Structure

**Function:** `data_loader.create_graph_edges()`

Define how lakes are connected based on Godavari basin water flow:
```
Adhala ←→ Girija        (Upper basin)
   ↓         ↓
Indravati ← Manjira     (Middle basin)
   ↓         ↓
Valamuru ←→ Sabari      (Lower basin)
```

This connectivity is encoded as edge indices for the Graph Neural Network.

### Step 6: Model Inference

**Function:** `prediction_engine.make_predictions(model, input_tensor, edge_index)`

The model processes:
- **Input:** Tensor (1, 30, 6, 9) + Graph structure
- **Processing:**
  1. **Input Projection:** 9 features → hidden dimension (256)
  2. **GAT Layers:** Apply spatial graph attention (how lakes influence each other)
  3. **LSTM Layers:** Process temporal patterns across 30 days
  4. **Temporal Attention:** Focus on important time steps
  5. **Output Layers:** Generate 7-day predictions
- **Output:** Array with shape (6, 7)
  - 6 lakes × 7 days forecast

### Step 7: Display Results

The app shows:
1. **Flood Warning Map:** Current water levels and flood status for each lake
2. **Forecast Charts:** Historical + Predicted water levels for selected lakes
3. **Metrics:** Flooded count, highest-risk lake, etc.

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  User selects date in Streamlit: "2022-01-15"              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  load_30day_sequence("Adhala", end_date="2022-01-15")      │
│  Returns: 30 rows from 2021-12-17 to 2022-01-15           │
│  Same for all 6 lakes                                       │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  prepare_input_sequence(lake_sequences)                     │
│  - Extract 9 features per lake per day                      │
│  - Stack into tensor (1, 30, 6, 9)                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  create_graph_edges()                                        │
│  Define lake connectivity (water flow network)              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  model.forward(input_tensor, edge_index)                    │
│  GNN processes:                                              │
│  - Spatial: GAT layers see how lakes influence each other   │
│  - Temporal: LSTM captures daily patterns                   │
│  - Attention: Focuses on important features                 │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Output: predictions_array with shape (6, 7)               │
│  [                                                           │
│    [pred_day1_Adhala, ..., pred_day7_Adhala],             │
│    [pred_day1_Girija, ..., pred_day7_Girija],             │
│    ... (4 more lakes)                                       │
│  ]                                                           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Display in Streamlit:                                       │
│  - Map with current water levels                            │
│  - Charts with 30-day history + 7-day forecast              │
│  - Flood warning status                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Insights

### ALL Data is Used
- You have **1035 days** of data (2020-01-01 to 2022-10-31)
- For any date, the model uses the **30 most recent days** ending at that date
- The model was trained on multiple 30-day windows from this data

### Data is Normalized
- All 9 features are already normalized (mean≈0, std≈1) in the CSV files
- The model expects normalized input
- Predictions are also in normalized space

### Graph Structure Matters
- The 6 lakes aren't treated independently
- The GNN uses the connectivity (water flows from upper → middle → lower basins)
- Each lake's prediction influences neighboring lakes

### Temporal Patterns Captured
- LSTM learns daily patterns across 30 days
- Attention mechanism learns which days are most important
- Combines spatial + temporal information for predictions

---

## How to Use in Streamlit

1. **Open the app:** `streamlit run app.py`
2. **Select a date:** Use the slider in the sidebar to pick any date from 2020-01-31 to 2022-10-31
3. **View predictions:** The app automatically:
   - Loads 30 days of data ending at your selected date
   - Runs model inference
   - Shows historical + predicted water levels
   - Displays flood warnings

---

## Testing the Pipeline

To verify the pipeline works:

```python
import utils.data_loader as dl
import config

# Get available dates
dates = dl.get_available_dates()
print(f"Data range: {dates[0]} to {dates[-1]}")

# Select a date and load data
selected_date = "2021-06-15"
lake_sequences = dl.load_30day_sequence(config.LAKES, end_date=selected_date)

# Create tensor
tensor = dl.prepare_input_sequence(lake_sequences, config.LAKES)
print(f"Tensor shape: {tensor.shape}")  # Should be (1, 30, 6, 9)

# Run inference
from utils import prediction_engine
model = prediction_engine.load_model()
edge_index = torch.as_tensor(dl.create_graph_edges(), dtype=torch.long).t().contiguous()
predictions = prediction_engine.make_predictions(model, tensor, edge_index)
print(f"Predictions shape: {predictions.shape}")  # Should be (6, 7)
```

---

## Summary

✅ **You CAN now:**
- Choose ANY date from your 3+ years of data
- See what data is fed into the model
- Understand exactly how 30 days × 6 lakes × 9 features becomes predictions
- Make predictions for historical dates, not just the latest date

✅ **The model uses:**
- Real data from your CSV files
- Normalized values (standardized)
- 30-day historical window ending at YOUR chosen date
- Graph structure capturing water flow relationships
- Both spatial (lake connections) and temporal (daily patterns) information
