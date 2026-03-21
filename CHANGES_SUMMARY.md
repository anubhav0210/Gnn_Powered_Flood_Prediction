# Changes Made - Date Selection & Data Pipeline Improvements

## Summary
Fixed the data pipeline to allow you to select ANY date from your 1035 days of available data (2020-01-01 to 2022-10-31), instead of always using the latest date.

---

## Files Modified

### 1. `utils/data_loader.py`

#### NEW FUNCTION: `load_30day_sequence(lake_names, end_date=None)`
```python
# OLD: Always used last 30 days
lake_sequences = load_30day_sequence(config.LAKES)

# NEW: Use 30 days ending at specified date
lake_sequences = load_30day_sequence(config.LAKES, end_date="2021-06-15")
```

**What it does:**
- Takes a specific end date (or None for latest)
- Loads all data up to that date
- Returns last 30 rows (30-day window ending at that date)
- Works for all 6 lakes

---

#### NEW FUNCTION: `get_available_dates(lake_name=None)`
```python
dates = get_available_dates()
# Returns: [datetime(2020,1,1), datetime(2020,1,2), ..., datetime(2022,10,31)]
```

**What it does:**
- Returns complete list of all dates in your data
- Used by Streamlit to populate date slider

---

#### NEW FUNCTION: `get_date_range_info()`
```python
info = get_date_range_info()
# Returns: {
#   'start_date': datetime(2020, 1, 1),
#   'end_date': datetime(2022, 10, 31),
#   'num_days': 1035
# }
```

**What it does:**
- Gets the complete date range available in data
- Used to display info to user

---

#### NEW FUNCTION: `load_historical_data_for_date_range(lake_name, start_date, end_date)`
```python
timestamps, levels = load_historical_data_for_date_range(
    "Adhala",
    start_date="2022-01-01",
    end_date="2022-01-31"
)
```

**What it does:**
- Get water level data for any custom date range
- Used for visualization (shows historical data on charts)

---

#### ENHANCED: `prepare_input_sequence(lake_sequences, lake_names)`
- Added comprehensive documentation explaining:
  - Input/output shapes
  - The 9 extracted features
  - Tensor memory layout
  - How normalized data flows into the model

---

### 2. `app.py`

#### NEW SIDEBAR SECTION: Date Selection
```python
# Get available date range
date_range_info = data_loader.get_date_range_info()
available_dates = data_loader.get_available_dates()

# Slider to select end date
selected_date_idx = st.sidebar.slider(
    "Select end date for 30-day prediction window:",
    min_value=29,  # Need at least 30 days
    max_value=len(available_dates) - 1,
    value=len(available_dates) - 1,  # Default: latest
    step=1
)
selected_end_date = available_dates[selected_date_idx]
```

**What it does:**
- Shows available data range to user
- Lets user select ANY date with a slider
- Displays selected date as feedback

---

#### UPDATED: Data Loading Section
```python
# OLD: Always used latest data
lake_sequences = data_loader.load_30day_sequence(config.LAKES)

# NEW: Uses selected date
lake_sequences = data_loader.load_30day_sequence(
    config.LAKES, 
    config.SEQUENCE_LENGTH, 
    end_date=selected_end_date_str
)
```

---

#### UPDATED: Water Level Loading
```python
# OLD: Always got latest water level
current_levels[lake_name] = data_loader.get_latest_water_level(lake_name)

# NEW: Gets water level for selected date
mask = df['timestamp'] <= pd.to_datetime(selected_end_date_str)
df_filtered = df[mask]
current_levels[lake_name] = df_filtered.iloc[-1]['water_level_m']
```

---

#### UPDATED: Historical Data for Charts
```python
# OLD: Always used last 30 days from latest data
timestamps, levels = data_loader.get_historical_and_current_data(lake_name, days_back=30)

# NEW: Uses 30 days ending at selected date
start_date_obj = selected_end_date - timedelta(days=29)
timestamps, levels = data_loader.load_historical_data_for_date_range(
    lake_name,
    start_date=start_date_str,
    end_date=selected_end_date_str
)
```

---

## Data Flow - Before vs After

### BEFORE (Only Latest Data)
```
User opens app
    ↓
App loads LAST 30 days (2022-10-02 to 2022-10-31)
    ↓
Model always predicts for latest date
    ↓
No way to analyze historical periods
```

### AFTER (Date Selection)
```
User opens app
    ↓
Sidebar shows: "Data from 2020-01-01 to 2022-10-31"
    ↓
User selects date: "2021-06-15"
    ↓
App loads 30 days ending at 2021-06-15 (2021-05-17 to 2021-06-15)
    ↓
Model makes predictions for 2021-06-15 + 7-day forecast
    ↓
Charts show June 2021 historical data + forecast
```

---

## How the 30-Day Tensor is Created

For any selected date, the pipeline:

1. **Load Data** (for your selected date)
   ```
   [30 days of CSV data for each lake]
   Each row: timestamp, rainfall, humidity, water_level, discharge, features...
   ```

2. **Extract 9 Features**
   ```
   Select: rainfall_mm, humidity_pct, water_level_m, discharge_m3s,
           day_of_year, month, year, day_sin, day_cos
   (normalized values already in CSV)
   ```

3. **Stack into Tensor**
   ```
   Shape: (1, 30, 6, 9)
   
   [1]    = batch size (1 = single inference)
   [30]   = days (your selected 30-day window)
   [6]    = lakes (Adhala, Girija, Indravati, Manjira, Valamuru, Sabari)
   [9]    = features per lake per day
   ```

4. **Pass to Model**
   ```
   GNN processes spatial (lake connections) + temporal (daily patterns)
   ```

5. **Get Predictions**
   ```
   Output shape: (6, 7)
   6 lakes × 7 days forecast
   ```

---

## Verification Tests Passed

✅ All 1035 dates load correctly
✅ Can load 30-day sequences for any date
✅ Tensor shape is correct: (1, 30, 6, 9)
✅ All 6 lakes have data
✅ All 9 features are extracted
✅ Model inference works
✅ Streamlit UI works without errors

---

## Usage

### In Streamlit
1. Run: `streamlit run app.py`
2. Look at sidebar → "📅 Date Selection"
3. Use slider to pick any date from 2020-01-31 to 2022-10-31
4. Watch as predictions change based on your selected date!

### In Python
```python
import utils.data_loader as dl
import config

# Pick any date
selected_date = "2021-09-15"

# Load 30-day sequence
sequences = dl.load_30day_sequence(config.LAKES, end_date=selected_date)

# Prepare tensor
tensor = dl.prepare_input_sequence(sequences, config.LAKES)

# Make predictions
predictions = model(tensor)  # Shape: (6, 7)
```

---

## Data You're Using

- **Source:** `/proccessed _data/processed/` (6 CSV files)
- **Date Range:** 2020-01-01 to 2022-10-31 (1,035 days)
- **Lakes:** Adhala, Girija, Indravati, Manjira, Valamuru, Sabari
- **Features:** Normalized rainfall, humidity, water level, discharge, temporal features
- **What's Predicted:** Water level (water_level_m) for next 7 days

---

## Summary of Improvements

| Aspect | Before | After |
|--------|--------|-------|
| Date selection | None - always latest | Slider for any date in range |
| Data used | Always 2022-10-02 to 2022-10-31 | Any 30-day window you choose |
| Historical data shown | Always last 30 days | Historical data for selected date range |
| Predictions for | Always latest date | Your selected date |
| Data pipeline clarity | Implicit | Explicit with documentation |

You can now analyze predictions across your entire 3+ years of data! 🎉
