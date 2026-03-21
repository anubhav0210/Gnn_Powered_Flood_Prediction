# Model Input Structure - Visual Explanation

## The Core Question You Asked: "How is data inputted into the model?"

## Answer: A (1, 30, 6, 9) Tensor

---

## Visual Breakdown

### What is (1, 30, 6, 9)?

```
┌─────────────────────────────────────────────────────┐
│ Tensor Shape: (1, 30, 6, 9)                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Dimension 1: Batch Size = 1                       │
│  Dimension 2: Time Steps = 30 (days)               │
│  Dimension 3: Nodes = 6 (lakes)                    │
│  Dimension 4: Features = 9 (per lake per day)      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Dimension Breakdown

### Dimension 1: Batch Size (1)
```
[1] = We process ONE prediction at a time
      (inference mode: not a batch of multiple)

During training: might be [32] or [64] (batches)
During inference: always [1]
```

### Dimension 2: Sequence Length (30)
```
[30] = 30 days of historical data

Example timeline:
  Day 1:  May 17, 2021
  Day 2:  May 18, 2021
  ...
  Day 30: June 15, 2021
  
  ↓ Then predict:
  Day +1: June 16, 2021
  Day +2: June 17, 2021
  ...
  Day +7: June 22, 2021
```

### Dimension 3: Graph Nodes (6)
```
[6] = 6 lakes in Godavari basin

Node 0: Adhala
Node 1: Girija
Node 2: Indravati
Node 3: Manjira
Node 4: Valamuru
Node 5: Sabari

The GNN learns how these lakes are connected!
```

### Dimension 4: Features (9)
```
[9] = 9 features per lake per day

Feature 0: rainfall_mm         → How much rain
Feature 1: humidity_pct        → Air humidity
Feature 2: water_level_m       → ⭐ WHAT WE PREDICT
Feature 3: discharge_m3s       → Water flow rate
Feature 4: day_of_year         → Day number (1-366)
Feature 5: month               → Month (1-12)
Feature 6: year                → Year
Feature 7: day_sin             → sin(day of year)
Feature 8: day_cos             → cos(day of year)

ALL VALUES ARE NORMALIZED (mean≈0, std≈1)
```

---

## Memory Layout

How to access tensor values:

```python
tensor[batch, time, node, feature]

Examples:
tensor[0, 0, 0, 0]     = Adhala's rainfall on May 17
tensor[0, 0, 0, 2]     = Adhala's water level on May 17
tensor[0, 15, 3, 2]    = Manjira's water level on May 31
tensor[0, 29, 5, 3]    = Sabari's discharge on June 15 (last day)
tensor[0, 5, 2, 7]     = Indravati's day_sin on May 22
```

---

## Complete Example

### User selects: "2021-06-15"

**What gets loaded from CSV:**

```
Adhala CSV (30 rows × 16 columns)
─────────────────────────────────────────────────────────
timestamp    | rainfall_mm | humidity_pct | ... | day_sin | day_cos
─────────────────────────────────────────────────────────
2021-05-17   | -0.194      | -0.503       | ... | 0.652   | 0.758
2021-05-18   | -0.194      | -1.441       | ... | 0.669   | 0.743
...          | ...         | ...          | ... | ...     | ...
2021-06-15   | -0.194      | 0.300        | ... | 0.423   | 0.906

(Same for other 5 lakes)
```

**Step 1: Extract 9 features**

```
Adhala features (30 rows × 9 columns)
────────────────────────────────────────────────────────
rainfall_mm | humidity_pct | water_level_m | ... | day_cos
────────────────────────────────────────────────────────
-0.194      | -0.503       | -0.573        | ... | 0.758
-0.194      | -1.441       | -0.573        | ... | 0.743
...         | ...          | ...           | ... | ...
-0.194      | 0.300        | -0.285        | ... | 0.906
```

**Step 2: Stack all lakes**

```
30 × 6 × 9 array
(30 days) × (6 lakes) × (9 features)

Position [0, 0, 0] = Day 1, Adhala, Feature 0 (rainfall)
Position [0, 0, 1] = Day 1, Adhala, Feature 1 (humidity)
...
Position [0, 0, 8] = Day 1, Adhala, Feature 8 (day_cos)
Position [0, 1, 0] = Day 1, Girija, Feature 0 (rainfall)
...
Position [29, 5, 8] = Day 30, Sabari, Feature 8 (day_cos)
```

**Step 3: Add batch dimension**

```
(1, 30, 6, 9) tensor

1 × 30 × 6 × 9 = 1,620 numbers fed to the model!
```

---

## What the Model Does With It

### Processing Flow

```
Input: (1, 30, 6, 9) tensor
       + Graph edges (lake connectivity)
       │
       ├─→ Input Projection (1, 30, 6, 256)
       │   Convert 9 features → 256 dimensional embeddings
       │
       ├─→ GAT Layers (temporal spatial learning)
       │   Learn how lakes influence each other
       │   Keep processing across 30 days
       │
       ├─→ LSTM (temporal patterns)
       │   Capture day-to-day changes
       │   Learn seasonal patterns
       │   Learn flow dependencies
       │
       ├─→ Temporal Attention
       │   Focus on important days
       │   Ignore noise
       │
       └─→ Output Layers (1, 6, 7)
           6 lakes × 7 days forecast
           ↓
Output: Predictions array (6, 7)
```

### Output Interpretation

```
predictions[0, 0] = Adhala water level prediction for June 16
predictions[0, 1] = Adhala water level prediction for June 17
...
predictions[0, 6] = Adhala water level prediction for June 22

predictions[5, 0] = Sabari water level prediction for June 16
...
predictions[5, 6] = Sabari water level prediction for June 22
```

---

## Data You're Providing

### From Your 3+ Years of Data (2020-2022)

```
CSV File: adhala_cleaned.csv (1036 rows)
─────────────────────────────────────────
timestamp    |rainfall|humidity|water_level|discharge|...|day_sin|day_cos
2020-01-01   |   -0.19|  -0.50|     -0.57|    -0.36|...|  0.017|  0.9999
2020-01-02   |   -0.19|  -1.44|     -0.57|    -0.36|...|  0.034|  0.9994
2020-01-03   |   -0.19|  -0.06|     -0.57|    -0.36|...|  0.052|  0.9987
...
2022-10-30   |   -0.19|   0.30|     -0.28|    -0.32|...|  -0.876| 0.4825
2022-10-31   |   -0.19|  -0.05|     -0.31|    -0.32|...|  -0.893| 0.4502

(Same structure for other 5 lakes)
```

### Selection Process

```
User: Selects date "2021-06-15"
       ↓
App: "Get last 30 rows ending at 2021-06-15"
       ↓
App: Finds rows from 2021-05-17 to 2021-06-15
       ↓
App: Extracts 9 features from 16 available columns
       ↓
App: Stacks into (1, 30, 6, 9) tensor
       ↓
Model: Processes with GNN layers
       ↓
Output: (6, 7) predictions
```

---

## Key Insights

### All Data is Normalized
- Every value in the CSV is already standardized
- Mean ≈ 0, Standard Deviation ≈ 1
- The model was trained on normalized data
- Predictions are in normalized space

### Graph Structure Matters
```
Water flows: Upper → Middle → Lower basins

Adhala ←→ Girija
   ↓         ↓
Indravati ← Manjira
   ↓         ↓
Valamuru ←→ Sabari

The GNN learns these connections!
Each lake's water level depends on upstream lakes.
```

### Temporal Patterns Captured
```
30-day history allows model to learn:
- Daily variations (diurnal cycles)
- Weekly patterns
- Monthly trends
- Seasonal changes
- Upstream/downstream effects

Longer sequences = better patterns
But 30 days balances history vs computation
```

---

## Concrete Numbers

### Tensor Size
```
Shape: (1, 30, 6, 9)
Total elements: 1 × 30 × 6 × 9 = 1,620 float32 values
Memory: 1,620 × 4 bytes = 6,480 bytes ≈ 6.4 KB
```

### Model Input
```
Input to GNN:
- 1,620 feature values
- Lake connectivity graph (6 nodes, ~12 edges)

Output from GNN:
- 42 predictions (6 lakes × 7 days)
```

---

## How Data Flows Through App

```
CSV Files (2020-2022)
    ↓
load_30day_sequence()
    ↓
30 rows × 6 lakes DataFrame
    ↓
prepare_input_sequence()
    ↓
Extract 9 features from 16 available
    ↓
Stack and normalize
    ↓
(1, 30, 6, 9) torch.Tensor
    ↓
Model.forward(tensor, edge_index)
    ↓
(6, 7) predictions array
    ↓
Streamlit displays:
- Water level charts
- Flood warnings
- Status indicators
```

---

## Summary

**Your data pipeline:**

| Step | Input | Output | Code |
|------|-------|--------|------|
| 1 | Date string "2021-06-15" | 30 days of CSV data | `load_30day_sequence()` |
| 2 | 30 rows × 16 columns | 30 rows × 9 features | `prepare_input_sequence()` |
| 3 | 30 × 6 × 9 array | (1, 30, 6, 9) tensor | `torch.from_numpy()` |
| 4 | (1, 30, 6, 9) + graph | (6, 7) predictions | `model.forward()` |
| 5 | (6, 7) array | Charts in UI | `visualization.plot()` |

**You control:** Date selection (1035 possible dates)  
**Model gets:** Exactly 1,620 normalized values  
**Model produces:** 42 predictions (7 days × 6 lakes)  
**You see:** Historical + forecast charts  

Everything is real data from your CSV files! ✓
