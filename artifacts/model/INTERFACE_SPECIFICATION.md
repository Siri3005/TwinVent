# Model Interface Specification

**Version:** 1.0.0  
**Date:** 2026-09-29  
**For:** TwinVent Offline App Integration

## Overview

This document defines the stable interface contract between the ventilator pressure prediction model and the offline replay application (Owner 2).

## Interface Contract

### Input Format

**One breath as ordered rows**

Required columns:
- `id` (int): Unique identifier for each time sample
- `time_step` (float): Time within breath cycle (seconds)
- `u_in` (float): Inspiratory flow control signal (range: 0-100)
- `u_out` (int): Expiratory valve control (0=closed, 1=open)
- `R` (int): Resistance parameter (5, 20, or 50)
- `C` (int): Compliance parameter (10, 20, or 50)

**Critical Requirements:**
1. Rows MUST be sorted by `time_step` in chronological order
2. R and C are REQUIRED by this benchmark model (cannot be null)
3. All values must be numeric and finite
4. Each breath typically contains 80 time samples

**Example Input:**
```csv
id,time_step,u_in,u_out,R,C
1,0.000000,0.083334,0,20,50
2,0.033652,18.383041,0,20,50
3,0.067514,22.509278,0,20,50
...
```

### Output Format

**Same ordered id values with predictions**

Output columns:
- `id` (int): Matches input id values (same order)
- `pressure` (float): Predicted airway pressure (or NaN if abstain)
- `status` (str): Prediction status ('ok' or 'abstain')
- `reason` (str): Empty if ok, error/warning message if abstain
- `model_version` (str): Model version identifier (e.g., "1.0.0")
- `uncertainty` (float, optional): Uncertainty estimate if requested

**Status Codes:**
- `'ok'`: Prediction completed successfully
- `'abstain'`: Model cannot make reliable prediction (see reason)

**Example Output:**
```csv
id,pressure,status,reason,model_version
1,5.837492,ok,,1.0.0
2,5.907794,ok,,1.0.0
3,7.876254,ok,,1.0.0
...
```

**Abstain Example:**
```csv
id,pressure,status,reason,model_version
1,NaN,abstain,Missing required columns: {'R'},1.0.0
2,NaN,abstain,Missing required columns: {'R'},1.0.0
...
```

## Units and Data Types

### Input Units
| Field | Units | Type | Range/Values |
|-------|-------|------|--------------|
| id | - | int | Positive integers |
| time_step | seconds | float | 0 to ~3 |
| u_in | arbitrary | float | 0 to ~100 |
| u_out | binary | int | 0 or 1 |
| R | category | int | 5, 20, or 50 |
| C | category | int | 10, 20, or 50 |

### Output Units
| Field | Units | Type | Range/Values |
|-------|-------|------|--------------|
| id | - | int | Same as input |
| pressure | cmH2O* | float | ~0 to 60 |
| status | - | str | 'ok' or 'abstain' |
| reason | - | str | Any text |
| model_version | - | str | Semantic version |
| uncertainty | cmH2O* | float | ≥0 (optional) |

*Assumed units based on typical ventilator pressure measurements

## Python Interface

### Class: PressurePredictor

Located in: `src/model_training/inference_wrapper.py`

#### Initialization
```python
from src.model_training.inference_wrapper import PressurePredictor

predictor = PressurePredictor()
predictor.load_model('artifacts/model/final_model.pkl')
```

#### Single Breath Prediction
```python
result_df = predictor.predict_breath(
    breath_df,              # pd.DataFrame with input columns
    return_uncertainty=False # Optional: include uncertainty
)
```

**Parameters:**
- `breath_df` (pd.DataFrame): Input data for one breath
- `return_uncertainty` (bool, default=False): Whether to include uncertainty column

**Returns:**
- `pd.DataFrame`: Predictions with output columns

#### Multiple Breaths Prediction
```python
results_df = predictor.predict_multiple_breaths(
    df,                     # pd.DataFrame with multiple breaths
    breath_id_col='breath_id', # Name of breath ID column
    return_uncertainty=False
)
```

**Parameters:**
- `df` (pd.DataFrame): Input data with multiple breaths
- `breath_id_col` (str): Column name containing breath identifiers
- `return_uncertainty` (bool): Whether to include uncertainty

**Returns:**
- `pd.DataFrame`: Predictions for all breaths

#### Model Information
```python
info = predictor.get_model_info()
```

**Returns:**
- `dict`: Model metadata including version, features, limitations

## Null Handling

### Current Model (v1.0.0)
**R and C nulls are NOT supported**

If R or C contain null values:
- Status: `'abstain'`
- Reason: `"R values contain nulls (required by this model)"` or similar
- Pressure: `NaN`

### Interface Definition
The interface supports R and C being nullable in the schema, but the current benchmark model implementation requires them.

**Rationale:** Future models may handle missing R/C through imputation or estimation, but this benchmark model does not.

**Do NOT silently invent values** - If a required field is missing, return `abstain` status with clear reason.

## Error Handling

### Model Returns Abstain When:
1. Missing required columns
2. Null values in R or C
3. Invalid data types (non-numeric)
4. Model produces non-finite predictions
5. Prediction errors occur during inference

### Abstain Reason Examples:
```
"Missing required columns: {'R', 'C'}"
"R values contain nulls (required by this model)"
"C values contain nulls (required by this model)"
"time_step contains null values"
"Invalid data type in features: ..."
"Model produced non-finite predictions"
"Prediction error: [exception message]"
```

### Warnings (Status Still 'ok'):
```
"Unusual R-C combination: R=15, C=30. Model trained on R=[5,20,50], C=[10,20,50]"
```

## Row Ordering

**CRITICAL:** The app MUST match results by `id`, never assume row order after file operations.

Why:
- File I/O may reorder rows
- Database queries may return unordered results
- Joins and merges may change order

**Always join/merge on the `id` column, not by row position.**

## Example Usage Scenarios

### Scenario 1: Normal Breath
```python
# Load normal breath
breath = pd.read_csv('artifacts/model/mock_breath_example.csv')

# Predict
result = predictor.predict_breath(breath)

# Check status
assert result['status'].iloc[0] == 'ok'

# Use predictions
pressures = result['pressure'].values
```

### Scenario 2: Missing R Value
```python
breath = pd.read_csv('some_breath.csv')
breath['R'] = None  # Missing R

result = predictor.predict_breath(breath)

# Model abstains
assert result['status'].iloc[0] == 'abstain'
assert 'R values contain nulls' in result['reason'].iloc[0]
```

### Scenario 3: Multiple Breaths
```python
# Load multiple breaths
df = pd.read_csv('multiple_breaths.csv')

# Predict all
results = predictor.predict_multiple_breaths(
    df, 
    breath_id_col='breath_id'
)

# Check per-breath status
for breath_id in df['breath_id'].unique():
    breath_results = results[results['id'].isin(
        df[df['breath_id'] == breath_id]['id']
    )]
    status = breath_results['status'].iloc[0]
    print(f"Breath {breath_id}: {status}")
```

## Mock Data for Testing

### Mock Breath Input
File: `artifacts/model/mock_breath_example.csv`
- 80 time steps
- R=20, C=50
- Complete valid breath

### Mock Breath Output
File: `artifacts/model/mock_prediction_example.csv`
- Corresponding predictions
- Status='ok'
- 80 pressure values

**App developers can use these files to test integration without the actual model.**

## Model Version Tracking

The `model_version` field in output allows the app to:
1. Verify which model produced predictions
2. Track model updates over time
3. Validate results match expected model

**Current Version:** `"1.0.0"`

If model is updated:
- Version string will change (e.g., "1.1.0")
- Interface contract remains stable
- New limitations/requirements will be documented

## Performance Expectations

### Inference Speed
- **Single breath (80 samples):** <1ms
- **1000 breaths (80k samples):** ~0.1s
- **50,000 breaths (4M samples):** ~3s

### Memory Requirements
- **Model size:** ~50 MB
- **Runtime memory:** <500 MB for typical use

## Integration Checklist for App Developers

- [ ] Load model using `PressurePredictor` class
- [ ] Test with `mock_breath_example.csv`
- [ ] Verify output matches `mock_prediction_example.csv`
- [ ] Handle `abstain` status gracefully in UI
- [ ] Always join results by `id` column, not row order
- [ ] Display `reason` when status is `abstain`
- [ ] Show `model_version` in app (e.g., "Using model v1.0.0")
- [ ] Test with missing R/C values (should abstain)
- [ ] Test with multiple breaths
- [ ] Verify no stale predictions shown after abstain

## Contact and Support

For interface questions or issues:
1. Check this specification document
2. Review `MODEL_CARD.md` for model details
3. Examine `inference_wrapper.py` source code
4. Test with provided mock examples

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-09-29 | Initial interface specification |

## Future Considerations

Potential future enhancements (not in v1.0.0):
- Support for missing R/C through imputation
- Batch prediction optimizations
- Streaming inference API
- Real-time uncertainty quantification
- Extended breath validation

**Note:** Any changes to the interface will be documented with version updates and maintain backward compatibility where possible.
