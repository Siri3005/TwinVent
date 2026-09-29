# TwinVent - Ventilator Pressure Prediction Model

**Workflow 1: Model and Data - COMPLETE**

A machine learning model for predicting airway pressure in mechanical ventilator systems using artificial test lung data.

## ⚠️ Important Notice

**This model is NOT validated for patient care or clinical decision-making.**  
This is a research and educational tool trained on artificial test lung benchmark data only.

## Project Overview

This project implements a complete machine learning pipeline for ventilator pressure prediction:

1. ✅ **Data Audit** - Comprehensive validation of training and test datasets
2. ✅ **Safe Split** - Breath-level train/validation split with no data leakage
3. ✅ **Baseline Model** - Simple Ridge regression baseline (MAE: 3.96)
4. ✅ **Better Model** - Advanced XGBoost model selected (MAE: 2.04)
5. ✅ **Error Review** - Detailed error analysis with visualizations
6. ✅ **Test Prediction** - Complete test set predictions generated
7. ✅ **Package** - Inference wrapper and model bundle ready for deployment

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Using the Model

```python
from src.model_training.inference_wrapper import PressurePredictor
import pandas as pd

# Initialize predictor
predictor = PressurePredictor()
predictor.load_model('artifacts/model/final_model.pkl')

# Load breath data
breath_df = pd.read_csv('artifacts/model/mock_breath_example.csv')

# Predict
result = predictor.predict_breath(breath_df, return_uncertainty=True)

print(result[['id', 'pressure', 'status', 'model_version']])
```

## Model Performance

| Metric | Training | Validation |
|--------|----------|------------|
| **MAE** | 2.02 | 2.04 |
| **RMSE** | 3.95 | 4.00 |

**Chosen Model:** XGBoost Regressor
- Selected for lowest validation MAE
- Good generalization (minimal train/val gap)
- Fast inference (3.16s for 4M predictions)

## Project Structure

```
ventilator-pressure-prediction/
├── src/
│   ├── model_data/           # Data loading and splitting
│   │   ├── data_loader.py    # Dataset validation and audit
│   │   ├── data_splitter.py  # Train/val split by breath_id
│   │   ├── train_breath_ids.npy
│   │   └── val_breath_ids.npy
│   ├── model_training/       # Model training scripts
│   │   ├── baseline_model.py        # Baseline Ridge model
│   │   ├── tree_models_comparison.py # XGBoost vs LightGBM
│   │   └── inference_wrapper.py     # Production interface
│   └── model_evaluation/     # Model evaluation
│       ├── error_analysis.py        # Error visualization
│       └── test_prediction.py       # Test set predictions
├── artifacts/
│   └── model/                # Trained models and outputs
│       ├── final_model.pkl           # Production model
│       ├── MODEL_CARD.md             # Model documentation
│       ├── submission.csv            # Test predictions
│       ├── mock_breath_example.csv   # Example input
│       └── mock_prediction_example.csv
├── reports/
│   └── model/                # Analysis reports
│       ├── data_audit_report.json
│       ├── split_report.txt
│       ├── baseline_report.txt
│       ├── model_comparison_report.txt
│       ├── error_analysis_report.txt
│       ├── sample_breaths.png
│       ├── error_by_rc_group.png
│       ├── error_distribution.png
│       └── worst_predictions.png
├── train.csv                 # Training data (6M rows)
├── test.csv                  # Test data (4M rows)
├── sample_submission.csv     # Submission format
└── requirements.txt          # Python dependencies
```

## Dataset

### Training Data
- **Rows:** 6,036,000
- **Breaths:** 75,450
- **Samples per breath:** 80
- **Features:** R, C, time_step, u_in, u_out
- **Target:** pressure

### R-C Combinations
9 distinct test lung configurations:
- **R (Resistance):** 5, 20, 50
- **C (Compliance):** 10, 20, 50

## Model Interface

### Input Schema
```python
{
    'id': int,           # Row identifier
    'time_step': float,  # Time in seconds
    'u_in': float,       # Inspiratory control (0-100)
    'u_out': int,        # Expiratory valve (0 or 1)
    'R': int,            # Resistance (5, 20, or 50)
    'C': int             # Compliance (10, 20, or 50)
}
```

**Requirements:**
- Rows must be sorted by `time_step` within each breath
- R and C cannot be null (required by this model)
- All values must be numeric and finite

### Output Schema
```python
{
    'id': int,              # Same as input
    'pressure': float,      # Predicted pressure
    'status': str,          # 'ok' or 'abstain'
    'reason': str,          # Error/warning message (if any)
    'model_version': str,   # e.g., "1.0.0"
    'uncertainty': float    # Optional, if requested
}
```

**Status Codes:**
- `ok` - Prediction successful
- `abstain` - Model cannot make reliable prediction (reason provided)

## Running the Pipeline

### Step 1: Data Audit
```bash
python src/model_data/data_loader.py
```
Output: `reports/model/data_audit_report.json`

### Step 2: Create Split
```bash
python src/model_data/data_splitter.py
```
Output: `src/model_data/train_breath_ids.npy`, `val_breath_ids.npy`

### Step 3: Train Baseline
```bash
python src/model_training/baseline_model.py
```
Output: `artifacts/model/baseline_model.pkl`

### Step 4: Compare Models
```bash
python src/model_training/tree_models_comparison.py
```
Output: `artifacts/model/final_model.pkl`

### Step 5: Error Analysis
```bash
python src/model_evaluation/error_analysis.py
```
Output: `reports/model/error_*.png`, error reports

### Step 6: Test Prediction
```bash
python src/model_evaluation/test_prediction.py
```
Output: `submission.csv`

### Step 7: Test Interface
```bash
python src/model_training/inference_wrapper.py
```
Output: Mock examples in `artifacts/model/`

## Model Limitations

### What This Model CAN Do
✅ Predict pressure on artificial test lung benchmark data  
✅ Handle standard R-C combinations (trained range)  
✅ Provide abstain status for invalid inputs  
✅ Fast inference (~3 seconds for 4M samples)

### What This Model CANNOT Do
❌ Work without R and C values (nulls not supported)  
❌ Predict for real patients (not validated)  
❌ Handle R-C values outside training range reliably  
❌ Model temporal dependencies (no sequence awareness)  
❌ Provide guaranteed accuracy on edge cases

### Known Weak Cases
- R=5, C=10 combinations (higher error)
- R=50, C=20 combinations (higher error)
- Extreme pressure values
- Rapid pressure transitions
- Breath start/end edge cases

## Error Analysis Summary

### Error Distribution
- Mean Error: -0.02 (nearly unbiased)
- Median Absolute Error: 0.58
- 95th Percentile Error: 9.70
- Max Absolute Error: 40.50

### Best Performance
- R=5, C=50: MAE=1.36
- R=20, C=50: MAE=1.42

### Worst Performance
- R=5, C=10: MAE=2.50
- R=20, C=10: MAE=2.44
- R=50, C=20: MAE=2.40

## Reproducibility

All results are reproducible with:
- **Random Seed:** 42 (fixed across all scripts)
- **Split:** 80% train, 20% validation (by breath_id)
- **Model:** XGBoost with documented hyperparameters
- **Data:** Original train.csv unchanged

## Documentation

- **Model Card:** `artifacts/model/MODEL_CARD.md` - Complete model documentation
- **Data Dictionary:** `reports/model/data_dictionary.json`
- **Audit Report:** `reports/model/data_audit_report.json`
- **Split Report:** `reports/model/split_report.txt`
- **Model Comparison:** `reports/model/model_comparison_report.txt`
- **Error Analysis:** `reports/model/error_analysis_report.txt`

## Dependencies

See `requirements.txt` for full list. Key packages:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- xgboost >= 1.7.0
- lightgbm >= 3.3.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## Version

**Model Version:** 1.0.0  
**Release Date:** 2026-09-29

## License & Disclaimer

**RESEARCH AND EDUCATIONAL USE ONLY**

This model:
- Is NOT a medical device
- Is NOT validated for clinical use
- Should NOT be used for patient care
- Predicts on artificial test lung data only

Any clinical application requires regulatory approval, validation studies, and medical oversight.

## Next Steps (Workflow 2)

Integration with offline replay application:
1. Connect inference wrapper to app predictor adapter
2. Test with fixture breaths
3. Validate output alignment by ID
4. Demonstrate normal, training, test, and abstain cases

## Contact

Refer to project documentation and Workflow 1 specification for details.
