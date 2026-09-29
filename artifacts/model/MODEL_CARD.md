# Ventilator Pressure Prediction Model Card

## Model Information

**Model Name:** Ventilator Pressure Prediction Model  
**Version:** 1.0.0  
**Model Type:** XGBoost Regressor  
**Training Date:** 2026-09-29  
**Framework:** XGBoost 3.4.1, scikit-learn 1.9.1

## Intended Use

### Primary Use
This model predicts airway pressure in mechanical ventilator systems based on test lung characteristics and control inputs. It is designed for **benchmarking and educational purposes only** using artificial test lung data.

### Intended Users
- Researchers working with ventilator simulation data
- Developers building offline ventilator analysis tools
- Students learning about ventilator mechanics modeling

### Out-of-Scope Use
- **NOT for clinical decision-making**
- **NOT for patient care or treatment**
- **NOT validated for real patient data**
- **NOT a medical device**

## Training Data

### Dataset
- **Source:** Artificial test lung benchmark dataset
- **Size:** 6,036,000 samples (75,450 breaths)
- **Training Split:** 4,828,800 samples (60,360 breaths, 80%)
- **Validation Split:** 1,207,200 samples (15,090 breaths, 20%)
- **Random Seed:** 42

### Features
The model uses 5 input features:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| R | Resistance (test lung category) | Categorical | 5, 20, 50 |
| C | Compliance (test lung category) | Categorical | 10, 20, 50 |
| time_step | Time within breath cycle | Continuous | 0 to ~3 seconds |
| u_in | Inspiratory flow control | Continuous | 0 to ~100 |
| u_out | Expiratory valve control | Binary | 0 or 1 |

### Target Variable
- **pressure:** Airway pressure measurement (continuous, range: ~0 to 60)

### Data Characteristics
- Each breath contains exactly 80 time samples
- 9 distinct R-C combinations representing different test lung configurations
- Chronological ordering preserved within breaths
- No missing values
- No real patient data or identifiable information

## Model Architecture

### Algorithm
XGBoost (Gradient Boosted Decision Trees)

### Hyperparameters
```python
{
    'objective': 'reg:squarederror',
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'
}
```

### Training Process
1. Data loaded and split by complete breath_id groups (no data leakage)
2. Chronological order maintained within breaths
3. Standard XGBoost training with early stopping on validation set
4. Model selection based on validation MAE

## Performance Metrics

### Overall Performance
| Metric | Training | Validation |
|--------|----------|------------|
| **MAE** | 2.0236 | 2.0382 |
| **RMSE** | 3.9545 | 3.9972 |

### Performance by R-C Group (Validation)
| R | C | MAE | RMSE | Samples |
|---|---|-----|------|---------|
| 5 | 10 | 2.4985 | 4.7942 | 134,080 |
| 5 | 20 | 1.6889 | 2.9130 | 134,720 |
| 5 | 50 | 1.3627 | 2.2366 | 132,400 |
| 20 | 10 | 2.4433 | 4.6614 | 95,280 |
| 20 | 20 | 1.7945 | 3.3570 | 96,320 |
| 20 | 50 | 1.4245 | 2.7917 | 132,560 |
| 50 | 10 | 2.4791 | 4.6336 | 221,120 |
| 50 | 20 | 2.3964 | 4.9651 | 127,440 |
| 50 | 50 | 2.0217 | 4.1747 | 133,280 |

### Error Characteristics
- **Mean Error:** -0.0219 (nearly unbiased)
- **Median Absolute Error:** 0.5841
- **95th Percentile Error:** 9.6991
- **Max Absolute Error:** 40.4950

## Known Limitations and Weaknesses

### Model Limitations
1. **Requires R and C values:** Cannot handle missing/null resistance or compliance values
2. **Artificial data only:** Trained exclusively on test lung data, not real patients
3. **Fixed test lung types:** Performance on R-C combinations outside training set is unknown
4. **No temporal modeling:** Treats each time step independently (no sequence awareness)

### Performance Weaknesses
1. **Higher errors on certain R-C combinations:**
   - R=5, C=10 shows highest error (MAE=2.50)
   - R=50, C=20 also shows elevated error (MAE=2.40)
2. **Extreme value prediction:** Larger errors on breaths with unusual pressure patterns
3. **Phase-dependent accuracy:** Errors vary slightly across different phases of breath cycle

### Known Weak Cases
- Breaths with very high or very low pressure values
- Rapid pressure transitions
- Edge cases at breath start/end
- R=50, C=10 and R=50, C=20 combinations show more variability

## Ethical Considerations

### Safety
- ⚠️ **This model is NOT validated for clinical use**
- ⚠️ **NOT a medical device**
- ⚠️ **Do not use for patient care decisions**
- This is a research/educational tool only

### Bias and Fairness
- Model trained on artificial test lung data only
- No patient demographics, outcomes, or clinical conditions included
- Does not capture individual patient physiology
- Cannot be generalized to real-world patient populations

### Privacy
- No patient data used
- No identifiable information
- Fully synthetic/simulated test lung data

## Model Interface

### Input Format
DataFrame with columns: `id`, `time_step`, `u_in`, `u_out`, `R`, `C`
- Must be sorted by `time_step` within each breath
- All values must be numeric and finite
- R and C are required (no nulls)

### Output Format
DataFrame with columns: `id`, `pressure`, `status`, `reason`, `model_version`
- `pressure`: Predicted pressure value (float)
- `status`: 'ok' or 'abstain'
- `reason`: Empty if ok, error/warning message if abstain
- `model_version`: Model version string (e.g., "1.0.0")

### Usage Example
```python
from inference_wrapper import PressurePredictor

# Initialize and load model
predictor = PressurePredictor()
predictor.load_model('artifacts/model/final_model.pkl')

# Predict for a single breath
result = predictor.predict_breath(breath_df)

# Predict for multiple breaths
results = predictor.predict_multiple_breaths(df, breath_id_col='breath_id')
```

## Model Files

### Artifacts
- `final_model.pkl` - Trained XGBoost model (primary)
- `best_model_xgboost.pkl` - Same model (backup reference)
- `baseline_model.pkl` - Simple baseline model for comparison
- `mock_breath_example.csv` - Example input data
- `mock_prediction_example.csv` - Example output data
- `submission.csv` - Test set predictions

### Code
- `inference_wrapper.py` - Python inference interface
- `baseline_model.py` - Baseline model training code
- `tree_models_comparison.py` - Model comparison and selection

### Documentation
- `MODEL_CARD.md` - This file
- `model_comparison_report.txt` - Model selection rationale
- `error_analysis_report.txt` - Detailed error analysis
- `test_prediction_report.txt` - Test set validation

## Maintenance and Updates

### Version History
- **v1.0.0** (2026-09-29): Initial release
  - XGBoost regressor
  - Trained on artificial test lung data
  - MAE: 2.04 on validation set

### Contact
For questions about this model, refer to the project documentation.

### Reproducibility
All training data, code, and random seeds are documented. Training can be reproduced by:
1. Using the provided training data (train.csv)
2. Running the data splitting script with seed=42
3. Running the model training script with documented hyperparameters

## Disclaimer

**THIS MODEL IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

This model:
- Is NOT a medical device
- Has NOT been validated for clinical use
- Should NOT be used for patient care
- Makes predictions on artificial test lung data only
- Does NOT capture real patient physiology
- Requires additional validation before any clinical application

Any use of this model in medical settings requires:
- Regulatory approval
- Clinical validation studies
- Patient safety protocols
- Medical professional oversight
- Compliance with applicable medical device regulations
