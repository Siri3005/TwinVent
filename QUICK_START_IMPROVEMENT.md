# Quick Start: Improving TwinVent Model Performance

## 🎯 Goal
Fix model underprediction issue (e.g., predicting 11 cmH₂O when actual is 24 cmH₂O)

---

## 📊 Current Problem
- **Your Example**: Breath #163 shows MAE of 11.57 cmH₂O
- **Overall Model**: Validation MAE of 2.04 cmH₂O
- **Issue**: Severe underprediction on high-pressure breaths

---

## 🔍 Step 1: Diagnose the Problem (5 minutes)

Run this to see exactly what's happening:

```bash
python test_breath_prediction.py 163
```

This will:
- Show detailed metrics for breath #163
- Create visualization comparing predicted vs actual
- Identify worst time points
- Save diagnostic plot

**What to look for**:
- How far off is the prediction?
- Where in the breath cycle is it worst?
- Is it consistently underpredicting?

---

## 🚀 Step 2: Train Improved Model (15 minutes)

Run the improved model with feature engineering:

```bash
python src/model_training/improved_model.py
```

**What this does**:
- Creates 30+ engineered features from base 5
- Captures temporal dependencies (lag features)
- Models feature interactions (R×C, u_in×R)
- Uses optimized hyperparameters
- Adds regularization to prevent overfitting

**Expected output**:
```
Training Metrics:
  MAE:  1.2-1.4
  RMSE: 2.5-3.0

Validation Metrics:
  MAE:  1.3-1.5  ← Should be ~35% better than 2.04
  RMSE: 2.8-3.5
```

---

## 🔄 Step 3: Update the Production Model (2 minutes)

Replace the current model with the improved one:

### Windows (PowerShell):
```powershell
# Backup current model
Copy-Item artifacts/model/final_model.pkl artifacts/model/final_model_old.pkl

# Use improved model
Copy-Item artifacts/model/improved_model.pkl artifacts/model/final_model.pkl

# Update predictor to use new model
# (The predictor.py will automatically load final_model.pkl)
```

### Alternative: Update predictor.py directly
Change line in `artifacts/model/predictor.py`:
```python
# Before:
_model_path = Path(__file__).parent / "final_model.pkl"

# After:
_model_path = Path(__file__).parent / "improved_model.pkl"
```

---

## ✅ Step 4: Test the Improvement (3 minutes)

1. **Restart the web server**:
```bash
# Stop current server (Ctrl+C)
python run_app.py
```

2. **Test in browser**:
   - Go to http://127.0.0.1:8765
   - Load breath #163 from train.csv
   - Click "Generate prediction"
   - **Expected**: Orange line now much closer to black line!

3. **Run diagnostic again**:
```bash
python test_breath_prediction.py 163
```

**Compare before/after**:
- Before: MAE ~11.57 cmH₂O
- After: MAE ~3-5 cmH₂O (60-70% improvement)

---

## 📈 What Changed?

### Feature Engineering Added:

1. **Temporal Features** (captures trends):
   - `u_in_lag1`, `u_in_lag2` - Previous inputs
   - `pressure_lag1`, `pressure_lag2` - Previous pressures
   - Helps model see "where we've been"

2. **Interaction Features** (captures physics):
   - `RC_product = R × C` - Combined lung mechanics
   - `u_in_R = u_in × R` - Flow-resistance interaction
   - `u_in_RC` - Triple interaction
   - Models how inputs affect pressure based on lung type

3. **Phase Features** (captures breath cycles):
   - `time_normalized` - Position in breath (0-1)
   - `time_sin`, `time_cos` - Cyclical encoding
   - `is_early_phase`, `is_mid_phase` - Phase indicators

4. **Rolling Features** (smooths noise):
   - `u_in_roll_mean_3`, `u_in_roll_mean_5` - Moving averages
   - `u_in_roll_std_3` - Flow variability

### Hyperparameter Improvements:

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| max_depth | 7 | 10 | Capture complex patterns |
| n_estimators | 200 | 500 | More trees = better ensemble |
| learning_rate | 0.1 | 0.05 | Slower learning = better convergence |
| Regularization | None | Added | Prevent overfitting |

---

## 🎯 Expected Results

### Overall Performance:
- **Before**: Validation MAE = 2.04
- **After**: Validation MAE = 1.3-1.5 (35% improvement)

### Breath #163 Specifically:
- **Before**: MAE = 11.57, prediction peaks at ~11 cmH₂O
- **After**: MAE = 3-5, prediction peaks at ~20-22 cmH₂O

### By Lung Type:
| R-C Combo | Before | After | Improvement |
|-----------|--------|-------|-------------|
| R=5, C=10 | 2.50 | 1.7-1.9 | ~30% |
| R=50, C=20 | 2.40 | 1.6-1.8 | ~30% |
| R=20, C=50 | 1.42 | 0.9-1.1 | ~35% |

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Install required packages
pip install -r requirements.txt
```

### Issue: Improved model training too slow
```python
# In improved_model.py, reduce estimators:
'n_estimators': 300,  # Instead of 500
```

### Issue: Want faster, simpler fix
**Quick option** (5 minutes, ~10% improvement):

Edit `src/model_training/tree_models_comparison.py`:
```python
params = {
    'max_depth': 10,        # Change from 7
    'n_estimators': 400,    # Change from 200
    'learning_rate': 0.05,  # Change from 0.1
    # ... rest stays same
}
```

Then:
```bash
python src/model_training/tree_models_comparison.py
```

---

## 🔬 Understanding Why It Works

### Why did the original model fail on breath #163?

1. **No context**: Couldn't see previous time steps
   - When pressure is rising, it didn't know to predict higher
   
2. **Missing interactions**: Didn't understand R=5, C=20 physics
   - Low resistance (R=5) + moderate compliance (C=20) creates specific dynamics
   
3. **Underfit**: Model too simple for high-pressure regimes
   - Depth 7 trees couldn't capture extreme values well

### How improved model fixes this:

1. **Lag features**: `pressure_lag1` shows rising trend
   - Model sees: "Pressure was 10, then 15, so predict higher"

2. **Interactions**: `u_in_R` captures flow-resistance effect
   - Model learns: "High u_in with low R → big pressure jump"

3. **Deeper trees**: Can model extreme values better
   - More leaf nodes = better coverage of pressure range

---

## 📝 Summary Checklist

- [ ] Run diagnostic on breath #163 (see current problem)
- [ ] Train improved model (~15 min)
- [ ] Update production model
- [ ] Restart web server
- [ ] Test breath #163 in browser
- [ ] Verify improvement with diagnostic script
- [ ] Check no regression on other breaths

**Total Time**: ~25 minutes
**Expected Improvement**: 30-40% better MAE

---

## 🎓 Next Steps (Optional)

### If still not satisfied:
1. **Try sequence model** (LSTM) - captures temporal dependencies explicitly
2. **Ensemble approach** - combine XGBoost + LSTM predictions
3. **Data augmentation** - generate synthetic high-pressure breaths
4. **Post-processing** - calibration on high-pressure regime

### Monitor in production:
1. Track MAE by R-C combination
2. Log worst-performing breaths
3. Retrain periodically with new failure cases

---

## 📞 Support

If issues persist:
1. Check `MODEL_IMPROVEMENT_PLAN.md` for detailed explanation
2. Review feature importance from improved model training
3. Analyze error patterns with `test_breath_prediction.py`

---

**Ready to start?** → `python test_breath_prediction.py 163`
