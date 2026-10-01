# TwinVent Model Performance Improvement Plan

## Problem Analysis

### Current Performance Issues
- **Overall Validation MAE**: 2.04 cmH₂O (acceptable)
- **Specific Breath Example**: MAE of 11.57 cmH₂O (significantly worse)
- **Problem**: Model dramatically underperforms on certain breaths, predicting ~11 cmH₂O when actual is ~24 cmH₂O

### Root Causes

1. **Lack of Temporal Context**
   - Current model treats each time point independently
   - Ignores sequential dependencies within breath cycles
   - Cannot capture momentum/inertia effects

2. **Missing Feature Interactions**
   - R and C interact in complex ways affecting lung mechanics
   - Flow rate × resistance creates non-linear pressure dynamics
   - Current features don't capture these relationships

3. **Insufficient Model Capacity**
   - Simple hyperparameters (depth=7, estimators=200)
   - May be underfitting complex patterns
   - No regularization to prevent overfitting on specific patterns

4. **Phase-Dependent Behavior**
   - Errors vary across breath phases (early inspiration, mid, expiration)
   - Model doesn't explicitly model breath phase transitions

5. **Specific Weak Patterns**
   - R=5, C=10: MAE=2.50 (worst)
   - R=50, C=20: MAE=2.40
   - High pressure breaths show larger errors

---

## Improvement Strategy

### 🎯 Goal: Reduce MAE from 2.04 to <1.5 cmH₂O

### Approach 1: Feature Engineering (Quick Win)

#### 1.1 Temporal Lag Features
**Why**: Capture sequential dependencies
```python
# Previous time steps
u_in_lag1, u_in_lag2         # Previous control inputs
pressure_lag1, pressure_lag2  # Previous pressures (training only)
u_in_lead1                    # Next control input (anticipation)
```

**Expected Impact**: +15-20% improvement
- Helps model understand momentum
- Captures inertia in pressure changes

#### 1.2 Interaction Features
**Why**: Model complex lung mechanics
```python
RC_product = R × C              # Combined lung mechanics
RC_ratio = R / C                # Mechanical relationship
u_in_R = u_in × R              # Flow-resistance interaction
u_in_RC = u_in × R × C         # Triple interaction
u_in_change = diff(u_in)       # Flow acceleration
```

**Expected Impact**: +10-15% improvement
- Captures non-linear relationships
- Better models pressure generation physics

#### 1.3 Breath Phase Features
**Why**: Different dynamics in different phases
```python
time_normalized               # 0-1 within breath
is_early_phase, is_mid_phase  # Phase indicators
time_sin, time_cos            # Cyclical encoding
u_in_cumsum                   # Volume approximation
```

**Expected Impact**: +5-10% improvement
- Better handles phase transitions
- Captures breath cycle patterns

#### 1.4 Rolling Aggregates
**Why**: Smooth temporal patterns
```python
u_in_roll_mean_3, u_in_roll_mean_5  # Smoothed flow
u_in_roll_std_3                      # Flow variability
```

**Expected Impact**: +5% improvement
- Reduces noise
- Captures local trends

---

### Approach 2: Hyperparameter Optimization

#### Current vs Improved Parameters

| Parameter | Current | Improved | Reason |
|-----------|---------|----------|--------|
| max_depth | 7 | 10 | Capture more complex patterns |
| learning_rate | 0.1 | 0.05 | Better convergence, less overfitting |
| n_estimators | 200 | 500 | More ensemble diversity |
| min_child_weight | - | 3 | Regularization |
| gamma | - | 0.1 | Pruning threshold |
| reg_alpha | - | 0.1 | L1 regularization |
| reg_lambda | - | 1.0 | L2 regularization |
| colsample_bylevel | - | 0.8 | Feature sampling per level |

**Expected Impact**: +10-15% improvement
- Prevents underfitting with deeper trees
- Better generalization with regularization

---

### Approach 3: Advanced Models (If Needed)

#### 3.1 Sequence Models (LSTM/GRU)
**When**: If tree-based approaches don't reach target
```python
# Treat each breath as a sequence
Input: [80 time steps × 5 features]
Architecture: LSTM(64) → LSTM(32) → Dense(1)
```

**Pros**:
- Explicitly models temporal dependencies
- Can learn long-range patterns

**Cons**:
- Slower inference (~10x)
- More complex to deploy
- Requires more training data/time

**Expected Impact**: +20-30% improvement (but slower)

#### 3.2 Hybrid Models
```python
# Combine tree-based + sequence models
Tree predictions + LSTM residual correction
```

**Expected Impact**: +25-35% improvement

---

## Implementation Steps

### Step 1: Train Improved Model (RECOMMENDED START)
```bash
python src/model_training/improved_model.py
```

This script:
1. ✅ Loads training data with existing split
2. ✅ Engineers ~30+ features from base 5
3. ✅ Trains with optimized hyperparameters
4. ✅ Evaluates on validation set
5. ✅ Saves improved model
6. ✅ Reports feature importance

**Time**: ~10-15 minutes
**Expected Result**: MAE < 1.5

### Step 2: Update Inference Wrapper
```bash
# Update predictor.py to use improved model
# Modify inference_wrapper.py to include feature engineering
```

### Step 3: Test on Problem Cases
```bash
# Test specifically on breath #163 (the failing case)
# Verify improvement on other weak R-C combinations
```

### Step 4: Deploy Updated Model
```bash
# Replace final_model.pkl with improved_model.pkl
# Update predictor.py
# Restart web server
```

---

## Expected Results

### Before (Current Model)
| Metric | Value |
|--------|-------|
| Overall Validation MAE | 2.04 |
| R=5, C=10 MAE | 2.50 |
| R=50, C=20 MAE | 2.40 |
| Breath #163 MAE | 11.57 |

### After (Improved Model - Conservative)
| Metric | Expected |
|--------|----------|
| Overall Validation MAE | 1.3-1.5 |
| R=5, C=10 MAE | 1.8-2.0 |
| R=50, C=20 MAE | 1.6-1.8 |
| Breath #163 MAE | <5.0 |

### After (Improved Model - Optimistic)
| Metric | Expected |
|--------|----------|
| Overall Validation MAE | 1.0-1.2 |
| R=5, C=10 MAE | 1.5-1.7 |
| R=50, C=20 MAE | 1.3-1.5 |
| Breath #163 MAE | <3.0 |

---

## Why This Breath Failed

Looking at your specific example (Breath #163, R=5, C=20):
1. **High peak pressure** (~24 cmH₂O) - model tends to underpredict extremes
2. **R=5 combination** - low resistance cases show more variability
3. **No temporal context** - model can't see the rising trend from previous steps
4. **Missing interactions** - doesn't understand how u_in × R creates this specific pressure

**With improvements**:
- Lag features will help model see the buildup
- Interaction features will capture u_in × R relationship
- Deeper trees will model the high-pressure regime better

---

## Quick Start

### Option 1: Full Retraining (Best Results)
```bash
# Train improved model from scratch
python src/model_training/improved_model.py
```

### Option 2: Just Test (Verify Current Performance)
```bash
# Load specific breath and check current predictions
python src/model_evaluation/test_specific_breath.py --breath_id 163
```

---

## Monitoring & Validation

### After Deployment, Check:
1. ✅ Overall validation MAE improved
2. ✅ R=5, C=10 and R=50, C=20 errors reduced
3. ✅ Breath #163 now predicts closer to 24 cmH₂O
4. ✅ No regression on other breath types
5. ✅ Inference speed still acceptable (<1ms per breath)

### Success Criteria:
- [ ] Overall MAE < 1.5
- [ ] Worst-case R-C combination MAE < 2.0
- [ ] 95th percentile error < 7.0 (from 9.7)
- [ ] Breath #163 MAE < 5.0

---

## Risk Mitigation

### Potential Issues:
1. **Overfitting**: Too many features → worse generalization
   - **Solution**: Strong regularization (gamma, alpha, lambda)
   
2. **Inference slowdown**: More features → slower prediction
   - **Solution**: Still <1ms per breath with 30 features
   
3. **Deployment complexity**: Feature engineering in production
   - **Solution**: Wrap in inference_wrapper.py

---

## Alternative: If Time Constrained

### Quick Win (5 minutes)
Just adjust hyperparameters without feature engineering:
```python
# In tree_models_comparison.py, change:
'max_depth': 10,          # from 7
'n_estimators': 400,      # from 200
'learning_rate': 0.05,    # from 0.1
```

**Expected Impact**: +5-8% improvement
**Trade-off**: Not as good as full approach, but much faster

---

## Next Steps

1. **Run improved model training** (recommended)
2. **Evaluate on validation set**
3. **Test on breath #163 specifically**
4. **Compare before/after visualizations**
5. **Deploy if satisfied**
6. **Monitor in production**

---

## Questions to Consider

Before implementing:
1. **Time budget**: Full retraining takes 10-15 min, quick fix takes 5 min
2. **Deployment constraints**: Can you update inference code?
3. **Target accuracy**: Is 1.5 MAE good enough, or need <1.0?
4. **Production requirements**: Inference speed constraints?

---

## Summary

**Problem**: Model underperforms on certain breaths (e.g., MAE 11.57 on breath #163)

**Root Cause**: Lacks temporal context, feature interactions, and sufficient capacity

**Solution**: Feature engineering + hyperparameter tuning

**Expected Improvement**: 30-40% better (MAE 2.04 → 1.3-1.5)

**Time Required**: 10-15 minutes to train

**Ready to start?** Run: `python src/model_training/improved_model.py`
