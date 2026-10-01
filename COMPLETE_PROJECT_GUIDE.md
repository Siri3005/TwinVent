# TwinVent: Ventilator Pressure Prediction System
## Complete Technical Documentation & Study Guide

**Project**: AI-Powered Ventilator Pressure Prediction with Offline Replay Application  
**Domain**: Medical Device Simulation, Machine Learning, Time Series Prediction  
**Tech Stack**: Python, TensorFlow/Keras, XGBoost, Pandas, Flask, HTML/CSS/JavaScript  
**Date**: October 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training & Evaluation](#7-model-training--evaluation)
8. [Web Application](#8-web-application)
9. [Integration & Deployment](#9-integration--deployment)
10. [Technical Achievements](#10-technical-achievements)
11. [Key Learnings](#11-key-learnings)
12. [Code Walkthrough](#12-code-walkthrough)

---

## 1. Project Overview

### 1.1 What is TwinVent?

TwinVent is an **offline ventilator pressure prediction system** that uses machine learning to predict airway pressure in mechanical ventilators based on test lung data. It consists of:

1. **ML Models**: Predict pressure from ventilator control inputs
2. **Web Application**: Visualize predictions vs measured values
3. **Prediction Adapter**: Bridge between models and application

### 1.2 Project Goals

✅ **Primary Goal**: Build accurate ML models for pressure prediction  
✅ **Secondary Goal**: Create offline replay system for demonstration  
✅ **Tertiary Goal**: Compare multiple ML approaches (XGBoost vs LSTM)

### 1.3 Key Constraints

⚠️ **NOT for clinical use** - Research/educational only  
⚠️ **Artificial test lung data** - Not real patient data  
⚠️ **Offline operation** - No live ventilator connection

---

## 2. Problem Statement

### 2.1 The Challenge

**Given**: Ventilator control inputs and lung parameters  
**Predict**: Airway pressure at each timestep

### 2.2 Input Features (Base)

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `R` | Respiratory resistance | cmH₂O/L/s | {5, 20, 50} |
| `C` | Lung compliance | mL/cmH₂O | {10, 20, 50} |
| `time_step` | Time in breath cycle | seconds | 0.0 - ~3.0 |
| `u_in` | Inspiratory control | % | 0 - 100 |
| `u_out` | Expiratory valve | binary | {0, 1} |

### 2.3 Output

**Target**: `pressure` (airway pressure in cmH₂O)

### 2.4 Data Structure

- **Breaths**: 75,450 in training set
- **Samples per breath**: 80 timesteps
- **Total training samples**: 6,036,000 rows
- **Test samples**: 4,024,000 rows
- **R-C combinations**: 9 different lung configurations

### 2.5 Success Metrics

- **Primary**: Mean Absolute Error (MAE)
- **Secondary**: Root Mean Squared Error (RMSE)
- **Tertiary**: Visual fit quality (predicted vs measured curves)

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TwinVent System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Data       │      │   Machine    │      │   Web    │ │
│  │  Pipeline    │─────▶│   Learning   │─────▶│   App    │ │
│  │              │      │    Models    │      │          │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│       │                     │                      │       │
│       │                     │                      │       │
│  ┌────▼─────┐          ┌───▼────┐           ┌────▼─────┐ │
│  │  CSV     │          │ reference for the TwinVent project. Study each section to understand the implementation details, design decisions, and best practices applied throughout the project.
un_app.py

# Open browser
# Navigate to: http://127.0.0.1:8765
```

### Switch Models
```python
# Edit: src/predictor_adapter/benchmark.py

# For XGBoost:
MODEL_PATH = ROOT / "artifacts" / "model" / "final_model.pkl"
from model_training.inference_wrapper_v2 import ImprovedPressurePredictor

# For LSTM:
MODEL_PATH = ROOT / "artifacts" / "model" / "lstm_model.h5"
from model_training.lstm_inference_wrapper import LSTMPressurePredictor
```

---

**End of Documentation**

This document serves as your complete technical         # Trained models
├── reports/model/              # Analysis reports
├── train.csv, test.csv         # Data files
├── requirements.txt            # Dependencies
└── run_app.py                  # Server entry point
```

---

## Quick Reference Commands

### Train Models
```bash
# Baseline
python src/model_training/baseline_model.py

# XGBoost v2.0
python src/model_training/improved_model.py

# Evaluate
python src/model_evaluation/error_analysis.py
```

### Run Application
```bash
# Start server
python rlogies Used

**Languages**: Python 3.10+, JavaScript, HTML/CSS  
**ML Libraries**: XGBoost, TensorFlow/Keras, scikit-learn  
**Data**: Pandas, NumPy  
**Web**: Flask, Chart.js  
**Tools**: Jupyter, Git, VS Code

### Project Structure

```
TwinVent/
├── src/
│   ├── model_training/         # ML models
│   ├── model_evaluation/       # Testing
│   ├── model_data/             # Data pipeline
│   ├── predictor_adapter/      # Integration layer
│   └── app/                    # Web server
├── artifacts/model/   y, real-time visualization  
✅ **Best Practices**: Version control, testing, documentation, reproducibility

### Final Metrics

| Component | Status | Key Metric |
|-----------|--------|------------|
| **Data Pipeline** | ✅ Complete | 0 data leaks |
| **Baseline Model** | ✅ Complete | MAE 3.96 |
| **XGBoost v2.0** | ✅ Deployed | **MAE 0.64** |
| **LSTM v3.0** | ✅ Deployed | Sequence-to-sequence |
| **Web App** | ✅ Running | http://127.0.0.1:8765 |
| **Integration** | ✅ Complete | Adapter pattern |

### Techno({
            'status': 'error',
            'reason': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8765)
```

---

## Summary

### Project Highlights

✅ **Data Pipeline**: Robust validation, proper splitting, no leakage  
✅ **Model Development**: 3 models, 84% improvement over baseline  
✅ **Feature Engineering**: Domain-informed, 23-35 features from 5 base  
✅ **System Architecture**: Clean separation, adapter pattern, modular  
✅ **Web Application**: User-friendl{
            'status': result.status,
            'model_version': result.model_version,
            'predictions': [
                {
                    'id': rid,
                    'pressure': press,
                    'uncertainty': unc
                }
                for rid, press, unc in zip(
                    result.row_ids,
                    result.pressure,
                    result.uncertainty
                )
            ]
        })
    except Exception as e:
        return jsonifyreath(source, breath_id)
    
    return jsonify({
        'breath_id': breath_id,
        'R': breath_data['R'].iloc[0],
        'C': breath_data['C'].iloc[0],
        'samples': breath_data.to_dict('records')
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate prediction for breath"""
    data = request.json
    breath_data = data['breath_data']  # List of dicts
    
    # Call predictor
    try:
        result = predictor.predict_breath(breath_data)
        
        return jsonify(om flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load predictor once at startup
from src.predictor_adapter.benchmark import BenchmarkModelAdapter
predictor = BenchmarkModelAdapter()

@app.route('/api/load_breath', methods=['POST'])
def load_breath():
    """Load breath data from CSV"""
    data = request.json
    source = data['source']  # 'train' or 'test'
    breath_id = data['breath_id']
    
    # Load from CSV using cached index
    breath_data = dataset_store.load_b         if return_uncertainty:
                result_df['uncertainty'] = np.abs(predictions) * 0.05
            
            return result_df
            
        except Exception as e:
            return self._create_abstain_result(breath_df, [f"Prediction error: {e}"])

# Usage
predictor = ImprovedPressurePredictor()
predictor.load_model('artifacts/model/final_model.pkl')
result = predictor.predict_breath(breath_df)
```

### 12.3 Web Server

**Entry Point**: `run_app.py` → `src/app/server.py`

```python
frath_df)
        except Exception as e:
            return self._create_abstain_result(breath_df, [f"Feature error: {e}"])
        
        # Predict
        try:
            X = df_features.values
            predictions = self.model.predict(X)
            
            result_df = pd.DataFrame({
                'id': breath_df['id'],
                'pressure': predictions,
                'status': 'ok',
                'reason': '',
                'model_version': self.VERSION
            })
            
   return engineered_df
    
    def predict_breath(self, breath_df, return_uncertainty=False):
        """Predict pressure for a breath"""
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        # Validate
        validation = self.validate_input(breath_df)
        if not validation['valid']:
            return self._create_abstain_result(breath_df, validation['errors'])
        
        # Engineer features
        try:
            df_features = self.engineer_features(bre {col} contains null values")
        
        # Check chronological ordering
        if not breath_df['time_step'].is_monotonic_increasing:
            warnings.append("time_step is not monotonically increasing")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def engineer_features(self, df):
        """Apply same feature engineering as training"""
        # [Same implementation as training pipeline]
        rue
    
    def validate_input(self, breath_df):
        """Validate input DataFrame"""
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_BASE_FEATURES + ['id']) - set(breath_df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for nulls
        for col in self.REQUIRED_BASE_FEATURES:
            if breath_df[col].isnull().any():
                errors.append(f"Columne

**Entry Point**: `src/model_training/inference_wrapper_v2.py`

```python
class ImprovedPressurePredictor:
    VERSION = "2.0.0-improved"
    REQUIRED_BASE_FEATURES = ['R', 'C', 'time_step', 'u_in', 'u_out']
    
    def __init__(self, model_path=None):
        self.model = None
        self.loaded = False
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = joblib.load(model_path)
        self.loaded = T)
        
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        
        return {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'n_features': X_train.shape[1]
        }

# Usage
if __name__ == "__main__":
    model = ImprovedPressureModel()
    train_df, val_df = model.load_data_with_split()
    results = model.train(train_df, val_df)
    model.save_model('artifacts/model/improved_model.pkl')
```

### 12.2 Inference Pipelin.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_predget_feature_columns(train_features)]
        y_train = train_df[self.target_column]
        
        X_val = val_features[self.get_feature_columns(val_features)]
        y_val = val_df[self.target_column]
        
        # Train XGBoost
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            max_depth=10,
            learning_rate=0.05,
            n_estimators=500,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0e engineering"""
        df = self.create_interaction_features(df)
        df = self.create_phase_features(df)
        df = self.create_lag_features(df)
        df = self.create_aggregation_features(df)
        return df
    
    def train(self, train_df, val_df):
        """Train the model"""
        # Engineer features
        train_features = self.engineer_features(train_df)
        val_features = self.engineer_features(val_df)
        
        # Get feature columns
        X_train = train_features[self.['time_normalized'] = df.groupby('breath_id')['time_step'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )
        df['time_sin'] = np.sin(2 * np.pi * df['time_normalized'])
        df['time_cos'] = np.cos(2 * np.pi * df['time_normalized'])
        df['is_early_phase'] = (df['time_normalized'] < 0.3).astype(int)
        df['is_late_phase'] = (df['time_normalized'] >= 0.7).astype(int)
        return df
    
    def engineer_features(self, df):
        """Apply all featurbreath_id')[col].shift(2).fillna(0)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        df['RC_product'] = df['R'] * df['C']
        df['RC_ratio'] = df['R'] / (df['C'] + 1e-6)
        df['u_in_R'] = df['u_in'] * df['R']
        df['u_in_C'] = df['u_in'] * df['C']
        df['u_in_RC'] = df['u_in'] * df['RC_product']
        return df
    
    def create_phase_features(self, df):
        """Create breath phase features"""
        df data_dir='.'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.base_features = ['R', 'C', 'time_step', 'u_in', 'u_out']
        self.target_column = 'pressure'
    
    def create_lag_features(self, df):
        """Create lag features within each breath"""
        df = df.sort_values(['breath_id', 'time_step'])
        
        for col in ['u_in', 'u_out']:
            df[f'{col}_lag1'] = df.groupby('breath_id')[col].shift(1).fillna(0)
            df[f'{col}_lag2'] = df.groupby('s pressure rise

**Takeaway**: Domain expertise guides feature engineering

#### Learning 13: Different Lung Configurations Behave Differently
- High compliance (C=50): Easier to predict
- Low compliance (C=10): More challenging
- Physical intuition matches model performance

**Takeaway**: Validate model behavior against domain knowledge

---

## 12. Code Walkthrough

### 12.1 Training Pipeline

**Entry Point**: `src/model_training/improved_model.py`

```python
class ImprovedPressureModel:
    def __init__(self,ation Issues
- Clear interface definitions
- Input/output specifications
- Error handling standards

**Takeaway**: Define contracts before implementation

#### Learning 11: Logging and Monitoring Essential
- Session logs track usage
- Error logs aid debugging
- Performance metrics guide optimization

**Takeaway**: Build observability from day one

### 11.4 Domain Knowledge

#### Learning 12: Respiratory Mechanics Inform Features
- R and C interact non-linearly
- Flow × resistance = pressure drop
- Compliance affectMatch architecture to problem structure

#### Learning 8: Bidirectional Processing Helps
- Forward LSTM sees past
- Backward LSTM sees future
- Together: full context for each timestep

**Takeaway**: Use when you have complete sequences

### 11.3 Software Engineering

#### Learning 9: Clean Architecture Enables Flexibility
- Swapped models without changing app
- Easy to add new features
- Simple to test components

**Takeaway**: Design for change from the start

#### Learning 10: API Contracts Prevent Integr- Held-out breaths are true test

**Takeaway**: Understand your data structure before splitting

### 11.2 Deep Learning

#### Learning 6: LSTMs Need More Data/Time
- Training time: Hours vs minutes for XGBoost
- More parameters: 922K vs 23K
- Requires careful tuning

**Takeaway**: Use when temporal patterns are complex

#### Learning 7: Sequence-to-Sequence Architecture
- LSTM outputs 80 values (one per timestep)
- More natural for time series than single output
- Better captures waveform shape

**Takeaway**: em inertia
- Phase features handled breath cycle dynamics
- Rolling features reduced noise

**Takeaway**: Time series problems need temporal context

#### Learning 4: Hyperparameter Tuning Impact
- Default XGBoost: MAE 2.04
- Tuned XGBoost: MAE 0.64
- **Depth, learning rate, regularization all critical**

**Takeaway**: Don't use default hyperparameters in production

#### Learning 5: Validation Strategy is Critical
- Breath-level splitting prevented data leakage
- Random row splitting would give false accuracy
oost v2 (23 features): MAE 0.64
- **3x better** with same algorithm, just better features

**Takeaway**: Invest time in understanding domain and creating meaningful features

#### Learning 2: Tree-Based Models Excel at Tabular Data
- XGBoost outperformed linear baseline by 48%
- Simple architecture, easy to interpret
- Fast training and inference
- Good generalization

**Takeaway**: Don't immediately jump to deep learning for tabular data

#### Learning 3: Temporal Features Matter
- Lag features captured syst```

#### Adapter Pattern Benefits
- Models can be swapped without changing app
- App can evolve without retraining models
- Clear interface contract
- Easy testing of individual components

#### Caching Strategy
- Breath index cached on disk (`.twinvent/cache/`)
- Model loaded once (singleton pattern)
- Session data in memory
- **Result**: Fast response times

---

## 11. Key Learnings

### 11.1 Machine Learning

#### Learning 1: Feature Engineering > Model Complexity
- XGBoost v1 (5 features): MAE 2.04
- XGB End-to-end validation

### 10.3 System Design

#### Separation of Concerns
```
Data Layer ─────▶ Model Layer ─────▶ Application Layer
  (CSV)           (ML Models)          (Web UI)
    │                  │                   │
    │                  │                   │
Independent        Independent         Independent
Can change         Can change          Can change
without            without             without
affecting          affecting           affecting
others             others              others
ure information in training
- Validation on held-out breaths

#### ✅ Reproducibility
- Fixed random seed (42)
- Saved split indices
- Documented hyperparameters
- Version-controlled code

#### ✅ Model Versioning
- Clear version numbers (v1.0, v2.0, v3.0)
- Model cards with metadata
- Performance tracking over time

#### ✅ Code Quality
- Type hints for clarity
- Docstrings for all functions
- Modular architecture
- Error handling

#### ✅ Testing
- Unit tests for data pipeline
- Integration tests for models
- Baseline MAE: 3.96 cmH₂O
- XGBoost v2.0 MAE: 0.64 cmH₂O
- **Reduction: 3.32 cmH₂O (84% better)**

#### Achievement 2: Consistent Across All R-C Combinations
- Best case (R=5, C=50): MAE 0.41
- Worst case (R=5, C=10): MAE 0.75
- **Variation: <0.35 cmH₂O**

#### Achievement 3: Fast Inference
- XGBoost: ~1ms per breath (80 samples)
- LSTM: ~10ms per breath
- **Real-time capable** for visualization

### 10.2 Engineering Best Practices

#### ✅ Data Leakage Prevention
- Breath-level splitting (not row-level)
- No futodel.pkl artifacts/model/final_model.pkl
```

**Step 3: Update Adapter**
```python
# src/predictor_adapter/benchmark.py
MODEL_PATH = ROOT / "artifacts" / "model" / "final_model.pkl"
MODEL_VERSION = "2.0.0-improved"
```

**Step 4: Start Server**
```bash
python run_app.py
# Server starts on http://127.0.0.1:8765
```

**Step 5: Test**
- Open browser
- Load breath
- Generate prediction
- Verify accuracy

---

## 10. Technical Achievements

### 10.1 Model Performance

#### Achievement 1: 84% Improvement Over Baseline
-d'].tolist(),
            pressure=result['pressure'].tolist(),
            uncertainty=[None] * len(result),
            status=result['status'].iloc[0],
            reason=result['reason'].iloc[0],
            model_version=result['model_version'].iloc[0]
        )
```

### 9.4 Deployment Workflow

**Step 1: Train Model**
```bash
python src/model_training/improved_model.py
# Output: artifacts/model/improved_model.pkl
```

**Step 2: Deploy Model**
```bash
# Replace production model
cp artifacts/model/improved_mnce_wrapper import LSTMPressurePredictor
            cls._predictor = LSTMPressurePredictor()
            cls._predictor.load_model('artifacts/model/lstm_model.h5')
    
    def predict_breath(self, rows):
        # Convert to DataFrame
        df = pd.DataFrame(rows, columns=['id', 'time_step', 'u_in', 'u_out', 'R', 'C'])
        
        # Call predictor
        result = self._predictor.predict_breath(df)
        
        # Convert to API format
        return PredictionResult(
            row_ids=result['if.engineer_features(breath_df)
        
        # Reshape for LSTM: (1, 80, 35)
        X = features.values.reshape(1, 80, 35)
        
        # Predict (returns 80 values)
        predictions = self.model.predict(X)[0]
        
        return result_df
```

### 9.3 Benchmark Adapter

```python
# src/predictor_adapter/benchmark.py

class BenchmarkModelAdapter:
    def __init__(self):
        # Load model once (singleton pattern)
        if cls._predictor is None:
            from model_training.lstm_inferebreath(self, breath_df):
        # Engineer features
        features = self.engineer_features(breath_df)
        
        # Predict
        predictions = self.model.predict(features)
        
        return result_df
```

**LSTM**:
```python
class LSTMPressurePredictor:
    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)  # Load .h5
        self.loaded = True
    
    def predict_breath(self, breath_df):
        # Engineer features (35 features)
        features = sel 
        Returns:
            PredictionResult with:
            - row_ids: List[int]
            - pressure: List[float]
            - uncertainty: List[float]
            - status: str ('ok', 'abstain', 'error')
            - reason: str
            - model_version: str
        """
```

### 9.2 Model Loading

**XGBoost**:
```python
class ImprovedPressurePredictor:
    def load_model(self, model_path):
        self.model = joblib.load(model_path)  # Load .pkl
        self.loaded = True
    
    def predict_',
    body: JSON.stringify({breath_data: currentBreath})
  });
  
  const predictions = await response.json();
  updateChart(predictions);
}
```

---

## 9. Integration & Deployment

### 9.1 Predictor Adapter Pattern

**Purpose**: Decouple ML models from web application

**Interface Contract**:
```python
class Predictor(Protocol):
    def predict_breath(self, rows: List[Dict]) -> PredictionResult:
        """
        Args:
            rows: List of dicts with keys [id, time_step, u_in, u_out, R, C]
       **JavaScript Logic** (`app.js`):
```javascript
async function loadBreath() {
  const source = document.getElementById('datasetSelect').value;
  const breathId = document.getElementById('breathId').value;
  
  const response = await fetch('/api/load_breath', {
    method: 'POST',
    body: JSON.stringify({source, breath_id: breathId})
  });
  
  const data = await response.json();
  renderCharts(data);
}

async function generatePrediction() {
  const response = await fetch('/api/predict', {
    method: 'POSTcontainer">
  <!-- Dataset selection -->
  <select id="datasetSelect">
    <option value="train">train.csv</option>
    <option value="test">test.csv</option>
  </select>
  
  <!-- Breath ID input -->
  <input type="number" id="breathId" />
  <button onclick="loadBreath()">Load Breath</button>
  
  <!-- Prediction controls -->
  <button onclick="generatePrediction()">Generate Prediction</button>
  
  <!-- Visualization -->
  <canvas id="pressureChart"></canvas>
  <canvas id="inputChart"></canvas>
</div>
```

": 5,
  "C": 20,
  "samples": [
    {"id": 1, "time_step": 0.0, "u_in": 0.0, "u_out": 0, "pressure": 5.2},
    ...80 samples...
  ]
}
```

#### POST `/api/predict`
```json
Request: {
  "breath_data": [...80 samples...],
  "model": "benchmark"  // or "mock"
}

Response: {
  "status": "ok",
  "model_version": "3.0.0-lstm",
  "predictions": [
    {"id": 1, "pressure": 5.18, "uncertainty": 0.26},
    ...80 predictions...
  ]
}
```

### 8.4 Frontend Components

**HTML Structure** (`index.html`):
```html
<div class="ualization
- **Chart.js** for interactive plots
- **Black line**: Measured pressure
- **Orange line**: Model predictions
- **Green line**: Control inputs (u_in)
- **Gray bars**: Valve state (u_out)

### 8.3 API Endpoints

#### GET `/api/metadata`
```json
{
  "source": "train",
  "total_breaths": 75450,
  "breath_ids": [0, 1, 2, ..., 75449],
  "rc_combinations": [[5,10], [5,20], ...]
}
```

#### POST `/api/load_breath`
```json
Request: {"source": "train", "breath_id": 163}

Response: {
  "breath_id": 163,
  "Rld_breath_index('train.csv')
# Cached for fast subsequent lookups
```

#### 8.2.2 Breath Loading
```python
# Load specific breath by ID
breath_data = load_breath_by_id(
    source='train',  # or 'test'
    breath_id=163,
    use_cache=True
)
# Returns 80 rows for that breath
```

#### 8.2.3 Prediction Generation
```python
# User clicks "Generate prediction"
# Backend calls predictor adapter
result = predictor.predict_breath(breath_df)
# Returns: {id, pressure, status, reason, model_version}
```

#### 8.2.4 Vis───┐
│         ML Models                           │
│  ┌─────────────────────────────────────┐   │
│  │  XGBoost (.pkl)    LSTM (.h5)       │   │
│  │  - Load weights                     │   │
│  │  - Generate predictions             │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### 8.2 Key Features

#### 8.2.1 Dataset Selection & Indexing
```python
# First time: Build index of breath IDs
# Format: {breath_id: {start_row, end_row, R, C}}
index = bui             │   │
│  └────────────┬────────────────────────┘   │
│               │                             │
│  ┌────────────▼────────────────────────┐   │
│  │  Predictor Adapter                  │   │
│  │  - Model loading                    │   │
│  │  - Feature engineering              │   │
│  │  - Prediction generation            │   │
│  └────────────┬────────────────────────┘   │
└───────────────┼─────────────────────────────┘
                │
                ↓
┌──────────────────────────────────────────        │   │
│  │  - Visualization (Chart.js)         │   │
│  └────────────┬────────────────────────┘   │
└───────────────┼─────────────────────────────┘
                │ HTTP/JSON
                ↓
┌─────────────────────────────────────────────┐
│        Flask Server (Backend)               │
│  ┌─────────────────────────────────────┐   │
│  │  server.py                          │   │
│  │  - REST API endpoints               │   │
│  │  - CSV indexing                     │   │
│  │  - Session management   across resistance values

**Error Distribution**:
- Mean error: -0.02 (nearly unbiased)
- Median absolute error: 0.58
- 95th percentile: 9.70
- Max error: 40.50 (rare outliers)

---

## 8. Web Application

### 8.1 Application Architecture

```
┌─────────────────────────────────────────────┐
│           Browser (Client)                  │
│  ┌─────────────────────────────────────┐   │
│  │  HTML/CSS/JavaScript                │   │
│  │  - File selection                   │   │
│  │  - Breath loading            | MAE | RMSE | Performance |
|---|---|-----|------|-------------|
| 5 | 50 | 0.41 | 0.68 | ⭐ Best |
| 20 | 50 | 0.43 | 0.72 | ⭐ Excellent |
| 50 | 50 | 0.61 | 1.03 | ✓ Good |
| 20 | 20 | 0.65 | 1.10 | ✓ Good |
| 5 | 20 | 0.70 | 1.18 | ✓ Good |
| 50 | 10 | 0.67 | 1.13 | ✓ Good |
| 20 | 10 | 0.73 | 1.23 | ⚠ Average |
| 5 | 10 | 0.75 | 1.27 | ⚠ Challenging |
| 50 | 20 | 0.72 | 1.21 | ⚠ Challenging |

**Observations**:
- Best performance: High compliance (C=50)
- Challenging cases: Low compliance (C=10)
- Consistent=512,
    callbacks=[
        EarlyStopping(patience=260, monitor='val_loss', restore_best_weights=True)
    ]
)
```

**Key Characteristics**:
- **Sequence-to-sequence**: Outputs 80 pressure values (one per timestep)
- **Bidirectional processing**: Sees both past and future context
- **CNN preprocessing**: Extracts local patterns before LSTM
- **922,775 parameters** (3.52 MB model size)

**Results**: Under evaluation (visually appears accurate)

### 7.5 Error Analysis

**By R-C Combination** (XGBoost v2.0):

| R | C  
    # Bidirectional LSTM
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    
    # Output
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.5),
    layers.Dense(80)  # 80 outputs (one per timestep)
])

model.compile(
    optimizer='adam',
    loss='mae'
)
```

**Features**: 35 engineered features

**Training**:
```python
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=3500,
    batch_size(1.7%) - Compliance parameter

### 7.4 LSTM v3.0 (Deep Learning)

**Purpose**: Capture complex temporal patterns

**Architecture**:
```python
from tensorflow.keras import layers, Sequential

model = Sequential([
    layers.Normalization(input_shape=[80, 35]),
    
    # CNN feature extraction
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(),
    layers.BatchNormalization(),
    
    layers.Conv1D(256, 3, activation='relu'),
    layers.MaxPooling1D(),
    layers.BatchNormalization(),
  in,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)
```

**Results**:
- Training MAE: 0.63
- Validation MAE: **0.64**
- Validation RMSE: **1.12**

**Improvement**: 
- 84% better than baseline (3.96 → 0.64)
- 69% better than v1.0 (2.04 → 0.64)

**Feature Importance (Top 5)**:
1. `u_out_lag2` (72%) - Previous expiratory valve states
2. `time_sin` (11%) - Cyclical breath phase
3. `RC_ratio` (2%) - Resistance/compliance ratio
4. `u_in_cumsum` (1.8%) - Volume approximation
5. `C`  0.1
    n_estimators=500,          # Increased from 200
    min_child_weight=3,        # Added regularization
    subsample=0.8,             # Sample 80% of data
    colsample_bytree=0.8,      # Sample 80% of features
    gamma=0.1,                 # Pruning threshold
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    random_state=42
)
```

**Features**: 23 engineered features

**Training**:
```python
# With early stopping
model.fit(
    X_train, y_tramators=200,
    random_state=42
)
model.fit(X_train, y_train)
```

**Features**: 5 base features only

**Results**:
- Training MAE: 2.02
- Validation MAE: **2.04**
- Validation RMSE: **4.00**

**Improvement**: 48% better than baseline (3.96 → 2.04)

### 7.3 XGBoost v2.0 (Improved)

**Purpose**: Maximize tree-based performance

**Architecture**:
```python
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=10,              # Increased from 7
    learning_rate=0.05,        # Decreased fromr_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

**Features**: 5 base features only

**Results**:
- Training MAE: 3.94
- Validation MAE: **3.96**
- Validation RMSE: **6.40**

**Analysis**: Simple linear model struggles with non-linear pressure dynamics

### 7.2 XGBoost v1.0

**Purpose**: Improve with gradient boosting

**Architecture**:
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=7,
    learning_rate=0.1,
    n_esti) / (x.std() + 1e-8)
    )
    
    # Forward lags (after)
    for i in range(1, 6):
        df[f'u_in_after{i if i > 1 else ""}'] = data['un_in_std'].shift(i)
    
    # Backward lags (back)
    for i in range(1, 6):
        df[f'u_in_back{i if i > 1 else ""}'] = data['un_in_std'].shift(-i)
    
    df.fillna(0, inplace=True)
    return df
```

---

## 7. Model Training & Evaluation

### 7.1 Baseline Model (Ridge Regression)

**Purpose**: Establish performance floor

**Architecture**:
```python
from sklearn.lineaata['u_in'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # Cumulative
    df['u_in_cumsum'] = data['u_in'].cumsum()
    
    return df
```

**LSTM Approach**:
```python
def engineer_features(df):
    df['breath_id'] = 0
    data = df.groupby('breath_id')
    
    # Standardize within breath
    df['un_in_std'] = data['u_in'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df['time_step_std'] = data['time_step'].transform(
        lambda x: (x - x.mean()eath_id'] = 0  # Temporary grouping
    data = df.groupby('breath_id')
    
    # Interactions
    df['RC_product'] = df['R'] * df['C']
    
    # Lags (within breath)
    df['u_in_lag1'] = data['u_in'].shift(1).fillna(0)
    
    # Phase features
    time_range = df['time_step'].max() - df['time_step'].min()
    df['time_normalized'] = (df['time_step'] - df['time_step'].min()) / time_range
    df['time_sin'] = np.sin(2 * np.pi * df['time_normalized'])
    
    # Rolling features
    df['u_in_roll_mean_5'] = dps):
```python
time_step_back, time_step_back2, ..., time_step_back5    # 5 features
u_in_back, u_in_back2, ..., u_in_back5                   # 5 features
u_out_back, u_out_back2, ..., u_out_back5                # 5 features
```

**Why?** 
- LSTM can use both past AND future context (during training)
- 5-step window captures local temporal patterns
- Bidirectional helps with smoother predictions

### 6.3 Feature Engineering Implementation

**XGBoost Approach**:
```python
def engineer_features(df):
    df['br
un_in_std = (u_in - mean_breath) / std_breath
time_step_std = (time_step - mean_breath) / std_breath
```

**Why?** Normalizes each breath independently, helps LSTM convergence

#### Bidirectional Lag Features (30)

**After Features** (previous timesteps):
```python
time_step_after, time_step_after2, ..., time_step_after5  # 5 features
u_in_after, u_in_after2, ..., u_in_after5                # 5 features
u_out_after, u_out_after2, ..., u_out_after5             # 5 features
```

**Back Features** (future timeste
```

**Why?** Reduces noise, captures local trends

#### Cumulative Features (1)
```python
u_in_cumsum = cumulative_sum(u_in)    # Volume approximation
```

**Why?** Integrates flow to estimate volume

#### Time Interactions (2)
```python
u_in_time = u_in × time_step
u_out_time = u_out × time_step
```

**Why?** Captures time-dependent behavior

### 6.2 LSTM v3.0 Features (35 total)

#### Base Features (3)
```python
['R', 'C', 'u_out']
```

#### Standardized Features (2)
```python
# Within-breath standardization
#### Phase Features (5)
```python
time_normalized = (time - min) / (max - min)  # 0-1 scale
time_sin = sin(2π × time_normalized)          # Cyclical encoding
time_cos = cos(2π × time_normalized)          # Cyclical encoding
is_early_phase = (time_normalized < 0.3)      # Inspiration start
is_late_phase = (time_normalized >= 0.7)      # Expiration phase
```

**Why?** Different dynamics in different breath phases

#### Rolling Aggregates (2)
```python
u_in_roll_mean_5 = rolling_mean(u_in, window=5)  # Smoothed flowC = u_in × C                     # Flow-compliance interaction
u_in_RC = u_in × R × C                # Triple interaction
```

**Why?** Models physical relationships in respiratory mechanics

#### Temporal Features (4)
```python
u_in_lag1 = u_in[t-1]                # Previous input
u_in_lag2 = u_in[t-2]                # Two steps back
u_out_lag1 = u_out[t-1]              # Previous valve state
u_out_lag2 = u_out[t-2]              # Two steps back
```

**Why?** Captures temporal dependencies and system inertia
64** | **1.12** | ~280 sec | ~1 ms |
| **LSTM v3.0** | Deep Learning | 35 engineered | TBD | TBD | ~hours | ~10 ms |

---

## 6. Feature Engineering

### 6.1 XGBoost v2.0 Features (23 total)

#### Base Features (5)
```python
['R', 'C', 'time_step', 'u_in', 'u_out']
```

#### Interaction Features (5)
```python
RC_product = R × C                    # Combined lung mechanics
RC_ratio = R / C                      # Resistance/compliance ratio
u_in_R = u_in × R                     # Flow-resistance interaction
u_in_imeline

```
Baseline (Ridge) → XGBoost v1 → XGBoost v2 → LSTM
MAE: 3.96        → MAE: 2.04   → MAE: 0.64   → MAE: TBD
```

### 5.2 Model Comparison Table

| Model | Type | Features | MAE | RMSE | Training Time | Inference |
|-------|------|----------|-----|------|---------------|-----------|
| **Baseline Ridge** | Linear Regression | 5 base | 3.96 | 6.40 | ~1 sec | <1 ms |
| **XGBoost v1.0** | Gradient Boosting | 5 base | 2.04 | 4.00 | ~60 sec | ~1 ms |
| **XGBoost v2.0** | Gradient Boosting | 23 engineered | **0.verlap** between sets

### 4.4 Data Characteristics

**Distribution by R-C**:
| R | C | Breaths | Percentage |
|---|---|---------|------------|
| 5 | 10 | 8,383 | 11.1% |
| 5 | 20 | 8,383 | 11.1% |
| 5 | 50 | 8,384 | 11.1% |
| 20 | 10 | 8,383 | 11.1% |
| 20 | 20 | 8,383 | 11.1% |
| 20 | 50 | 8,384 | 11.1% |
| 50 | 10 | 8,383 | 11.1% |
| 50 | 20 | 8,383 | 11.1% |
| 50 | 50 | 8,384 | 11.1% |

**Balanced dataset** - each configuration equally represented

---

## 5. Machine Learning Models

### 5.1 Model Evolution Tementation**:
```python
# Get unique breath IDs
breath_ids = train_df['breath_id'].unique()

# Split at breath level (80% train, 20% val)
train_breaths, val_breaths = train_test_split(
    breath_ids, 
    test_size=0.2, 
    random_seed=42
)

# Filter data by breath IDs
train_data = train_df[train_df['breath_id'].isin(train_breaths)]
val_data = train_df[train_df['breath_id'].isin(val_breaths)]
```

**Result**:
- Train: 60,360 breaths (4,828,800 samples)
- Validation: 15,090 breaths (1,207,200 samples)
- **Zero oal columns
✓ 80 samples per breath (consistent)
✓ Chronological time_step ordering within breaths
✓ All 9 R-C combinations present
✓ Valid value ranges for each feature
```

**Output**: `data_audit_report.json`, `data_dictionary.json`

### 4.3 Data Splitting (`data_splitter.py`)

**Critical Requirement**: Split by **breath_id**, not by random rows

**Why?**
- Each breath is 80 sequential samples
- Splitting rows randomly would leak information
- Model would "see" parts of test breaths during training

**Impl - `predictor_adapter/` - Model integration

---

## 4. Data Pipeline

### 4.1 Data Flow

```
Raw CSV Files
    ↓
Data Loader (Validation)
    ↓
Feature Engineering
    ↓
Train/Val Split (by breath)
    ↓
Model Training
    ↓
Inference Wrapper
    ↓
Web Application
```

### 4.2 Data Validation (`data_loader.py`)

**Purpose**: Ensure data integrity before training

**Checks Performed**:
```python
✓ All required columns present
✓ Correct number of rows (6,036,000 train, 4,024,000 test)
✓ No null values in criticset_store.py` - Caching and indexing

#### Model Layer
- **Purpose**: Train, evaluate, and serve predictions
- **Components**:
  - `baseline_model.py` - Ridge regression baseline
  - `improved_model.py` - XGBoost with feature engineering
  - `lstm_inference_wrapper.py` - Deep learning model
  - Model artifacts (`.pkl`, `.h5` files)

#### Application Layer
- **Purpose**: User interface and visualization
- **Components**:
  - `server.py` - Flask backend
  - `index.html` - Web UI
  - `app.js` - Frontend logic
  Model  │           │ Browser  │ │
│  │  Files   │          │ Files  │           │ UI       │ │
│  └──────────┘          └────────┘           └──────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Breakdown

#### Data Layer
- **Purpose**: Load, validate, and split data
- **Components**:
  - `data_loader.py` - CSV loading and validation
  - `data_splitter.py` - Train/validation split by breath
  - `data