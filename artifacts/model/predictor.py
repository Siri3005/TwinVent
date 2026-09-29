"""
TwinVent Benchmark Model Predictor
Adapter interface for the app to use the trained XGBoost model
"""

import sys
from pathlib import Path

# Add src to path to import our inference wrapper
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_training.inference_wrapper import PressurePredictor
import pandas as pd

# Model version
MODEL_VERSION = "1.0.0"

# Initialize predictor
_predictor = None


def _get_predictor():
    """Lazy load the predictor"""
    global _predictor
    if _predictor is None:
        model_path = Path(__file__).parent / "final_model.pkl"
        _predictor = PressurePredictor()
        _predictor.load_model(str(model_path))
    return _predictor


def predict_breath(rows):
    """
    Predict pressure for a single breath
    
    Args:
        rows: List of dicts with keys: id, time_step, u_in, u_out, R, C
        
    Returns:
        dict with keys:
            - row_ids: list of int (same order as input)
            - pressure: list of float (or None if abstain)
            - uncertainty: list of float or None (optional)
            - status: 'ok' or 'abstain'
            - reason: str (error message if abstain)
            - model_version: str (e.g., "1.0.0")
    """
    predictor = _get_predictor()
    
    # Convert rows to DataFrame
    df = pd.DataFrame(rows)
    
    # Make sure columns are in the right order
    required_cols = ['id', 'time_step', 'u_in', 'u_out', 'R', 'C']
    for col in required_cols:
        if col not in df.columns:
            # Return abstain status
            return {
                'row_ids': [row['id'] for row in rows],
                'pressure': [None] * len(rows),
                'uncertainty': [None] * len(rows),
                'status': 'abstain',
                'reason': f'Missing required column: {col}',
                'model_version': MODEL_VERSION
            }
    
    # Sort by time_step to maintain chronological order
    df = df.sort_values('time_step')
    
    # Predict
    result_df = predictor.predict_breath(df, return_uncertainty=True)
    
    # Convert back to the expected format
    return {
        'row_ids': result_df['id'].tolist(),
        'pressure': result_df['pressure'].tolist(),
        'uncertainty': result_df.get('uncertainty', [None] * len(result_df)).tolist(),
        'status': result_df['status'].iloc[0] if len(result_df) > 0 else 'abstain',
        'reason': result_df['reason'].iloc[0] if len(result_df) > 0 else '',
        'model_version': result_df['model_version'].iloc[0] if len(result_df) > 0 else MODEL_VERSION
    }


# Test if running directly
if __name__ == "__main__":
    print("Testing predictor adapter...")
    
    # Create a simple test breath
    test_rows = []
    for i in range(80):
        test_rows.append({
            'id': i + 1,
            'time_step': i * 0.025,
            'u_in': 20.0 if i < 40 else 0.0,
            'u_out': 0 if i < 40 else 1,
            'R': 20,
            'C': 50
        })
    
    # Predict
    result = predict_breath(test_rows)
    
    print(f"✓ Prediction successful")
    print(f"  Status: {result['status']}")
    print(f"  Model Version: {result['model_version']}")
    print(f"  Predictions: {len(result['pressure'])} values")
    print(f"  Mean pressure: {sum([p for p in result['pressure'] if p is not None]) / len(result['pressure']):.2f}")
    print(f"  Sample pressures: {result['pressure'][:5]}")
