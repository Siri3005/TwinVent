"""
Owner 1 Benchmark Model - Predictor Interface
Implements the agreed interface contract for the TwinVent offline app
"""

from pathlib import Path
from typing import Any
import pandas as pd
import sys

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_training.inference_wrapper_v2 import ImprovedPressurePredictor


MODEL_VERSION = "2.0.0-improved"

# Initialize and load model
_predictor = ImprovedPressurePredictor()
_model_path = Path(__file__).parent / "final_model.pkl"
_predictor.load_model(str(_model_path))


def predict_breath(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Predict airway pressure for a single breath.
    
    Args:
        rows: List of dictionaries, each containing:
              - id (int): Row identifier
              - time_step (float): Time in seconds
              - u_in (float): Inspiratory flow control
              - u_out (int): Expiratory valve control (0/1)
              - R (int): Resistance parameter
              - C (int): Compliance parameter
    
    Returns:
        Dictionary with:
        - row_ids: List of row IDs (matches input order)
        - pressure: List of predicted pressures (or NaN if abstain)
        - uncertainty: List of uncertainty estimates (or None)
        - status: 'ok' or 'abstain'
        - reason: Error/warning message (empty if ok)
        - model_version: Model version string
    """
    # Convert rows to DataFrame
    breath_df = pd.DataFrame(rows)
    
    # Predict using the wrapper
    result_df = _predictor.predict_breath(breath_df, return_uncertainty=True)
    
    # Convert to the expected dictionary format
    return {
        "row_ids": result_df["id"].tolist(),
        "pressure": result_df["pressure"].tolist(),
        "uncertainty": result_df["uncertainty"].tolist(),
        "status": result_df["status"].iloc[0],
        "reason": result_df["reason"].iloc[0],
        "model_version": MODEL_VERSION
    }
