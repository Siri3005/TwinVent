"""
Inference Wrapper for Ventilator Pressure Prediction
Provides a stable interface for the offline app to use the model
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Union, Optional
import json


class PressurePredictor:
    """
    Inference wrapper for ventilator pressure prediction model
    
    Interface Contract:
    -------------------
    Input: One breath as ordered rows with columns: id, time_step, u_in, u_out, R, C
    Output: Same ordered id values with predicted pressure, status, and optional uncertainty
    
    Status codes:
    - 'ok': Prediction successful
    - 'abstain': Model cannot make reliable prediction
    
    Version: 1.0.0
    Model Type: XGBoost Regressor
    """
    
    VERSION = "1.0.0"
    MODEL_TYPE = "XGBoost"
    REQUIRED_FEATURES = ['R', 'C', 'time_step', 'u_in', 'u_out']
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to the trained model file. If None, uses default path.
        """
        self.model = None
        self.model_path = model_path
        self.loaded = False
        
    def load_model(self, model_path: str = None):
        """
        Load the trained model
        
        Args:
            model_path: Path to model file. If None, uses default.
        """
        if model_path is None:
            model_path = self.model_path or 'artifacts/model/final_model.pkl'
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        self.loaded = True
        return self
    
    def validate_input(self, breath_df: pd.DataFrame) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate input breath data
        
        Args:
            breath_df: DataFrame with breath data
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        missing_cols = set(self.REQUIRED_FEATURES + ['id']) - set(breath_df.columns)
        if missing_cols:
            validation['valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for nulls in R and C (required by this model)
        if 'R' in breath_df.columns and breath_df['R'].isnull().any():
            validation['valid'] = False
            validation['errors'].append("R values contain nulls (required by this model)")
        
        if 'C' in breath_df.columns and breath_df['C'].isnull().any():
            validation['valid'] = False
            validation['errors'].append("C values contain nulls (required by this model)")
        
        # Check for nulls in other features
        for col in ['time_step', 'u_in', 'u_out']:
            if col in breath_df.columns and breath_df[col].isnull().any():
                validation['valid'] = False
                validation['errors'].append(f"{col} contains null values")
        
        # Check data types
        if validation['valid']:
            try:
                for col in self.REQUIRED_FEATURES:
                    breath_df[col].astype(float)
            except (ValueError, TypeError) as e:
                validation['valid'] = False
                validation['errors'].append(f"Invalid data type in features: {str(e)}")
        
        # Warning for unusual R-C combinations
        if 'R' in breath_df.columns and 'C' in breath_df.columns:
            r_val = breath_df['R'].iloc[0]
            c_val = breath_df['C'].iloc[0]
            
            expected_r = [5, 20, 50]
            expected_c = [10, 20, 50]
            
            if r_val not in expected_r or c_val not in expected_c:
                validation['warnings'].append(
                    f"Unusual R-C combination: R={r_val}, C={c_val}. "
                    f"Model trained on R={expected_r}, C={expected_c}"
                )
        
        return validation
    
    def predict_breath(self, breath_df: pd.DataFrame, 
                      return_uncertainty: bool = False) -> pd.DataFrame:
        """
        Predict pressure for a single breath
        
        Args:
            breath_df: DataFrame with columns [id, time_step, u_in, u_out, R, C]
                      Rows must be in chronological order (sorted by time_step)
            return_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            DataFrame with columns [id, pressure, status, reason, model_version]
            If return_uncertainty=True, also includes 'uncertainty' column
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        validation = self.validate_input(breath_df)
        
        if not validation['valid']:
            # Return abstain status for all rows
            result_df = pd.DataFrame({
                'id': breath_df['id'],
                'pressure': np.nan,
                'status': 'abstain',
                'reason': '; '.join(validation['errors']),
                'model_version': self.VERSION
            })
            
            if return_uncertainty:
                result_df['uncertainty'] = np.nan
            
            return result_df
        
        # Extract features
        X = breath_df[self.REQUIRED_FEATURES].values
        
        # Predict
        try:
            predictions = self.model.predict(X)
            
            # Check for invalid predictions
            if not np.all(np.isfinite(predictions)):
                result_df = pd.DataFrame({
                    'id': breath_df['id'],
                    'pressure': np.nan,
                    'status': 'abstain',
                    'reason': 'Model produced non-finite predictions',
                    'model_version': self.VERSION
                })
            else:
                result_df = pd.DataFrame({
                    'id': breath_df['id'],
                    'pressure': predictions,
                    'status': 'ok',
                    'reason': '',
                    'model_version': self.VERSION
                })
            
            # Add uncertainty if requested (placeholder - actual uncertainty estimation would require ensemble)
            if return_uncertainty:
                # Simple uncertainty estimate based on prediction magnitude
                # In production, this should be replaced with proper uncertainty quantification
                result_df['uncertainty'] = np.abs(predictions) * 0.1
            
            # Add warnings if any
            if validation['warnings']:
                result_df['reason'] = '; '.join(validation['warnings'])
            
            return result_df
            
        except Exception as e:
            # Return abstain on any error
            result_df = pd.DataFrame({
                'id': breath_df['id'],
                'pressure': np.nan,
                'status': 'abstain',
                'reason': f'Prediction error: {str(e)}',
                'model_version': self.VERSION
            })
            
            if return_uncertainty:
                result_df['uncertainty'] = np.nan
            
            return result_df
    
    def predict_multiple_breaths(self, df: pd.DataFrame, 
                                breath_id_col: str = 'breath_id',
                                return_uncertainty: bool = False) -> pd.DataFrame:
        """
        Predict pressure for multiple breaths
        
        Args:
            df: DataFrame with multiple breaths (must include breath_id column)
            breath_id_col: Name of the breath ID column
            return_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            DataFrame with predictions for all breaths
        """
        if breath_id_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{breath_id_col}' column")
        
        results = []
        
        for breath_id, breath_data in df.groupby(breath_id_col):
            # Sort by time_step to maintain order
            breath_data = breath_data.sort_values('time_step')
            result = self.predict_breath(breath_data, return_uncertainty=return_uncertainty)
            results.append(result)
        
        return pd.concat(results, ignore_index=True)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'version': self.VERSION,
            'model_type': self.MODEL_TYPE,
            'required_features': self.REQUIRED_FEATURES,
            'loaded': self.loaded,
            'input_format': {
                'required_columns': ['id'] + self.REQUIRED_FEATURES,
                'optional_columns': [],
                'ordering': 'Must be sorted by time_step within each breath'
            },
            'output_format': {
                'columns': ['id', 'pressure', 'status', 'reason', 'model_version'],
                'status_codes': {
                    'ok': 'Prediction successful',
                    'abstain': 'Model cannot make reliable prediction'
                }
            },
            'limitations': [
                'Requires R and C values (cannot handle nulls)',
                'Trained on artificial test lung data only',
                'Not validated for patient care',
                'Performance varies by R-C combination'
            ]
        }


def create_mock_breath() -> pd.DataFrame:
    """
    Create a mock breath for testing the interface
    
    Returns:
        DataFrame with a sample breath
    """
    # Create a simple mock breath with 80 time steps
    time_steps = np.linspace(0, 2, 80)
    
    mock_data = {
        'id': range(1, 81),
        'time_step': time_steps,
        'R': [20] * 80,
        'C': [50] * 80,
        'u_in': np.concatenate([np.linspace(0, 25, 40), np.linspace(25, 0, 40)]),
        'u_out': [0] * 40 + [1] * 40
    }
    
    return pd.DataFrame(mock_data)


if __name__ == "__main__":
    print("="*60)
    print("TESTING INFERENCE WRAPPER")
    print("="*60)
    
    # Initialize predictor
    predictor = PressurePredictor()
    
    # Load model
    print("\nLoading model...")
    predictor.load_model('artifacts/model/final_model.pkl')
    print("✓ Model loaded")
    
    # Get model info
    print("\nModel Info:")
    info = predictor.get_model_info()
    print(f"  Version: {info['version']}")
    print(f"  Type: {info['model_type']}")
    print(f"  Features: {', '.join(info['required_features'])}")
    
    # Create mock breath
    print("\nTesting with mock breath...")
    mock_breath = create_mock_breath()
    print(f"Mock breath: {len(mock_breath)} time steps")
    
    # Predict
    result = predictor.predict_breath(mock_breath, return_uncertainty=True)
    print(f"\nPrediction result:")
    print(f"  Status: {result['status'].iloc[0]}")
    print(f"  Predictions: {len(result)} values")
    print(f"  Mean pressure: {result['pressure'].mean():.2f}")
    print(f"  Std pressure: {result['pressure'].std():.2f}")
    
    # Save mock breath for app developer
    mock_file = Path('artifacts/model/mock_breath_example.csv')
    mock_breath.to_csv(mock_file, index=False)
    print(f"\n✓ Saved mock breath to {mock_file}")
    
    # Save mock prediction
    result_file = Path('artifacts/model/mock_prediction_example.csv')
    result.to_csv(result_file, index=False)
    print(f"✓ Saved mock prediction to {result_file}")
    
    print("\n" + "="*60)
    print("INFERENCE WRAPPER TEST COMPLETE")
    print("="*60)
