"""
LSTM Model Inference Wrapper for External H5 Model
Supports time series LSTM model with feature engineering
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMPressurePredictor:
    """
    Inference wrapper for LSTM pressure prediction model
    """
    
    VERSION = "3.0.0-lstm"
    REQUIRED_BASE_FEATURES = ['R', 'C', 'time_step', 'u_in', 'u_out']
    EXPECTED_TIMESTEPS = 80  # Standard breath length
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to .h5 model file
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not installed. Install with: pip install tensorflow")
        
        self.model = None
        self.loaded = False
        self.n_features = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str = None):
        """
        Load trained LSTM model from H5 file
        
        Args:
            model_path: Path to .h5 model file
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "artifacts" / "model" / "lstm_model.h5"
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try to load with compile=False to avoid optimizer issues
            self.model = keras.models.load_model(str(model_path), compile=False)
            self.loaded = True
            
            # Determine number of features from input shape
            input_shape = self.model.input_shape
            if len(input_shape) == 3:  # (batch, timesteps, features)
                self.n_features = input_shape[2]
            else:
                self.n_features = None
            
            print(f"✓ LSTM model loaded from {model_path}")
            print(f"  Input shape: {input_shape}")
            print(f"  Expected features: {self.n_features}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LSTM model: {e}")
    
    def validate_input(self, breath_df: pd.DataFrame) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate input DataFrame
        
        Args:
            breath_df: DataFrame with breath data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings_list = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_BASE_FEATURES + ['id']) - set(breath_df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings_list}
        
        # Check for null values
        for col in self.REQUIRED_BASE_FEATURES:
            if breath_df[col].isnull().any():
                errors.append(f"Column {col} contains null values")
        
        # Check for finite values
        for col in self.REQUIRED_BASE_FEATURES:
            if not breath_df[col].apply(np.isfinite).all():
                errors.append(f"Column {col} contains non-finite values")
        
        # Check number of timesteps
        if len(breath_df) != self.EXPECTED_TIMESTEPS:
            warnings_list.append(f"Expected {self.EXPECTED_TIMESTEPS} timesteps, got {len(breath_df)}")
        
        # Check chronological ordering
        if not breath_df['time_step'].is_monotonic_increasing:
            warnings_list.append("time_step is not monotonically increasing")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings_list
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features exactly as in the training notebook
        Creates 35 features matching the trained model
        
        Args:
            df: DataFrame with base features [R, C, time_step, u_in, u_out]
            
        Returns:
            DataFrame with 35 engineered features
        """
        df = df.copy()
        df['breath_id'] = 0  # Temporary for grouping
        df = df.sort_values('time_step', ignore_index=True)
        
        # Create groupby object for within-breath operations
        data = df.groupby('breath_id')
        
        # 1. Standardize u_in and time_step within breath (as in notebook)
        df['un_in_std'] = data['u_in'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        df['time_step_std'] = data['time_step'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        
        # 2. Create lag features (forward and backward) - exactly as in notebook
        # after = shift(), back = shift(-1)
        # Features: time_step_std, un_in_std, u_out
        
        # After (previous timesteps) - shift(1) to shift(5)
        for i in range(1, 6):
            df[f'time_step_after{i if i > 1 else ""}'] = data['time_step_std'].shift(i)
            df[f'u_in_after{i if i > 1 else ""}'] = data['un_in_std'].shift(i)
            df[f'u_out_after{i if i > 1 else ""}'] = data['u_out'].shift(i)
        
        # Back (future timesteps) - shift(-1) to shift(-5)
        for i in range(1, 6):
            df[f'time_step_back{i if i > 1 else ""}'] = data['time_step_std'].shift(-i)
            df[f'u_in_back{i if i > 1 else ""}'] = data['un_in_std'].shift(-i)
            df[f'u_out_back{i if i > 1 else ""}'] = data['u_out'].shift(-i)
        
        # 3. Fill NaN values with 0 (as in notebook)
        df.fillna(0, inplace=True)
        
        # 4. Select features in correct order (drop: pressure, id, breath_id, u_in, time_step)
        # Keep: R, C, u_out, un_in_std, time_step_std, and 30 lag features
        feature_order = [
            'R', 'C', 'u_out',
            'un_in_std', 'time_step_std',
            # After features (lag 1-5)
            'time_step_after', 'u_in_after', 'u_out_after',
            'time_step_after2', 'u_in_after2', 'u_out_after2',
            'time_step_after3', 'u_in_after3', 'u_out_after3',
            'time_step_after4', 'u_in_after4', 'u_out_after4',
            'time_step_after5', 'u_in_after5', 'u_out_after5',
            # Back features (future 1-5)
            'time_step_back', 'u_in_back', 'u_out_back',
            'time_step_back2', 'u_in_back2', 'u_out_back2',
            'time_step_back3', 'u_in_back3', 'u_out_back3',
            'time_step_back4', 'u_in_back4', 'u_out_back4',
            'time_step_back5', 'u_in_back5', 'u_out_back5',
        ]
        
        return df[feature_order]
    
    def predict_breath(self, breath_df: pd.DataFrame,
                      return_uncertainty: bool = False) -> pd.DataFrame:
        """
        Predict pressure for a single breath using LSTM model
        
        Args:
            breath_df: DataFrame with columns [id, time_step, u_in, u_out, R, C]
            return_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            DataFrame with columns [id, pressure, status, reason, model_version]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Validate input
        validation = self.validate_input(breath_df)
        
        if not validation['valid']:
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
        
        # Engineer features
        try:
            df_features = self.engineer_features(breath_df)
        except Exception as e:
            result_df = pd.DataFrame({
                'id': breath_df['id'],
                'pressure': np.nan,
                'status': 'abstain',
                'reason': f'Feature engineering error: {str(e)}',
                'model_version': self.VERSION
            })
            
            if return_uncertainty:
                result_df['uncertainty'] = np.nan
            
            return result_df
        
        # Prepare input for LSTM (needs 3D shape: batch, timesteps, features)
        try:
            X = df_features.values
            X = X.reshape(1, X.shape[0], X.shape[1])  # (1, timesteps, features)
            
            # Make prediction
            predictions = self.model.predict(X, verbose=0)
            
            # Model outputs 80 values in shape (1, 80)
            # This matches the 80 timesteps in the input
            predictions = predictions[0]  # Get the 80 predictions
            
            # Ensure we have the right number of predictions
            if len(predictions) != len(breath_df):
                raise ValueError(f"Model output length mismatch: {len(predictions)} != {len(breath_df)}")
            
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
            
            # Add uncertainty if requested
            if return_uncertainty:
                # Simple uncertainty based on prediction magnitude
                result_df['uncertainty'] = np.abs(predictions) * 0.05
            
            # Add warnings if any
            if validation['warnings']:
                if result_df['status'].iloc[0] == 'ok':
                    result_df['reason'] = '; '.join(validation['warnings'])
            
            return result_df
            
        except Exception as e:
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
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self.loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'version': self.VERSION,
            'model_type': 'LSTM',
            'features_required': self.REQUIRED_BASE_FEATURES,
            'engineered_features': True,
            'feature_count': self.n_features,
            'expected_timesteps': self.EXPECTED_TIMESTEPS
        }


if __name__ == "__main__":
    # Test the wrapper
    print("Testing LSTM Pressure Predictor...")
    
    predictor = LSTMPressurePredictor()
    
    # Try to load model
    model_path = Path(__file__).parent.parent.parent / "artifacts" / "model" / "lstm_model_compatible.h5"
    
    if model_path.exists():
        predictor.load_model(str(model_path))
        
        # Load mock breath
        mock_path = Path(__file__).parent.parent.parent / "artifacts" / "model" / "mock_breath_example.csv"
        if mock_path.exists():
            breath_df = pd.read_csv(mock_path)
            
            print(f"\nLoaded mock breath: {len(breath_df)} samples")
            
            # Predict
            result = predictor.predict_breath(breath_df, return_uncertainty=True)
            
            print(f"\nPrediction Status: {result['status'].iloc[0]}")
            print(f"Model Version: {result['model_version'].iloc[0]}")
            
            if result['status'].iloc[0] == 'ok':
                print(f"Pressure Range: {result['pressure'].min():.2f} to {result['pressure'].max():.2f} cmH₂O")
                print(f"Mean Pressure: {result['pressure'].mean():.2f} cmH₂O")
            else:
                print(f"Reason: {result['reason'].iloc[0]}")
        else:
            print(f"\nMock breath not found: {mock_path}")
    else:
        print(f"\nLSTM model not found: {model_path}")
        print("Please ensure the model is converted and saved to this location")
