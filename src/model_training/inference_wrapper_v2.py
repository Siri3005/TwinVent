"""
Improved Model Inference Wrapper with Feature Engineering
Version 2.0 - Supports engineered features for improved model
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import numpy as np
import joblib


class ImprovedPressurePredictor:
    """
    Inference wrapper for improved pressure prediction model with feature engineering
    """
    
    VERSION = "2.0.0-improved"
    REQUIRED_BASE_FEATURES = ['R', 'C', 'time_step', 'u_in', 'u_out']
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model file (optional, can load later)
        """
        self.model = None
        self.loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str = None):
        """
        Load trained model from file
        
        Args:
            model_path: Path to .pkl model file
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "artifacts" / "model" / "final_model.pkl"
        
        self.model = joblib.load(model_path)
        self.loaded = True
        print(f"Model loaded from {model_path}")
    
    def validate_input(self, breath_df: pd.DataFrame) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate input DataFrame
        
        Args:
            breath_df: DataFrame with breath data
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = set(self.REQUIRED_BASE_FEATURES + ['id']) - set(breath_df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check for null values in required features
        for col in self.REQUIRED_BASE_FEATURES:
            if breath_df[col].isnull().any():
                errors.append(f"Column {col} contains null values")
        
        # Check for finite values
        for col in self.REQUIRED_BASE_FEATURES:
            if not breath_df[col].apply(np.isfinite).all():
                errors.append(f"Column {col} contains non-finite values")
        
        # Check chronological ordering
        if not breath_df['time_step'].is_monotonic_increasing:
            warnings.append("time_step is not monotonically increasing")
        
        # Check R and C are valid values
        valid_R = {5, 20, 50}
        valid_C = {10, 20, 50}
        
        if not breath_df['R'].isin(valid_R).all():
            warnings.append(f"R values outside trained range {valid_R}")
        
        if not breath_df['C'].isin(valid_C).all():
            warnings.append(f"C values outside trained range {valid_C}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations
        
        Args:
            df: DataFrame with base features [R, C, time_step, u_in, u_out]
            
        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()
        
        # Since we're processing a single breath, add a temporary breath_id for grouping
        df['breath_id'] = 0
        
        # Ensure data is sorted by time_step
        df = df.sort_values('time_step', ignore_index=True)
        
        # 1. Interaction features
        df['RC_product'] = df['R'] * df['C']
        df['RC_ratio'] = df['R'] / (df['C'] + 1e-6)
        df['u_in_R'] = df['u_in'] * df['R']
        df['u_in_C'] = df['u_in'] * df['C']
        df['u_in_RC'] = df['u_in'] * df['RC_product']
        
        # 2. Time interactions
        df['u_in_time'] = df['u_in'] * df['time_step']
        df['u_out_time'] = df['u_out'] * df['time_step']
        
        # 3. Phase features
        time_min = df['time_step'].min()
        time_max = df['time_step'].max()
        time_range = time_max - time_min
        
        if time_range > 0:
            df['time_normalized'] = (df['time_step'] - time_min) / time_range
        else:
            df['time_normalized'] = 0.5
        
        df['time_sin'] = np.sin(2 * np.pi * df['time_normalized'])
        df['time_cos'] = np.cos(2 * np.pi * df['time_normalized'])
        
        df['is_early_phase'] = (df['time_normalized'] < 0.3).astype(int)
        df['is_late_phase'] = (df['time_normalized'] >= 0.7).astype(int)
        
        # 4. Lag features (within breath, using groupby)
        for col in ['u_in', 'u_out']:
            df[f'{col}_lag1'] = df.groupby('breath_id', sort=False)[col].shift(1).fillna(0)
            df[f'{col}_lag2'] = df.groupby('breath_id', sort=False)[col].shift(2).fillna(0)
        
        # 5. Rolling aggregates (within breath)
        window = 5
        df['u_in_roll_mean_5'] = df.groupby('breath_id', sort=False)['u_in'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # 6. Cumulative features (within breath)
        df['u_in_cumsum'] = df.groupby('breath_id', sort=False)['u_in'].cumsum()
        
        # Remove the temporary breath_id column
        df = df.drop(columns=['breath_id'])
        
        return df
    
    def predict_breath(self, breath_df: pd.DataFrame,
                      return_uncertainty: bool = False) -> pd.DataFrame:
        """
        Predict pressure for a single breath with feature engineering
        
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
        
        # Get feature columns (exclude id and target if present)
        feature_cols = [col for col in df_features.columns 
                       if col not in ['id', 'pressure', 'breath_id']]
        
        # Predict
        try:
            X = df_features[feature_cols].values
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
            
            # Add uncertainty if requested
            if return_uncertainty:
                # Simple uncertainty estimate based on prediction magnitude
                # In production, this could be replaced with proper uncertainty quantification
                result_df['uncertainty'] = np.abs(predictions) * 0.05
            
            # Add warnings if any
            if validation['warnings']:
                if result_df['status'].iloc[0] == 'ok':
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
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded model
        
        Returns:
            Dictionary with model metadata
        """
        if not self.loaded:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'version': self.VERSION,
            'model_type': type(self.model).__name__,
            'features_required': self.REQUIRED_BASE_FEATURES,
            'engineered_features': True,
            'feature_count': 23
        }


# Convenience function for backward compatibility
def create_predictor(model_path: str = None) -> ImprovedPressurePredictor:
    """Create and return a predictor instance"""
    return ImprovedPressurePredictor(model_path)


if __name__ == "__main__":
    # Test the wrapper
    print("Testing Improved Pressure Predictor...")
    
    predictor = ImprovedPressurePredictor()
    predictor.load_model()
    
    # Load mock breath
    mock_path = Path(__file__).parent.parent.parent / "artifacts" / "model" / "mock_breath_example.csv"
    breath_df = pd.read_csv(mock_path)
    
    print(f"Loaded mock breath: {len(breath_df)} samples")
    print(f"Columns: {list(breath_df.columns)}")
    
    # Predict
    result = predictor.predict_breath(breath_df, return_uncertainty=True)
    
    print(f"\nPrediction Status: {result['status'].iloc[0]}")
    print(f"Model Version: {result['model_version'].iloc[0]}")
    
    if result['status'].iloc[0] == 'ok':
        print(f"Pressure Range: {result['pressure'].min():.2f} to {result['pressure'].max():.2f} cmH₂O")
        print(f"Mean Pressure: {result['pressure'].mean():.2f} cmH₂O")
    else:
        print(f"Reason: {result['reason'].iloc[0]}")
    
    print("\n✓ Inference wrapper test complete")
