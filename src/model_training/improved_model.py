"""
Improved Ventilator Pressure Prediction Model
Addresses performance issues with feature engineering and advanced techniques
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import pickle
import time
import warnings
warnings.filterwarnings('ignore')


class ImprovedPressureModel:
    """Enhanced model with feature engineering and hyperparameter tuning"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.base_features = ['R', 'C', 'time_step', 'u_in', 'u_out']
        self.target_column = 'pressure'
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features to capture temporal dependencies"""
        print("  Creating lag features...")
        df = df.copy()
        
        # Sort by breath_id and time_step to ensure proper ordering
        df = df.sort_values(['breath_id', 'time_step'], ignore_index=True)
        
        # Create lag features within each breath (only for u_in and u_out)
        # Skip pressure lags to avoid data leakage during inference
        for col in ['u_in', 'u_out']:
            if col in df.columns:
                # Lag 1 (previous time step)
                df[f'{col}_lag1'] = df.groupby('breath_id', sort=False)[col].shift(1).fillna(0)
                # Lag 2 (two time steps back)
                df[f'{col}_lag2'] = df.groupby('breath_id', sort=False)[col].shift(2).fillna(0)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between R, C, and control signals"""
        print("  Creating interaction features...")
        df = df.copy()
        
        # R-C interactions (lung mechanics)
        df['RC_product'] = df['R'] * df['C']
        df['RC_ratio'] = df['R'] / (df['C'] + 1e-6)
        
        # Flow-resistance interactions
        df['u_in_R'] = df['u_in'] * df['R']
        df['u_in_C'] = df['u_in'] * df['C']
        df['u_in_RC'] = df['u_in'] * df['RC_product']
        
        # Valve-time interactions
        df['u_out_time'] = df['u_out'] * df['time_step']
        df['u_in_time'] = df['u_in'] * df['time_step']
        
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window aggregate features"""
        print("  Creating aggregation features...")
        df = df.copy()
        
        # Rolling window features (window=5 time steps only)
        window = 5
        # Rolling mean of u_in
        df[f'u_in_roll_mean_{window}'] = df.groupby('breath_id', sort=False)['u_in'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Cumulative sum of u_in (approximation of volume)
        df['u_in_cumsum'] = df.groupby('breath_id', sort=False)['u_in'].cumsum()
        
        return df
    
    def create_phase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features to identify breath phases"""
        print("  Creating phase features...")
        df = df.copy()
        
        # Normalize time within breath (0 to 1)
        df['time_normalized'] = df.groupby('breath_id', sort=False)['time_step'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )
        
        # Breath phase indicators
        df['is_early_phase'] = (df['time_normalized'] < 0.3).astype(int)
        df['is_late_phase'] = (df['time_normalized'] >= 0.7).astype(int)
        
        # Cyclical encoding of time (captures periodicity)
        df['time_sin'] = np.sin(2 * np.pi * df['time_normalized'])
        df['time_cos'] = np.cos(2 * np.pi * df['time_normalized'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        print("Engineering features...")
        
        # Create all feature types
        df = self.create_interaction_features(df)
        df = self.create_phase_features(df)
        df = self.create_aggregation_features(df)
        
        if is_training:
            # Only create lag features with pressure during training
            df = self.create_lag_features(df)
        else:
            # For inference, create lag features without pressure
            df = self.create_lag_features(df)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of all feature columns"""
        # Exclude metadata and target columns
        exclude_cols = ['id', 'breath_id', 'pressure']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def load_data_with_split(self):
        """Load training data with saved split"""
        print("Loading training data...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        print("Loading split breath IDs...")
        train_breath_ids = np.load(self.data_dir / 'src/model_data/train_breath_ids.npy')
        val_breath_ids = np.load(self.data_dir / 'src/model_data/val_breath_ids.npy')
        
        print("Splitting data by breath IDs...")
        train_data = train_df[train_df['breath_id'].isin(train_breath_ids)].copy()
        val_data = train_df[train_df['breath_id'].isin(val_breath_ids)].copy()
        
        print(f"Train: {len(train_data):,} rows ({len(train_breath_ids):,} breaths)")
        print(f"Val: {len(val_data):,} rows ({len(val_breath_ids):,} breaths)")
        
        return train_data, val_data
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
              use_feature_engineering: bool = True):
        """Train improved XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING IMPROVED MODEL")
        print("="*60)
        
        # Feature engineering
        if use_feature_engineering:
            train_df = self.engineer_features(train_df, is_training=True)
            val_df = self.engineer_features(val_df, is_training=True)
            feature_cols = self.get_feature_columns(train_df)
            print(f"\nUsing {len(feature_cols)} features (including engineered)")
        else:
            feature_cols = self.base_features
            print(f"\nUsing {len(feature_cols)} base features")
        
        # Prepare data
        X_train = train_df[feature_cols].values
        y_train = train_df[self.target_column].values
        X_val = val_df[feature_cols].values
        y_val = val_df[self.target_column].values
        
        print(f"Training on {len(X_train):,} samples...")
        start_time = time.time()
        
        # Improved hyperparameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 9,               # Slightly reduced from 10
            'learning_rate': 0.05,        
            'n_estimators': 300,          # Reduced from 500 for speed
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,    
            'min_child_weight': 3,        
            'gamma': 0.1,                 
            'reg_alpha': 0.1,             
            'reg_lambda': 1.0,            
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=25
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"\n{'='*60}")
        print("FINAL METRICS")
        print(f"{'='*60}")
        print(f"\nTraining:")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nValidation:")
        print(f"  MAE:  {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'training_time': training_time,
            'n_features': len(feature_cols),
            'feature_importance': feature_importance
        }
    
    def save_model(self, output_path: Path):
        """Save the trained model"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved to {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("IMPROVED PRESSURE PREDICTION MODEL")
    print("="*60)
    
    # Initialize
    model = ImprovedPressureModel(data_dir='.')
    
    # Load data
    train_df, val_df = model.load_data_with_split()
    
    # Train with feature engineering
    results = model.train(train_df, val_df, use_feature_engineering=True)
    
    # Save improved model
    model.save_model(Path('artifacts/model/improved_model.pkl'))
    
    # Save results
    results_file = Path('artifacts/model/improved_model_results.json')
    with open(results_file, 'w') as f:
        # Convert DataFrame to dict for JSON serialization
        results_copy = results.copy()
        results_copy['feature_importance'] = results['feature_importance'].to_dict('records')
        json.dump(results_copy, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    print("\n" + "="*60)
    print("IMPROVEMENT COMPLETE")
    print("="*60)
