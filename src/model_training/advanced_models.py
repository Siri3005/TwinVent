"""
Advanced Models for Ventilator Pressure Prediction
Compares tree-based (XGBoost/LightGBM) and sequence (LSTM) models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import pickle
from typing import Dict, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class AdvancedModelComparison:
    """Compare advanced models for pressure prediction"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.feature_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
        self.target_column = 'pressure'
        self.results = {}
        
    def load_data_with_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load full train data and split using saved breath IDs"""
        print("Loading training data...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        print("Loading split breath IDs...")
        train_breath_ids = np.load(self.data_dir / 'src/model_data/train_breath_ids.npy')
        val_breath_ids = np.load(self.data_dir / 'src/model_data/val_breath_ids.npy')
        
        print(f"Splitting data by breath IDs...")
        train_data = train_df[train_df['breath_id'].isin(train_breath_ids)].copy()
        val_data = train_df[train_df['breath_id'].isin(val_breath_ids)].copy()
        
        print(f"Train: {len(train_data):,} rows")
        print(f"Val: {len(val_data):,} rows")
        
        return train_data, val_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and target from DataFrame"""
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        return X, y
    
    def train_xgboost(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        print(f"Training on {len(X_train):,} samples...")
        start_time = time.time()
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"\nTraining Metrics:")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nValidation Metrics:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        
        self.models['xgboost'] = model
        
        return {
            'model_name': 'XGBoost',
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'training_time': float(training_time),
            'params': params
        }
    
    def train_lightgbm(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train LightGBM model"""
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM MODEL")
        print("="*60)
        
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        print(f"Training on {len(X_train):,} samples...")
        start_time = time.time()
        
        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'max_depth': 7,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"\nTraining Metrics:")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"\nValidation Metrics:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        
        self.models['lightgbm'] = model
        
        return {
            'model_name': 'LightGBM',
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'training_time': float(training_time),
            'params': params
        }
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 80) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM - each breath is a sequence"""
        sequences = []
        targets = []
        
        for breath_id, group in df.groupby('breath_id'):
            # Each breath has 80 time steps
            breath_data = group.sort_values('time_step')
            
            # Features for each time step
            features = breath_data[self.feature_columns].values
            target = breath_data[self.target_column].values
            
            sequences.append(features)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """Train LSTM sequence model"""
        print("\n" + "="*60)
        print("TRAINING LSTM SEQUENCE MODEL")
        print("="*60)
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Set random seeds
            tf.random.set_seed(42)
            np.random.seed(42)
            
            print("Creating sequences...")
            X_train, y_train = self.create_sequences(train_df)
            X_val, y_val = self.create_sequences(val_df)
            
            print(f"Train sequences: {X_train.shape}")
            print(f"Val sequences: {X_val.shape}")
            
            # Use a smaller subset for faster training
            max_train_samples = 10000
            if len(X_train) > max_train_samples:
                print(f"\nUsing subset of {max_train_samples:,} breaths for faster training")
                indices = np.random.choice(len(X_train), max_train_samples, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            print(f"\nTraining on {len(X_train):,} breath sequences...")
            start_time = time.time()
            
            # Build LSTM model
            model = keras.Sequential([
                layers.LSTM(64, return_sequences=True, input_shape=(80, len(self.feature_columns))),
                layers.Dropout(0.2),
                layers.LSTM(32, return_sequences=True),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mae',
                metrics=['mae', 'mse']
            )
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=64,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Evaluate
            y_train_pred = model.predict(X_train, verbose=0).reshape(-1, 80)
            y_val_pred = model.predict(X_val, verbose=0).reshape(-1, 80)
            
            # Flatten for metrics
            y_train_flat = y_train.reshape(-1)
            y_train_pred_flat = y_train_pred.reshape(-1)
            y_val_flat = y_val.reshape(-1)
            y_val_pred_flat = y_val_pred.reshape(-1)
            
            train_mae = mean_absolute_error(y_train_flat, y_train_pred_flat)
            train_rmse = np.sqrt(mean_squared_error(y_train_flat, y_train_pred_flat))
            val_mae = mean_absolute_error(y_val_flat, y_val_pred_flat)
            val_rmse = np.sqrt(mean_squared_error(y_val_flat, y_val_pred_flat))
            
            print(f"\nTraining Metrics:")
            print(f"  MAE: {train_mae:.4f}")
            print(f"  RMSE: {train_rmse:.4f}")
            print(f"\nValidation Metrics:")
            print(f"  MAE: {val_mae:.4f}")
            print(f"  RMSE: {val_rmse:.4f}")
            
            self.models['lstm'] = model
            
            return {
                'model_name': 'LSTM',
                'train_mae': float(train_mae),
                'train_rmse': float(train_rmse),
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'training_time': float(training_time),
                'architecture': 'LSTM(64) -> LSTM(32) -> Dense(16) -> Dense(1)'
            }
            
        except Exception as e:
            print(f"❌ LSTM training failed: {str(e)}")
            return {
                'model_name': 'LSTM',
                'error': str(e),
                'train_mae': None,
                'val_mae': None
            }
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, result in self.results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': result['model_name'],
                    'Train MAE': result['train_mae'],
                    'Val MAE': result['val_mae'],
                    'Train RMSE': result['train_rmse'],
                    'Val RMSE': result['val_rmse'],
                    'Training Time (s)': result['training_time']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by validation MAE (lower is better)
        comparison_df = comparison_df.sort_values('Val MAE')
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n✓ Best model: {best_model_name} (lowest validation MAE)")
        
        return comparison_df
    
    def save_results(self):
        """Save comparison results and best model"""
        output_path = self.data_dir / 'artifacts/model'
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save comparison results
        results_file = output_path / 'model_comparison.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved comparison results to {results_file}")
        
        # Determine best model
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        if valid_results:
            best_model_key = min(valid_results, key=lambda k: valid_results[k]['val_mae'])
            best_model = self.models[best_model_key]
            
            # Save best model
            if best_model_key in ['xgboost', 'lightgbm']:
                model_file = output_path / f'best_model_{best_model_key}.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(best_model, f)
                print(f"✓ Saved best model ({best_model_key}) to {model_file}")
            elif best_model_key == 'lstm':
                model_file = output_path / 'best_model_lstm.h5'
                best_model.save(model_file)
                print(f"✓ Saved best model (LSTM) to {model_file}")
        
        # Create report
        report_file = self.data_dir / 'reports/model/advanced_models_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("ADVANCED MODELS COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            for model_key, result in self.results.items():
                f.write(f"{result['model_name'].upper()}\n")
                f.write("-"*60 + "\n")
                
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n\n")
                else:
                    f.write(f"Training MAE: {result['train_mae']:.4f}\n")
                    f.write(f"Training RMSE: {result['train_rmse']:.4f}\n")
                    f.write(f"Validation MAE: {result['val_mae']:.4f}\n")
                    f.write(f"Validation RMSE: {result['val_rmse']:.4f}\n")
                    f.write(f"Training Time: {result['training_time']:.2f}s\n\n")
            
            if valid_results:
                f.write(f"\nBEST MODEL: {valid_results[best_model_key]['model_name']}\n")
                f.write(f"Validation MAE: {valid_results[best_model_key]['val_mae']:.4f}\n")
                f.write(f"Validation RMSE: {valid_results[best_model_key]['val_rmse']:.4f}\n")
        
        print(f"✓ Saved report to {report_file}")


if __name__ == "__main__":
    # Create comparison
    comparison = AdvancedModelComparison(data_dir='.')
    
    # Load data
    train_df, val_df = comparison.load_data_with_split()
    
    # Train models
    comparison.results['xgboost'] = comparison.train_xgboost(train_df, val_df)
    comparison.results['lightgbm'] = comparison.train_lightgbm(train_df, val_df)
    comparison.results['lstm'] = comparison.train_lstm(train_df, val_df)
    
    # Compare
    comparison.compare_models()
    
    # Save
    comparison.save_results()
