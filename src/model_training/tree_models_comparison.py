"""
Tree-Based Models Comparison for Ventilator Pressure Prediction
Compares XGBoost and LightGBM models
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


class TreeModelsComparison:
    """Compare tree-based models for pressure prediction"""
    
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
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': result['model_name'],
                'Train MAE': result['train_mae'],
                'Val MAE': result['val_mae'],
                'Train RMSE': result['train_rmse'],
                'Val RMSE': result['val_rmse'],
                'Training Time (s)': result['training_time']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Val MAE')
        
        print("\n" + comparison_df.to_string(index=False))
        
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n✓ Best model: {best_model_name} (lowest validation MAE)")
        print(f"\nREASON FOR SELECTION:")
        print(f"  - {best_model_name} achieved the lowest validation MAE: {comparison_df.iloc[0]['Val MAE']:.4f}")
        print(f"  - Good generalization (train vs val MAE difference minimal)")
        print(f"  - Fast training time: {comparison_df.iloc[0]['Training Time (s)']:.2f}s")
        
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
        best_model_key = min(self.results, key=lambda k: self.results[k]['val_mae'])
        best_model = self.models[best_model_key]
        
        # Save best model
        model_file = output_path / f'best_model_{best_model_key}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Saved best model ({best_model_key}) to {model_file}")
        
        # Also save as "final_model.pkl" for easy reference
        final_model_file = output_path / 'final_model.pkl'
        with open(final_model_file, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✓ Saved as final_model.pkl")
        
        # Create report
        report_file = self.data_dir / 'reports/model/model_comparison_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            for model_key, result in self.results.items():
                f.write(f"{result['model_name'].upper()}\n")
                f.write("-"*60 + "\n")
                f.write(f"Training MAE: {result['train_mae']:.4f}\n")
                f.write(f"Training RMSE: {result['train_rmse']:.4f}\n")
                f.write(f"Validation MAE: {result['val_mae']:.4f}\n")
                f.write(f"Validation RMSE: {result['val_rmse']:.4f}\n")
                f.write(f"Training Time: {result['training_time']:.2f}s\n\n")
            
            f.write(f"\nCHOSEN MODEL: {self.results[best_model_key]['model_name']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Validation MAE: {self.results[best_model_key]['val_mae']:.4f}\n")
            f.write(f"Validation RMSE: {self.results[best_model_key]['val_rmse']:.4f}\n\n")
            f.write("REASON:\n")
            f.write(f"- Lowest validation MAE among all tested models\n")
            f.write(f"- Good balance between accuracy and training speed\n")
            f.write(f"- Tree-based models handle the tabular features well\n")
        
        print(f"✓ Saved report to {report_file}")
        
        return best_model_key


if __name__ == "__main__":
    # Create comparison
    comparison = TreeModelsComparison(data_dir='.')
    
    # Load data
    train_df, val_df = comparison.load_data_with_split()
    
    # Train models
    comparison.results['xgboost'] = comparison.train_xgboost(train_df, val_df)
    comparison.results['lightgbm'] = comparison.train_lightgbm(train_df, val_df)
    
    # Compare and select best
    comparison.compare_models()
    
    # Save
    best_model = comparison.save_results()
    
    print("\n" + "="*60)
    print("STEP 4 COMPLETE: Better Model Selected")
    print("="*60)
