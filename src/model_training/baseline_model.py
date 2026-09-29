"""
Baseline Model for Ventilator Pressure Prediction
Simple reference model using R, C, time_step, u_in, and u_out
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import pickle
from typing import Dict, Tuple
import time


class BaselineModel:
    """Simple baseline model for pressure prediction"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.feature_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
        self.target_column = 'pressure'
        self.metrics = {}
        
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
        
        print(f"Train: {len(train_data):,} rows, {len(train_breath_ids):,} breaths")
        print(f"Val: {len(val_data):,} rows, {len(val_breath_ids):,} breaths")
        
        return train_data, val_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and target from DataFrame"""
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        return X, y
    
    def train(self, train_df: pd.DataFrame) -> None:
        """Train the baseline model"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)
        print(f"Model: Ridge Regression")
        print(f"Features: {', '.join(self.feature_columns)}")
        
        # Prepare data
        X_train, y_train = self.prepare_features(train_df)
        
        print(f"\nTraining on {len(X_train):,} samples...")
        start_time = time.time()
        
        # Train model
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Training metrics
        y_train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        print(f"\nTraining Metrics:")
        print(f"  MAE: {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        
        self.metrics['train'] = {
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'samples': len(X_train),
            'training_time_seconds': float(training_time)
        }
    
    def evaluate(self, val_df: pd.DataFrame) -> Dict:
        """Evaluate model on validation set"""
        print("\n" + "="*60)
        print("EVALUATING ON VALIDATION SET")
        print("="*60)
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        X_val, y_val = self.prepare_features(val_df)
        
        print(f"Evaluating on {len(X_val):,} samples...")
        
        # Predictions
        y_val_pred = self.model.predict(X_val)
        
        # Calculate metrics
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        print(f"\nValidation Metrics:")
        print(f"  MAE: {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        
        # Per-breath metrics
        val_df_copy = val_df.copy()
        val_df_copy['predicted_pressure'] = y_val_pred
        
        breath_metrics = []
        for breath_id, group in val_df_copy.groupby('breath_id'):
            breath_mae = mean_absolute_error(group[self.target_column], group['predicted_pressure'])
            breath_rmse = np.sqrt(mean_squared_error(group[self.target_column], group['predicted_pressure']))
            breath_metrics.append({
                'breath_id': int(breath_id),
                'mae': float(breath_mae),
                'rmse': float(breath_rmse)
            })
        
        breath_metrics_df = pd.DataFrame(breath_metrics)
        avg_breath_mae = breath_metrics_df['mae'].mean()
        avg_breath_rmse = breath_metrics_df['rmse'].mean()
        
        print(f"\nPer-Breath Average Metrics:")
        print(f"  Avg MAE: {avg_breath_mae:.4f}")
        print(f"  Avg RMSE: {avg_breath_rmse:.4f}")
        
        self.metrics['validation'] = {
            'mae': float(val_mae),
            'rmse': float(val_rmse),
            'samples': len(X_val),
            'breaths': len(breath_metrics),
            'avg_breath_mae': float(avg_breath_mae),
            'avg_breath_rmse': float(avg_breath_rmse)
        }
        
        return self.metrics
    
    def evaluate_by_rc_group(self, val_df: pd.DataFrame) -> Dict:
        """Evaluate performance by R-C combinations"""
        print("\n" + "="*60)
        print("EVALUATING BY R-C GROUPS")
        print("="*60)
        
        X_val, y_val = self.prepare_features(val_df)
        y_val_pred = self.model.predict(X_val)
        
        val_df_copy = val_df.copy()
        val_df_copy['predicted_pressure'] = y_val_pred
        
        rc_metrics = []
        for (r, c), group in val_df_copy.groupby(['R', 'C']):
            mae = mean_absolute_error(group[self.target_column], group['predicted_pressure'])
            rmse = np.sqrt(mean_squared_error(group[self.target_column], group['predicted_pressure']))
            rc_metrics.append({
                'R': int(r),
                'C': int(c),
                'mae': float(mae),
                'rmse': float(rmse),
                'samples': len(group)
            })
            print(f"R={r}, C={c}: MAE={mae:.4f}, RMSE={rmse:.4f} ({len(group):,} samples)")
        
        self.metrics['rc_groups'] = rc_metrics
        return rc_metrics
    
    def save_model(self, output_dir: str = 'artifacts/model'):
        """Save trained model and metrics"""
        output_path = self.data_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING BASELINE MODEL")
        print("="*60)
        
        # Save model
        model_file = output_path / 'baseline_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Saved model to {model_file}")
        
        # Save metrics
        metrics_file = output_path / 'baseline_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Saved metrics to {metrics_file}")
        
        # Save model info
        model_info = {
            'model_type': 'Ridge Regression',
            'model_class': str(type(self.model)),
            'features': self.feature_columns,
            'target': self.target_column,
            'alpha': float(self.model.alpha),
            'coefficients': {
                feat: float(coef) 
                for feat, coef in zip(self.feature_columns, self.model.coef_)
            },
            'intercept': float(self.model.intercept_)
        }
        
        info_file = output_path / 'baseline_model_info.json'
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Saved model info to {info_file}")
        
        # Create report
        report_file = self.data_dir / 'reports/model/baseline_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("BASELINE MODEL REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("MODEL DETAILS:\n")
            f.write(f"  Type: Ridge Regression\n")
            f.write(f"  Features: {', '.join(self.feature_columns)}\n")
            f.write(f"  Alpha: {self.model.alpha}\n\n")
            
            f.write("TRAINING METRICS:\n")
            f.write(f"  MAE: {self.metrics['train']['mae']:.4f}\n")
            f.write(f"  RMSE: {self.metrics['train']['rmse']:.4f}\n")
            f.write(f"  Samples: {self.metrics['train']['samples']:,}\n")
            f.write(f"  Training Time: {self.metrics['train']['training_time_seconds']:.2f}s\n\n")
            
            f.write("VALIDATION METRICS:\n")
            f.write(f"  MAE: {self.metrics['validation']['mae']:.4f}\n")
            f.write(f"  RMSE: {self.metrics['validation']['rmse']:.4f}\n")
            f.write(f"  Samples: {self.metrics['validation']['samples']:,}\n")
            f.write(f"  Breaths: {self.metrics['validation']['breaths']:,}\n")
            f.write(f"  Avg Breath MAE: {self.metrics['validation']['avg_breath_mae']:.4f}\n")
            f.write(f"  Avg Breath RMSE: {self.metrics['validation']['avg_breath_rmse']:.4f}\n\n")
            
            f.write("R-C GROUP METRICS:\n")
            for rc_metric in self.metrics['rc_groups']:
                f.write(f"  R={rc_metric['R']}, C={rc_metric['C']}: ")
                f.write(f"MAE={rc_metric['mae']:.4f}, RMSE={rc_metric['rmse']:.4f}\n")
        
        print(f"✓ Saved report to {report_file}")
        print("\n" + "="*60)
        print("BASELINE MODEL SAVE COMPLETE")
        print("="*60)
    
    def load_model(self, model_path: str = 'artifacts/model/baseline_model.pkl'):
        """Load a previously saved model"""
        model_file = self.data_dir / model_path
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✓ Loaded model from {model_file}")


if __name__ == "__main__":
    # Create baseline model
    baseline = BaselineModel(data_dir='.')
    
    # Load data with split
    train_df, val_df = baseline.load_data_with_split()
    
    # Train model
    baseline.train(train_df)
    
    # Evaluate
    baseline.evaluate(val_df)
    baseline.evaluate_by_rc_group(val_df)
    
    # Save
    baseline.save_model()
