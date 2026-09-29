"""
Test Set Prediction for Ventilator Pressure Prediction
Generates predictions for test.csv and creates submission file
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict
import json
import time


class TestPredictor:
    """Generate predictions for test set"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.feature_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
        
    def load_model(self, model_path: str = 'artifacts/model/final_model.pkl'):
        """Load the trained model"""
        model_file = self.data_dir / model_path
        print(f"Loading model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded successfully")
        
    def load_test_data(self) -> pd.DataFrame:
        """Load test dataset"""
        test_file = self.data_dir / 'test.csv'
        print(f"\nLoading test data from {test_file}...")
        test_df = pd.read_csv(test_file)
        print(f"✓ Loaded {len(test_df):,} rows")
        print(f"✓ Unique breaths: {test_df['breath_id'].nunique():,}")
        return test_df
    
    def validate_test_data(self, test_df: pd.DataFrame) -> Dict:
        """Validate test data structure"""
        print("\n" + "="*60)
        print("VALIDATING TEST DATA")
        print("="*60)
        
        validation = {}
        
        # Check required columns
        missing_cols = set(self.feature_columns + ['id']) - set(test_df.columns)
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            validation['missing_columns'] = list(missing_cols)
        else:
            print("✓ All required columns present")
            validation['columns_present'] = True
        
        # Check for nulls
        null_counts = test_df[self.feature_columns].isnull().sum()
        if null_counts.sum() > 0:
            print(f"❌ Found null values:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"   {col}: {count} nulls")
            validation['null_values'] = null_counts.to_dict()
        else:
            print("✓ No null values in features")
            validation['no_nulls'] = True
        
        # Check breath structure
        breath_sizes = test_df.groupby('breath_id').size()
        if not all(breath_sizes == 80):
            print(f"❌ Not all breaths have 80 samples")
            validation['breath_structure'] = 'invalid'
        else:
            print(f"✓ All {len(breath_sizes):,} breaths have 80 samples")
            validation['breath_structure'] = 'valid'
        
        # Check R-C combinations
        rc_combos = test_df.groupby(['R', 'C']).size()
        print(f"\n✓ Found {len(rc_combos)} R-C combinations")
        for (r, c), count in rc_combos.items():
            print(f"   R={r}, C={c}: {count:,} samples")
        
        return validation
    
    def generate_predictions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for test data"""
        print("\n" + "="*60)
        print("GENERATING TEST PREDICTIONS")
        print("="*60)
        
        print(f"Predicting {len(test_df):,} samples...")
        start_time = time.time()
        
        # Extract features
        X_test = test_df[self.feature_columns].values
        
        # Predict
        predictions = self.model.predict(X_test)
        
        prediction_time = time.time() - start_time
        print(f"✓ Predictions completed in {prediction_time:.2f} seconds")
        
        # Verify all predictions are finite
        if not np.all(np.isfinite(predictions)):
            n_invalid = np.sum(~np.isfinite(predictions))
            print(f"❌ Warning: {n_invalid} invalid predictions found!")
        else:
            print("✓ All predictions are finite")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'id': test_df['id'],
            'pressure': predictions
        })
        
        print(f"\nPrediction Statistics:")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Std: {predictions.std():.4f}")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        
        return results_df
    
    def validate_submission(self, submission_df: pd.DataFrame, 
                           sample_submission_df: pd.DataFrame) -> Dict:
        """Validate submission file format"""
        print("\n" + "="*60)
        print("VALIDATING SUBMISSION FORMAT")
        print("="*60)
        
        validation = {}
        
        # Check row count
        if len(submission_df) != len(sample_submission_df):
            print(f"❌ Row count mismatch!")
            print(f"   Expected: {len(sample_submission_df):,}")
            print(f"   Got: {len(submission_df):,}")
            validation['row_count'] = 'mismatch'
        else:
            print(f"✓ Row count matches: {len(submission_df):,}")
            validation['row_count'] = 'match'
        
        # Check columns
        expected_cols = list(sample_submission_df.columns)
        actual_cols = list(submission_df.columns)
        
        if expected_cols != actual_cols:
            print(f"❌ Column mismatch!")
            print(f"   Expected: {expected_cols}")
            print(f"   Got: {actual_cols}")
            validation['columns'] = 'mismatch'
        else:
            print(f"✓ Columns match: {expected_cols}")
            validation['columns'] = 'match'
        
        # Check ID alignment
        if not submission_df['id'].equals(sample_submission_df['id']):
            mismatched = (submission_df['id'] != sample_submission_df['id']).sum()
            print(f"❌ ID mismatch in {mismatched} rows!")
            validation['id_alignment'] = 'mismatch'
        else:
            print("✓ All IDs match sample submission")
            validation['id_alignment'] = 'match'
        
        # Check for finite values
        if not np.all(np.isfinite(submission_df['pressure'])):
            invalid_count = np.sum(~np.isfinite(submission_df['pressure']))
            print(f"❌ {invalid_count} non-finite pressure values!")
            validation['finite_values'] = False
        else:
            print("✓ All pressure values are finite")
            validation['finite_values'] = True
        
        # Check for exactly one pressure per ID
        duplicate_ids = submission_df['id'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"❌ Found {duplicate_ids} duplicate IDs!")
            validation['unique_ids'] = False
        else:
            print("✓ Each ID has exactly one pressure value")
            validation['unique_ids'] = True
        
        return validation
    
    def save_submission(self, submission_df: pd.DataFrame, 
                       filename: str = 'submission.csv'):
        """Save submission file"""
        output_file = self.data_dir / filename
        submission_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved submission to {output_file}")
        
        # Also save to artifacts
        artifacts_file = self.data_dir / 'artifacts/model/submission.csv'
        submission_df.to_csv(artifacts_file, index=False)
        print(f"✓ Saved copy to {artifacts_file}")
    
    def save_prediction_report(self, validation: Dict, submission_df: pd.DataFrame):
        """Save prediction report"""
        report = {
            'validation': validation,
            'prediction_stats': {
                'mean': float(submission_df['pressure'].mean()),
                'std': float(submission_df['pressure'].std()),
                'min': float(submission_df['pressure'].min()),
                'max': float(submission_df['pressure'].max()),
                'total_predictions': len(submission_df)
            }
        }
        
        report_file = self.data_dir / 'reports/model/test_prediction_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved prediction report to {report_file}")
        
        # Text report
        text_file = self.data_dir / 'reports/model/test_prediction_report.txt'
        with open(text_file, 'w') as f:
            f.write("TEST PREDICTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write("VALIDATION CHECKS:\n")
            f.write("-"*60 + "\n")
            for key, value in validation.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write("PREDICTION STATISTICS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Total Predictions: {report['prediction_stats']['total_predictions']:,}\n")
            f.write(f"Mean Pressure: {report['prediction_stats']['mean']:.4f}\n")
            f.write(f"Std Pressure: {report['prediction_stats']['std']:.4f}\n")
            f.write(f"Min Pressure: {report['prediction_stats']['min']:.4f}\n")
            f.write(f"Max Pressure: {report['prediction_stats']['max']:.4f}\n")
        
        print(f"✓ Saved text report to {text_file}")


if __name__ == "__main__":
    print("="*60)
    print("STEP 6: TEST PREDICTION")
    print("="*60)
    
    predictor = TestPredictor(data_dir='.')
    
    # Load model
    predictor.load_model()
    
    # Load test data
    test_df = predictor.load_test_data()
    
    # Validate test data
    test_validation = predictor.validate_test_data(test_df)
    
    # Generate predictions
    submission_df = predictor.generate_predictions(test_df)
    
    # Load sample submission for validation
    print("\nLoading sample submission...")
    sample_submission = pd.read_csv(predictor.data_dir / 'sample_submission.csv')
    print(f"✓ Loaded sample submission: {len(sample_submission):,} rows")
    
    # Validate submission format
    submission_validation = predictor.validate_submission(submission_df, sample_submission)
    
    # Save submission
    predictor.save_submission(submission_df)
    
    # Save report
    all_validation = {**test_validation, **submission_validation}
    predictor.save_prediction_report(all_validation, submission_df)
    
    print("\n" + "="*60)
    print("STEP 6 COMPLETE: Test Predictions Generated")
    print("="*60)
