"""
Data Loader and Validator for Ventilator Pressure Prediction
Loads and validates train.csv, test.csv, and sample_submission.csv
Ensures data integrity without silently fixing errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DataValidator:
    """Validates ventilator dataset structure and integrity"""
    
    EXPECTED_COLUMNS = {
        'train': ['id', 'breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out', 'pressure'],
        'test': ['id', 'breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out'],
        'sample_submission': ['id', 'pressure']
    }
    
    EXPECTED_R_VALUES = [5, 20, 50]
    EXPECTED_C_VALUES = [10, 20, 50]
    SAMPLES_PER_BREATH = 80
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def load_data(self, dataset: str) -> pd.DataFrame:
        """Load dataset without modifications"""
        file_path = self.data_dir / f"{dataset}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        print(f"Loading {dataset}.csv...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} rows")
        return df
    
    def validate_columns(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Check if all expected columns are present"""
        expected = self.EXPECTED_COLUMNS[dataset]
        actual = df.columns.tolist()
        
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        
        result = {
            'passed': len(missing) == 0 and len(extra) == 0,
            'expected': expected,
            'actual': actual,
            'missing': list(missing),
            'extra': list(extra)
        }
        
        if not result['passed']:
            print(f"❌ Column validation FAILED for {dataset}")
            if missing:
                print(f"   Missing columns: {missing}")
            if extra:
                print(f"   Extra columns: {extra}")
        else:
            print(f"✓ Column validation passed for {dataset}")
            
        return result
    
    def validate_nulls(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Check for null values"""
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].to_dict()
        
        result = {
            'passed': len(null_cols) == 0,
            'null_columns': null_cols,
            'total_nulls': int(null_counts.sum())
        }
        
        if not result['passed']:
            print(f"❌ Null validation FAILED for {dataset}")
            for col, count in null_cols.items():
                print(f"   {col}: {count:,} nulls ({count/len(df)*100:.2f}%)")
        else:
            print(f"✓ No null values in {dataset}")
            
        return result
    
    def validate_row_count(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Validate expected row counts"""
        expected_counts = {
            'train': 6_036_000,
            'test': 4_024_000
        }
        
        if dataset not in expected_counts:
            return {'passed': True, 'note': 'No expected count for this dataset'}
        
        expected = expected_counts[dataset]
        actual = len(df)
        
        result = {
            'passed': actual == expected,
            'expected': expected,
            'actual': actual,
            'difference': actual - expected
        }
        
        if not result['passed']:
            print(f"❌ Row count mismatch for {dataset}")
            print(f"   Expected: {expected:,}, Got: {actual:,}, Diff: {result['difference']:,}")
        else:
            print(f"✓ Row count matches expected: {actual:,}")
            
        return result
    
    def validate_breath_structure(self, df: pd.DataFrame) -> Dict:
        """Validate that each breath has exactly 80 samples"""
        if 'breath_id' not in df.columns:
            return {'passed': False, 'error': 'breath_id column not found'}
        
        breath_counts = df.groupby('breath_id').size()
        invalid_breaths = breath_counts[breath_counts != self.SAMPLES_PER_BREATH]
        
        result = {
            'passed': len(invalid_breaths) == 0,
            'total_breaths': len(breath_counts),
            'invalid_breath_count': len(invalid_breaths),
            'expected_samples': self.SAMPLES_PER_BREATH
        }
        
        if not result['passed']:
            print(f"❌ Breath structure validation FAILED")
            print(f"   {len(invalid_breaths)} breaths don't have {self.SAMPLES_PER_BREATH} samples")
            print(f"   Sample invalid breaths:")
            for breath_id, count in invalid_breaths.head(5).items():
                print(f"      breath_id {breath_id}: {count} samples")
        else:
            print(f"✓ All {result['total_breaths']:,} breaths have {self.SAMPLES_PER_BREATH} samples")
            
        return result
    
    def validate_ordering(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Validate chronological ordering within breaths"""
        if 'breath_id' not in df.columns or 'time_step' not in df.columns:
            return {'passed': False, 'error': 'Required columns not found'}
        
        ordering_issues = []
        
        for breath_id, group in df.groupby('breath_id'):
            time_steps = group['time_step'].values
            if not np.all(time_steps[:-1] <= time_steps[1:]):
                ordering_issues.append(breath_id)
                if len(ordering_issues) <= 5:  # Log first 5
                    print(f"   Ordering issue in breath_id {breath_id}")
        
        result = {
            'passed': len(ordering_issues) == 0,
            'breaths_with_issues': len(ordering_issues),
            'sample_issues': ordering_issues[:10]
        }
        
        if not result['passed']:
            print(f"❌ Ordering validation FAILED for {dataset}")
            print(f"   {len(ordering_issues)} breaths have non-chronological time_steps")
        else:
            print(f"✓ Time steps are chronologically ordered within breaths")
            
        return result
    
    def validate_rc_combinations(self, df: pd.DataFrame) -> Dict:
        """Validate all 9 R-C combinations are present"""
        if 'R' not in df.columns or 'C' not in df.columns:
            return {'passed': False, 'error': 'R or C column not found'}
        
        rc_combinations = df.groupby(['R', 'C']).size().reset_index(name='count')
        
        expected_combinations = [
            (r, c) for r in self.EXPECTED_R_VALUES for c in self.EXPECTED_C_VALUES
        ]
        
        found_combinations = set(zip(rc_combinations['R'], rc_combinations['C']))
        missing = set(expected_combinations) - found_combinations
        extra = found_combinations - set(expected_combinations)
        
        result = {
            'passed': len(missing) == 0 and len(extra) == 0,
            'expected_count': len(expected_combinations),
            'found_count': len(found_combinations),
            'missing': list(missing),
            'extra': list(extra),
            'distribution': rc_combinations.to_dict('records')
        }
        
        if not result['passed']:
            print(f"❌ R-C combination validation FAILED")
            if missing:
                print(f"   Missing combinations: {missing}")
            if extra:
                print(f"   Unexpected combinations: {extra}")
        else:
            print(f"✓ All 9 R-C combinations present")
            print("   Distribution:")
            for row in rc_combinations.itertuples():
                print(f"      R={row.R}, C={row.C}: {row.count:,} samples")
            
        return result
    
    def validate_id_uniqueness(self, df: pd.DataFrame, dataset: str) -> Dict:
        """Check if id column has unique values"""
        if 'id' not in df.columns:
            return {'passed': False, 'error': 'id column not found'}
        
        duplicates = df['id'].duplicated().sum()
        
        result = {
            'passed': duplicates == 0,
            'duplicate_count': int(duplicates),
            'unique_ids': int(df['id'].nunique()),
            'total_ids': len(df)
        }
        
        if not result['passed']:
            print(f"❌ ID uniqueness validation FAILED for {dataset}")
            print(f"   {duplicates:,} duplicate IDs found")
        else:
            print(f"✓ All IDs are unique in {dataset}")
            
        return result
    
    def run_full_audit(self, datasets: List[str] = None) -> Dict:
        """Run complete audit on specified datasets"""
        if datasets is None:
            datasets = ['train', 'test', 'sample_submission']
        
        audit_results = {}
        
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Auditing {dataset}.csv")
            print(f"{'='*60}")
            
            try:
                df = self.load_data(dataset)
                
                results = {
                    'columns': self.validate_columns(df, dataset),
                    'nulls': self.validate_nulls(df, dataset),
                    'row_count': self.validate_row_count(df, dataset),
                    'id_uniqueness': self.validate_id_uniqueness(df, dataset)
                }
                
                # Additional validations for train/test datasets
                if dataset in ['train', 'test']:
                    results['breath_structure'] = self.validate_breath_structure(df)
                    results['ordering'] = self.validate_ordering(df, dataset)
                    results['rc_combinations'] = self.validate_rc_combinations(df)
                
                # Overall pass/fail
                all_passed = all(
                    result.get('passed', False) 
                    for result in results.values()
                )
                
                results['overall_passed'] = all_passed
                audit_results[dataset] = results
                
                print(f"\n{'='*60}")
                if all_passed:
                    print(f"✓ {dataset}.csv: ALL CHECKS PASSED")
                else:
                    print(f"❌ {dataset}.csv: SOME CHECKS FAILED")
                print(f"{'='*60}")
                
            except Exception as e:
                print(f"❌ Error auditing {dataset}: {str(e)}")
                audit_results[dataset] = {
                    'error': str(e),
                    'overall_passed': False
                }
        
        self.validation_results = audit_results
        return audit_results
    
    def generate_data_dictionary(self, df: pd.DataFrame) -> Dict:
        """Generate data dictionary with column statistics"""
        dictionary = {}
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                })
            else:
                col_info.update({
                    'unique_count': int(df[col].nunique()),
                    'sample_values': df[col].value_counts().head(5).to_dict()
                })
            
            dictionary[col] = col_info
        
        return dictionary
    
    def save_audit_report(self, output_path: str = 'reports/model/data_audit_report.json'):
        """Save audit results to JSON file"""
        output_file = self.data_dir / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy bools to Python bools
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        converted_results = convert_types(self.validation_results)
        
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        print(f"\n✓ Audit report saved to {output_file}")
        
    def save_data_dictionary(self, output_path: str = 'reports/model/data_dictionary.json'):
        """Generate and save data dictionary"""
        output_file = self.data_dir / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating data dictionary...")
        train_df = self.load_data('train')
        dictionary = self.generate_data_dictionary(train_df)
        
        with open(output_file, 'w') as f:
            json.dump(dictionary, f, indent=2)
        
        print(f"✓ Data dictionary saved to {output_file}")


if __name__ == "__main__":
    # Run data audit
    validator = DataValidator(data_dir='.')
    
    print("="*60)
    print("VENTILATOR PRESSURE PREDICTION - DATA AUDIT")
    print("="*60)
    
    # Run full audit
    audit_results = validator.run_full_audit(['train', 'test', 'sample_submission'])
    
    # Save results
    validator.save_audit_report()
    validator.save_data_dictionary()
    
    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)
