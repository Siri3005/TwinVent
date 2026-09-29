"""
Data Splitter for Ventilator Pressure Prediction
Splits data by complete breath_id groups into train and validation sets
Preserves chronological order within breaths and verifies R-C distribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict


class DataSplitter:
    """Splits data by breath_id while maintaining chronological order"""
    
    def __init__(self, data_dir: str = '.', random_seed: int = 42):
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed
        self.split_info = {}
        np.random.seed(random_seed)
        
    def load_train_data(self) -> pd.DataFrame:
        """Load the training dataset"""
        file_path = self.data_dir / 'train.csv'
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} rows with {df['breath_id'].nunique():,} unique breaths")
        return df
    
    def split_breath_ids(self, df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split breath_ids into train and validation sets
        
        Args:
            df: DataFrame with breath_id column
            val_ratio: Ratio of data to use for validation (default 0.2)
            
        Returns:
            Tuple of (train_breath_ids, val_breath_ids)
        """
        # Get unique breath IDs
        unique_breaths = df['breath_id'].unique()
        print(f"\nTotal unique breaths: {len(unique_breaths):,}")
        
        # Shuffle breath IDs
        np.random.shuffle(unique_breaths)
        
        # Calculate split point
        val_size = int(len(unique_breaths) * val_ratio)
        train_size = len(unique_breaths) - val_size
        
        # Split
        train_breath_ids = unique_breaths[:train_size]
        val_breath_ids = unique_breaths[train_size:]
        
        print(f"Train breaths: {len(train_breath_ids):,} ({(1-val_ratio)*100:.1f}%)")
        print(f"Val breaths: {len(val_breath_ids):,} ({val_ratio*100:.1f}%)")
        
        # Verify no overlap
        overlap = set(train_breath_ids) & set(val_breath_ids)
        if len(overlap) > 0:
            raise ValueError(f"Found {len(overlap)} breath_ids in both train and val!")
        
        print("✓ No overlap between train and validation breath IDs")
        
        return train_breath_ids, val_breath_ids
    
    def verify_chronological_order(self, df: pd.DataFrame) -> bool:
        """Verify that time_steps are in chronological order within each breath"""
        for breath_id, group in df.groupby('breath_id'):
            time_steps = group['time_step'].values
            if not np.all(time_steps[:-1] <= time_steps[1:]):
                print(f"❌ Chronological order violated in breath_id {breath_id}")
                return False
        
        print("✓ Chronological order maintained within all breaths")
        return True
    
    def check_rc_distribution(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """Check R-C combination distribution"""
        rc_dist = df.groupby(['R', 'C']).size().reset_index(name='count')
        rc_dist['percentage'] = (rc_dist['count'] / len(df) * 100).round(2)
        
        print(f"\n{dataset_name} R-C Distribution:")
        for _, row in rc_dist.iterrows():
            print(f"  R={row['R']}, C={row['C']}: {row['count']:,} samples ({row['percentage']:.2f}%)")
        
        return rc_dist.to_dict('records')
    
    def create_split_datasets(self, df: pd.DataFrame, train_breath_ids: np.ndarray, 
                             val_breath_ids: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train and validation datasets from breath IDs"""
        print("\nCreating split datasets...")
        
        train_df = df[df['breath_id'].isin(train_breath_ids)].copy()
        val_df = df[df['breath_id'].isin(val_breath_ids)].copy()
        
        print(f"Train dataset: {len(train_df):,} rows ({len(train_df)/len(df)*100:.2f}%)")
        print(f"Val dataset: {len(val_df):,} rows ({len(val_df)/len(df)*100:.2f}%)")
        
        # Sort by breath_id and time_step to maintain order
        train_df = train_df.sort_values(['breath_id', 'time_step']).reset_index(drop=True)
        val_df = val_df.sort_values(['breath_id', 'time_step']).reset_index(drop=True)
        
        return train_df, val_df
    
    def perform_split(self, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform complete train/validation split
        
        Args:
            val_ratio: Ratio of data to use for validation
            
        Returns:
            Tuple of (train_df, val_df)
        """
        print("="*60)
        print("PERFORMING TRAIN/VALIDATION SPLIT")
        print("="*60)
        print(f"Random seed: {self.random_seed}")
        print(f"Validation ratio: {val_ratio}")
        
        # Load data
        df = self.load_train_data()
        
        # Split breath IDs
        train_breath_ids, val_breath_ids = self.split_breath_ids(df, val_ratio)
        
        # Create datasets
        train_df, val_df = self.create_split_datasets(df, train_breath_ids, val_breath_ids)
        
        # Verify chronological order
        print("\nVerifying chronological order...")
        train_ordered = self.verify_chronological_order(train_df)
        val_ordered = self.verify_chronological_order(val_df)
        
        if not (train_ordered and val_ordered):
            raise ValueError("Chronological order check failed!")
        
        # Check R-C distributions
        train_rc_dist = self.check_rc_distribution(train_df, "Train")
        val_rc_dist = self.check_rc_distribution(val_df, "Validation")
        
        # Store split info
        self.split_info = {
            'random_seed': int(self.random_seed),
            'validation_ratio': float(val_ratio),
            'total_breaths': int(len(train_breath_ids) + len(val_breath_ids)),
            'train_breaths': int(len(train_breath_ids)),
            'val_breaths': int(len(val_breath_ids)),
            'train_rows': int(len(train_df)),
            'val_rows': int(len(val_df)),
            'train_rc_distribution': train_rc_dist,
            'val_rc_distribution': val_rc_dist,
            'chronological_order_maintained': True
        }
        
        return train_df, val_df
    
    def save_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   output_dir: str = 'src/model_data'):
        """Save split datasets and breath IDs"""
        output_path = self.data_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("SAVING SPLIT DATA")
        print("="*60)
        
        # Save breath IDs (not full datasets to save space)
        train_breath_ids = train_df['breath_id'].unique()
        val_breath_ids = val_df['breath_id'].unique()
        
        train_ids_file = output_path / 'train_breath_ids.npy'
        val_ids_file = output_path / 'val_breath_ids.npy'
        
        np.save(train_ids_file, train_breath_ids)
        np.save(val_ids_file, val_breath_ids)
        
        print(f"✓ Saved train breath IDs to {train_ids_file}")
        print(f"✓ Saved validation breath IDs to {val_ids_file}")
        
        # Save split info
        split_info_file = output_path / 'split_info.json'
        with open(split_info_file, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        
        print(f"✓ Saved split info to {split_info_file}")
        
        # Create a summary report
        report_file = self.data_dir / 'reports/model/split_report.txt'
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("TRAIN/VALIDATION SPLIT REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Random Seed: {self.split_info['random_seed']}\n")
            f.write(f"Validation Ratio: {self.split_info['validation_ratio']}\n\n")
            f.write(f"Total Breaths: {self.split_info['total_breaths']:,}\n")
            f.write(f"Train Breaths: {self.split_info['train_breaths']:,}\n")
            f.write(f"Validation Breaths: {self.split_info['val_breaths']:,}\n\n")
            f.write(f"Train Rows: {self.split_info['train_rows']:,}\n")
            f.write(f"Validation Rows: {self.split_info['val_rows']:,}\n\n")
            f.write("Chronological Order: MAINTAINED\n")
            f.write("No Overlap: VERIFIED\n\n")
            f.write("TRAIN R-C DISTRIBUTION:\n")
            for dist in self.split_info['train_rc_distribution']:
                f.write(f"  R={dist['R']}, C={dist['C']}: {dist['count']:,} ({dist['percentage']:.2f}%)\n")
            f.write("\nVALIDATION R-C DISTRIBUTION:\n")
            for dist in self.split_info['val_rc_distribution']:
                f.write(f"  R={dist['R']}, C={dist['C']}: {dist['count']:,} ({dist['percentage']:.2f}%)\n")
        
        print(f"✓ Saved split report to {report_file}")
        
        print("\n" + "="*60)
        print("SPLIT COMPLETE")
        print("="*60)
    
    def load_split(self, split_dir: str = 'src/model_data') -> Tuple[np.ndarray, np.ndarray]:
        """Load previously saved split breath IDs"""
        split_path = self.data_dir / split_dir
        
        train_ids_file = split_path / 'train_breath_ids.npy'
        val_ids_file = split_path / 'val_breath_ids.npy'
        
        if not train_ids_file.exists() or not val_ids_file.exists():
            raise FileNotFoundError("Split files not found. Run perform_split() first.")
        
        train_breath_ids = np.load(train_ids_file)
        val_breath_ids = np.load(val_ids_file)
        
        print(f"Loaded {len(train_breath_ids):,} train breath IDs")
        print(f"Loaded {len(val_breath_ids):,} validation breath IDs")
        
        return train_breath_ids, val_breath_ids


if __name__ == "__main__":
    # Create splitter with fixed random seed
    splitter = DataSplitter(data_dir='.', random_seed=42)
    
    # Perform split
    train_df, val_df = splitter.perform_split(val_ratio=0.2)
    
    # Save split
    splitter.save_split(train_df, val_df)
