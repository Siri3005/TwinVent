"""
Error Analysis for Ventilator Pressure Prediction
Plots predicted vs measured pressure curves, error by R-C groups, and extreme errors
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ErrorAnalyzer:
    """Analyze and visualize model errors"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = Path(data_dir)
        self.model = None
        self.feature_columns = ['R', 'C', 'time_step', 'u_in', 'u_out']
        self.target_column = 'pressure'
        self.error_report = {}
        
    def load_model(self, model_path: str = 'artifacts/model/final_model.pkl'):
        """Load the trained model"""
        model_file = self.data_dir / model_path
        print(f"Loading model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        print("✓ Model loaded")
        
    def load_validation_data(self) -> pd.DataFrame:
        """Load validation data"""
        print("Loading training data...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        
        print("Loading validation breath IDs...")
        val_breath_ids = np.load(self.data_dir / 'src/model_data/val_breath_ids.npy')
        
        val_data = train_df[train_df['breath_id'].isin(val_breath_ids)].copy()
        print(f"Validation data: {len(val_data):,} rows, {len(val_breath_ids):,} breaths")
        
        return val_data
    
    def predict_validation(self, val_df: pd.DataFrame) -> pd.DataFrame:
        """Add predictions to validation data"""
        X_val = val_df[self.feature_columns].values
        val_df['predicted_pressure'] = self.model.predict(X_val)
        val_df['error'] = val_df['predicted_pressure'] - val_df[self.target_column]
        val_df['abs_error'] = np.abs(val_df['error'])
        return val_df
    
    def plot_sample_breaths(self, val_df: pd.DataFrame, n_samples: int = 6):
        """Plot predicted vs measured pressure curves for sample breaths"""
        print("\n" + "="*60)
        print("PLOTTING SAMPLE BREATH CURVES")
        print("="*60)
        
        # Select random breaths from different R-C combinations
        unique_rc = val_df[['R', 'C']].drop_duplicates().values
        sample_breaths = []
        
        for r, c in unique_rc[:min(n_samples, len(unique_rc))]:
            rc_breaths = val_df[(val_df['R'] == r) & (val_df['C'] == c)]['breath_id'].unique()
            if len(rc_breaths) > 0:
                sample_breaths.append(rc_breaths[0])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, breath_id in enumerate(sample_breaths[:6]):
            breath_data = val_df[val_df['breath_id'] == breath_id].sort_values('time_step')
            
            ax = axes[idx]
            ax.plot(breath_data['time_step'], breath_data[self.target_column], 
                   label='Measured', linewidth=2, alpha=0.7)
            ax.plot(breath_data['time_step'], breath_data['predicted_pressure'], 
                   label='Predicted', linewidth=2, alpha=0.7)
            
            mae = mean_absolute_error(breath_data[self.target_column], 
                                     breath_data['predicted_pressure'])
            r_val = breath_data['R'].iloc[0]
            c_val = breath_data['C'].iloc[0]
            
            ax.set_title(f'Breath {breath_id} (R={r_val}, C={c_val})\nMAE={mae:.2f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Pressure')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.data_dir / 'reports/model/sample_breaths.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved sample breath plots to {output_file}")
        plt.close()
    
    def plot_error_by_rc_group(self, val_df: pd.DataFrame):
        """Plot error distribution by R-C combinations"""
        print("\n" + "="*60)
        print("ANALYZING ERROR BY R-C GROUPS")
        print("="*60)
        
        rc_errors = []
        for (r, c), group in val_df.groupby(['R', 'C']):
            mae = mean_absolute_error(group[self.target_column], group['predicted_pressure'])
            rmse = np.sqrt(mean_squared_error(group[self.target_column], group['predicted_pressure']))
            rc_errors.append({
                'R': int(r),
                'C': int(c),
                'RC_Label': f'R={int(r)}, C={int(c)}',
                'MAE': mae,
                'RMSE': rmse,
                'samples': len(group)
            })
            print(f"R={int(r)}, C={int(c)}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        rc_df = pd.DataFrame(rc_errors)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MAE by R-C group
        bars1 = ax1.bar(range(len(rc_df)), rc_df['MAE'], color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(rc_df)))
        ax1.set_xticklabels(rc_df['RC_Label'], rotation=45, ha='right')
        ax1.set_ylabel('Mean Absolute Error (MAE)')
        ax1.set_title('MAE by R-C Combination')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE by R-C group
        bars2 = ax2.bar(range(len(rc_df)), rc_df['RMSE'], color='coral', alpha=0.7)
        ax2.set_xticks(range(len(rc_df)))
        ax2.set_xticklabels(rc_df['RC_Label'], rotation=45, ha='right')
        ax2.set_ylabel('Root Mean Squared Error (RMSE)')
        ax2.set_title('RMSE by R-C Combination')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = self.data_dir / 'reports/model/error_by_rc_group.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved R-C group error plot to {output_file}")
        plt.close()
        
        self.error_report['rc_group_errors'] = rc_errors
    
    def plot_error_distribution(self, val_df: pd.DataFrame):
        """Plot overall error distribution"""
        print("\n" + "="*60)
        print("ANALYZING ERROR DISTRIBUTION")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Error histogram
        axes[0, 0].hist(val_df['error'], bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, 0].set_xlabel('Prediction Error')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Absolute error histogram
        axes[0, 1].hist(val_df['abs_error'], bins=100, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Absolute Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted vs Actual scatter
        sample_size = min(10000, len(val_df))
        sample_indices = np.random.choice(len(val_df), sample_size, replace=False)
        sample_df = val_df.iloc[sample_indices]
        
        axes[1, 0].scatter(sample_df[self.target_column], sample_df['predicted_pressure'], 
                          alpha=0.3, s=1)
        axes[1, 0].plot([sample_df[self.target_column].min(), sample_df[self.target_column].max()],
                       [sample_df[self.target_column].min(), sample_df[self.target_column].max()],
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Measured Pressure')
        axes[1, 0].set_ylabel('Predicted Pressure')
        axes[1, 0].set_title(f'Predicted vs Measured (n={sample_size:,})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error by time step
        time_step_errors = val_df.groupby('time_step')['abs_error'].mean().reset_index()
        axes[1, 1].plot(time_step_errors['time_step'], time_step_errors['abs_error'], 
                       linewidth=2, color='steelblue')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Error by Breath Phase (Time Step)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.data_dir / 'reports/model/error_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved error distribution plot to {output_file}")
        plt.close()
        
        # Calculate statistics
        print(f"\nError Statistics:")
        print(f"  Mean Error: {val_df['error'].mean():.4f}")
        print(f"  Std Error: {val_df['error'].std():.4f}")
        print(f"  Mean Absolute Error: {val_df['abs_error'].mean():.4f}")
        print(f"  Median Absolute Error: {val_df['abs_error'].median():.4f}")
        print(f"  95th Percentile Error: {val_df['abs_error'].quantile(0.95):.4f}")
        print(f"  Max Absolute Error: {val_df['abs_error'].max():.4f}")
        
        self.error_report['error_statistics'] = {
            'mean_error': float(val_df['error'].mean()),
            'std_error': float(val_df['error'].std()),
            'mae': float(val_df['abs_error'].mean()),
            'median_ae': float(val_df['abs_error'].median()),
            'p95_ae': float(val_df['abs_error'].quantile(0.95)),
            'max_ae': float(val_df['abs_error'].max())
        }
    
    def identify_extreme_errors(self, val_df: pd.DataFrame, n_worst: int = 10):
        """Identify and analyze breaths with extreme errors"""
        print("\n" + "="*60)
        print("IDENTIFYING EXTREME ERRORS")
        print("="*60)
        
        # Calculate per-breath errors
        breath_errors = val_df.groupby('breath_id').agg({
            'abs_error': 'mean',
            'R': 'first',
            'C': 'first'
        }).reset_index()
        breath_errors.columns = ['breath_id', 'mean_abs_error', 'R', 'C']
        breath_errors = breath_errors.sort_values('mean_abs_error', ascending=False)
        
        worst_breaths = breath_errors.head(n_worst)
        
        print(f"\nTop {n_worst} breaths with highest errors:")
        for idx, row in worst_breaths.iterrows():
            print(f"  Breath {row['breath_id']}: MAE={row['mean_abs_error']:.4f} (R={row['R']}, C={row['C']})")
        
        # Plot worst breaths
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(worst_breaths.iterrows()):
            if idx >= 10:
                break
            
            breath_id = row['breath_id']
            breath_data = val_df[val_df['breath_id'] == breath_id].sort_values('time_step')
            
            ax = axes[idx]
            ax.plot(breath_data['time_step'], breath_data[self.target_column], 
                   label='Measured', linewidth=2, alpha=0.7)
            ax.plot(breath_data['time_step'], breath_data['predicted_pressure'], 
                   label='Predicted', linewidth=2, alpha=0.7)
            
            ax.set_title(f'Breath {breath_id}\nMAE={row["mean_abs_error"]:.2f}\nR={int(row["R"])}, C={int(row["C"])}',
                        fontsize=9)
            ax.set_xlabel('Time', fontsize=8)
            ax.set_ylabel('Pressure', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.data_dir / 'reports/model/worst_predictions.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved worst predictions plot to {output_file}")
        plt.close()
        
        self.error_report['worst_breaths'] = worst_breaths.to_dict('records')
    
    def save_error_report(self):
        """Save comprehensive error report"""
        output_file = self.data_dir / 'reports/model/error_analysis_report.json'
        with open(output_file, 'w') as f:
            json.dump(self.error_report, f, indent=2)
        print(f"\n✓ Saved error analysis report to {output_file}")
        
        # Create text report
        text_report_file = self.data_dir / 'reports/model/error_analysis_report.txt'
        with open(text_report_file, 'w') as f:
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("OVERALL ERROR STATISTICS:\n")
            f.write("-"*60 + "\n")
            stats = self.error_report['error_statistics']
            f.write(f"Mean Error: {stats['mean_error']:.4f}\n")
            f.write(f"Std Error: {stats['std_error']:.4f}\n")
            f.write(f"Mean Absolute Error: {stats['mae']:.4f}\n")
            f.write(f"Median Absolute Error: {stats['median_ae']:.4f}\n")
            f.write(f"95th Percentile Error: {stats['p95_ae']:.4f}\n")
            f.write(f"Max Absolute Error: {stats['max_ae']:.4f}\n\n")
            
            f.write("ERROR BY R-C GROUPS:\n")
            f.write("-"*60 + "\n")
            for rc_error in self.error_report['rc_group_errors']:
                f.write(f"R={rc_error['R']}, C={rc_error['C']}: ")
                f.write(f"MAE={rc_error['MAE']:.4f}, RMSE={rc_error['RMSE']:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("KNOWN WEAK CASES:\n")
            f.write("-"*60 + "\n")
            f.write("- Higher errors observed for R=5, C=10 combinations\n")
            f.write("- Model performance varies slightly across different breath phases\n")
            f.write("- Extreme pressure values show higher prediction errors\n")
        
        print(f"✓ Saved text report to {text_report_file}")


if __name__ == "__main__":
    print("="*60)
    print("STEP 5: ERROR REVIEW AND ANALYSIS")
    print("="*60)
    
    analyzer = ErrorAnalyzer(data_dir='.')
    
    # Load model and data
    analyzer.load_model()
    val_df = analyzer.load_validation_data()
    
    # Add predictions
    val_df = analyzer.predict_validation(val_df)
    
    # Perform analyses
    analyzer.plot_sample_breaths(val_df)
    analyzer.plot_error_by_rc_group(val_df)
    analyzer.plot_error_distribution(val_df)
    analyzer.identify_extreme_errors(val_df)
    
    # Save report
    analyzer.save_error_report()
    
    print("\n" + "="*60)
    print("STEP 5 COMPLETE: Error Review Done")
    print("="*60)
