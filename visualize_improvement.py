"""
Visualization script to compare current vs improved model predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def compare_models_on_breath(breath_id: int, dataset: str = 'train'):
    """Compare current and improved model predictions side-by-side"""
    
    # Load data
    df = pd.read_csv(f'{dataset}.csv')
    breath_df = df[df['breath_id'] == breath_id].copy().sort_values('time_step')
    
    if len(breath_df) == 0:
        print(f"Breath {breath_id} not found!")
        return
    
    # Load current model
    current_model_path = Path('artifacts/model/final_model.pkl')
    improved_model_path = Path('artifacts/model/improved_model.pkl')
    
    models = {}
    if current_model_path.exists():
        with open(current_model_path, 'rb') as f:
            models['Current'] = pickle.load(f)
    
    if improved_model_path.exists():
        with open(improved_model_path, 'rb') as f:
            models['Improved'] = pickle.load(f)
    
    if not models:
        print("No models found! Train models first.")
        return
    
    # Prepare features
    feature_cols = ['R', 'C', 'time_step', 'u_in', 'u_out']
    X = breath_df[feature_cols].values
    y_true = breath_df['pressure'].values
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Model Comparison: Breath #{breath_id} (R={breath_df["R"].iloc[0]}, C={breath_df["C"].iloc[0]})', 
                 fontsize=14, fontweight='bold')
    
    colors = {'Current': 'orange', 'Improved': 'green'}
    predictions = {}
    
    # Predict with each model
    for model_name, model in models.items():
        y_pred = model.predict(X)
        predictions[model_name] = y_pred
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Plot 1: Pressure curves
        axes[0, 0].plot(breath_df['time_step'], y_pred, 
                       color=colors[model_name], linestyle='--', linewidth=2, 
                       label=f'{model_name} (MAE={mae:.2f})', alpha=0.8)
    
    # Add measured pressure
    axes[0, 0].plot(breath_df['time_step'], y_true, 
                   'k-', linewidth=2.5, label='Measured (True)', zorder=10)
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('Pressure (cmH₂O)', fontsize=11)
    axes[0, 0].set_title('Pressure Predictions Comparison', fontsize=12)
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error comparison
    for model_name, y_pred in predictions.items():
        error = y_true - y_pred
        axes[0, 1].plot(breath_df['time_step'], error, 
                       color=colors[model_name], linewidth=2, 
                       label=f'{model_name}', alpha=0.8)
    
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Error (cmH₂O)', fontsize=11)
    axes[0, 1].set_title('Prediction Error Over Time\n(Positive = Underprediction)', fontsize=12)
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    for model_name, y_pred in predictions.items():
        error = y_true - y_pred
        axes[1, 0].hist(error, bins=30, alpha=0.6, 
                       color=colors[model_name], label=model_name, 
                       edgecolor='black', linewidth=0.5)
    
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Error (cmH₂O)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Error Distribution', fontsize=12)
    axes[1, 0].legend(loc='best', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Metrics comparison
    metrics_data = []
    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        max_err = np.max(np.abs(y_true - y_pred))
        metrics_data.append([model_name, mae, rmse, max_err])
    
    df_metrics = pd.DataFrame(metrics_data, columns=['Model', 'MAE', 'RMSE', 'Max Error'])
    
    x_pos = np.arange(len(df_metrics))
    width = 0.25
    
    metrics_to_plot = ['MAE', 'RMSE', 'Max Error']
    for i, metric in enumerate(metrics_to_plot):
        values = df_metrics[metric].values
        axes[1, 1].bar(x_pos + i*width, values, width, 
                      label=metric, alpha=0.8)
    
    axes[1, 1].set_xlabel('Model', fontsize=11)
    axes[1, 1].set_ylabel('Error (cmH₂O)', fontsize=11)
    axes[1, 1].set_title('Error Metrics Comparison', fontsize=12)
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(df_metrics['Model'])
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(f'reports/model/comparison_breath_{breath_id}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BREATH #{breath_id} COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(df_metrics.to_string(index=False))
    
    if len(predictions) == 2:
        current_mae = df_metrics[df_metrics['Model'] == 'Current']['MAE'].values[0]
        improved_mae = df_metrics[df_metrics['Model'] == 'Improved']['MAE'].values[0]
        improvement = (current_mae - improved_mae) / current_mae * 100
        print(f"\n✓ Improvement: {improvement:.1f}% better MAE")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        breath_id = int(sys.argv[1])
    else:
        breath_id = 163  # Default to the problematic breath
    
    print(f"Comparing models on breath #{breath_id}...")
    compare_models_on_breath(breath_id)
