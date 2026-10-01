"""
Quick diagnostic script to test model performance on specific breaths
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def load_and_test_breath(breath_id: int, dataset: str = 'train'):
    """Load a specific breath and test current model"""
    
    print(f"\n{'='*60}")
    print(f"TESTING BREATH #{breath_id} from {dataset}.csv")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(f'{dataset}.csv')
    breath_df = df[df['breath_id'] == breath_id].copy()
    
    if len(breath_df) == 0:
        print(f"❌ Breath {breath_id} not found in {dataset}.csv")
        return
    
    breath_df = breath_df.sort_values('time_step')
    
    # Display breath info
    print(f"\nBreath Information:")
    print(f"  Samples: {len(breath_df)}")
    print(f"  R: {breath_df['R'].iloc[0]}")
    print(f"  C: {breath_df['C'].iloc[0]}")
    print(f"  Time range: {breath_df['time_step'].min():.3f} to {breath_df['time_step'].max():.3f} s")
    print(f"  Pressure range: {breath_df['pressure'].min():.2f} to {breath_df['pressure'].max():.2f} cmH₂O")
    
    # Load model
    model_path = Path('artifacts/model/final_model.pkl')
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Prepare features
    feature_cols = ['R', 'C', 'time_step', 'u_in', 'u_out']
    X = breath_df[feature_cols].values
    y_true = breath_df['pressure'].values
    
    # Predict
    y_pred = model.predict(X)
    
    # Calculate errors
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    max_error = np.max(np.abs(y_true - y_pred))
    mean_error = np.mean(y_true - y_pred)
    
    print(f"\nPrediction Metrics:")
    print(f"  MAE:        {mae:.2f} cmH₂O")
    print(f"  RMSE:       {rmse:.2f} cmH₂O")
    print(f"  Max Error:  {max_error:.2f} cmH₂O")
    print(f"  Mean Error: {mean_error:.2f} cmH₂O (+ = overpredicting)")
    
    # Find worst time points
    errors = np.abs(y_true - y_pred)
    worst_idx = np.argsort(errors)[-5:][::-1]
    
    print(f"\nWorst 5 Time Points:")
    print(f"  {'Time':>6s} {'True':>8s} {'Pred':>8s} {'Error':>8s} {'u_in':>8s} {'u_out':>6s}")
    for idx in worst_idx:
        print(f"  {breath_df['time_step'].iloc[idx]:6.3f} "
              f"{y_true[idx]:8.2f} {y_pred[idx]:8.2f} "
              f"{errors[idx]:8.2f} {breath_df['u_in'].iloc[idx]:8.2f} "
              f"{breath_df['u_out'].iloc[idx]:6.0f}")
    
    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Pressure comparison
    plt.subplot(3, 1, 1)
    plt.plot(breath_df['time_step'], y_true, 'k-', linewidth=2, label='Measured (True)')
    plt.plot(breath_df['time_step'], y_pred, 'orange', linestyle='--', linewidth=2, label='Model Prediction')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (cmH₂O)')
    plt.title(f'Breath #{breath_id} - Pressure Prediction (R={breath_df["R"].iloc[0]}, C={breath_df["C"].iloc[0]})', 
              fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction error
    plt.subplot(3, 1, 2)
    plt.plot(breath_df['time_step'], y_true - y_pred, 'r-', linewidth=1.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.fill_between(breath_df['time_step'], 0, y_true - y_pred, alpha=0.3, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (cmH₂O)')
    plt.title(f'Prediction Error (Positive = Model Underpredicts)', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Control inputs
    plt.subplot(3, 1, 3)
    ax1 = plt.gca()
    ax1.plot(breath_df['time_step'], breath_df['u_in'], 'b-', linewidth=1.5, label='u_in (flow)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('u_in (flow control)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(breath_df['time_step'], breath_df['u_out'], 'g-', linewidth=1.5, label='u_out (valve)')
    ax2.set_ylabel('u_out (valve state)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(-0.1, 1.1)
    
    plt.title('Control Inputs', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(f'reports/model/breath_{breath_id}_diagnosis.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")
    
    plt.show()
    
    return {
        'breath_id': breath_id,
        'R': breath_df['R'].iloc[0],
        'C': breath_df['C'].iloc[0],
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
        'mean_error': mean_error
    }


def compare_multiple_breaths(breath_ids: list):
    """Test multiple breaths and compare"""
    results = []
    
    for breath_id in breath_ids:
        result = load_and_test_breath(breath_id)
        if result:
            results.append(result)
    
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        print(f"\nAverage MAE: {df['mae'].mean():.2f} cmH₂O")
        print(f"Worst MAE: {df['mae'].max():.2f} cmH₂O (Breath #{df.loc[df['mae'].idxmax(), 'breath_id']:.0f})")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific breath from command line
        breath_id = int(sys.argv[1])
        load_and_test_breath(breath_id)
    else:
        # Test the problematic breath and some others
        print("Testing multiple breaths...")
        print("="*60)
        
        # Test breath #163 (the failing case from your screenshot)
        compare_multiple_breaths([
            163,    # Your problem case
            100,    # Random samples
            500,
            1000,
            5000,
        ])
