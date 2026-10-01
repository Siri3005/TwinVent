"""
Test LSTM model integration
"""

from src.model_training.lstm_inference_wrapper import LSTMPressurePredictor
import pandas as pd

print("Testing LSTM Model Integration...")
print("=" * 70)

# Load predictor
predictor = LSTMPressurePredictor()
predictor.load_model('artifacts/model/lstm_model.h5')

# Load mock breath
df = pd.read_csv('artifacts/model/mock_breath_example.csv')
print(f"\nInput: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Make prediction
result = predictor.predict_breath(df, return_uncertainty=True)

print(f"\nPrediction Status: {result['status'].iloc[0]}")
print(f"Model Version: {result['model_version'].iloc[0]}")
print(f"Output: {len(result)} predictions")

if result['status'].iloc[0] == 'ok':
    print(f"\nPressure Statistics:")
    print(f"  Min: {result['pressure'].min():.2f} cmH₂O")
    print(f"  Max: {result['pressure'].max():.2f} cmH₂O")
    print(f"  Mean: {result['pressure'].mean():.2f} cmH₂O")
    print(f"  Std: {result['pressure'].std():.2f} cmH₂O")
    
    print(f"\nSample predictions (first 10):")
    print(result[['id', 'pressure']].head(10))
    
    print("\n✅ LSTM MODEL INTEGRATION SUCCESSFUL!")
else:
    print(f"\n✗ Prediction failed: {result['reason'].iloc[0]}")

print("=" * 70)
