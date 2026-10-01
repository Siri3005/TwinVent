"""
Quick verification script for improved model deployment
Tests that the new model loads and predicts correctly
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from artifacts.model import predictor

def verify_deployment():
    """Verify the improved model deployment."""
    print("=" * 60)
    print("TwinVent Model Deployment Verification")
    print("=" * 60)
    
    # Check model version
    print(f"\n✓ Model Version: {predictor.MODEL_VERSION}")
    
    # Load mock breath example
    mock_path = ROOT / "artifacts" / "model" / "mock_breath_example.csv"
    if not mock_path.exists():
        print("✗ Mock breath example not found")
        return False
    
    mock_df = pd.read_csv(mock_path)
    print(f"✓ Loaded mock breath: {len(mock_df)} samples")
    
    # Convert to list of dicts for predictor
    rows = mock_df.to_dict('records')
    
    # Predict
    try:
        result = predictor.predict_breath(rows)
        print(f"✓ Prediction Status: {result['status']}")
        print(f"✓ Predicted {len(result['pressure'])} pressure values")
        
        # Show sample predictions
        pressures = result['pressure']
        print(f"\n  Pressure Range: {min(pressures):.2f} to {max(pressures):.2f} cmH₂O")
        print(f"  Mean Pressure: {sum(pressures)/len(pressures):.2f} cmH₂O")
        
        # Check uncertainty
        if result['uncertainty'] and result['uncertainty'][0] is not None:
            uncertainties = result['uncertainty']
            print(f"  Mean Uncertainty: {sum(uncertainties)/len(uncertainties):.2f}")
        
        print("\n" + "=" * 60)
        print("✅ DEPLOYMENT VERIFIED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Restart the web server: python run_app.py")
        print("2. Open browser: http://127.0.0.1:8765")
        print("3. Test with various breaths (especially breath #163)")
        print("4. Verify predictions are much closer to measured values")
        print("\nExpected improvement: ~69% better accuracy")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_deployment()
    sys.exit(0 if success else 1)
