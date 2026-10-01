"""
Script to load and inspect external H5 model
"""

import sys
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    
    model_path = r"C:\Amrita VishwaVidyapeetam\Twin Vent\TwinVent\Ventilator_Pressure_Prediction_final.h5"
    
    print(f"\nAttempting to load model from: {model_path}")
    print("-" * 70)
    
    # Try multiple loading methods
    try:
        # Method 1: Standard load
        model = keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully with keras.models.load_model")
    except Exception as e1:
        print(f"✗ Method 1 failed: {e1}")
        
        try:
            # Method 2: Load with custom objects
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'LSTM': keras.layers.LSTM})
            print("✓ Model loaded successfully with custom_objects")
        except Exception as e2:
            print(f"✗ Method 2 failed: {e2}")
            
            try:
                # Method 3: Load architecture and weights separately
                import h5py
                with h5py.File(model_path, 'r') as f:
                    print("\nH5 file structure:")
                    def print_structure(name, obj):
                        print(f"  {name}: {type(obj)}")
                    f.visititems(print_structure)
            except Exception as e3:
                print(f"✗ Could not inspect H5 file: {e3}")
            
            print("\n✗ All loading methods failed")
            sys.exit(1)
    
    # Model loaded successfully, inspect it
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    print(f"\nInput shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    print("\nModel architecture:")
    print("-" * 70)
    model.summary()
    
    print("\n" + "=" * 70)
    print("LAYER DETAILS")
    print("=" * 70)
    
    for i, layer in enumerate(model.layers):
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Type: {type(layer).__name__}")
        print(f"  Input shape: {layer.input_shape}")
        print(f"  Output shape: {layer.output_shape}")
        if hasattr(layer, 'units'):
            print(f"  Units: {layer.units}")
    
    # Test prediction with dummy data
    print("\n" + "=" * 70)
    print("TESTING PREDICTION")
    print("=" * 70)
    
    import numpy as np
    
    # Determine input shape
    if isinstance(model.input_shape, list):
        input_shape = model.input_shape[0]
    else:
        input_shape = model.input_shape
    
    print(f"\nInput shape: {input_shape}")
    
    # Create dummy input
    batch_size = 1
    if len(input_shape) == 3:  # (batch, timesteps, features)
        timesteps = input_shape[1] if input_shape[1] else 80
        n_features = input_shape[2]
        dummy_input = np.random.randn(batch_size, timesteps, n_features).astype(np.float32)
        print(f"Created dummy input: shape {dummy_input.shape}")
    elif len(input_shape) == 2:  # (batch, features)
        n_features = input_shape[1]
        dummy_input = np.random.randn(batch_size, n_features).astype(np.float32)
        print(f"Created dummy input: shape {dummy_input.shape}")
    else:
        print(f"Unexpected input shape: {input_shape}")
        sys.exit(1)
    
    # Make prediction
    prediction = model.predict(dummy_input, verbose=0)
    print(f"✓ Prediction successful!")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Output sample: {prediction[0][:5] if len(prediction[0]) > 5 else prediction[0]}")
    
    print("\n" + "=" * 70)
    print("MODEL READY FOR INTEGRATION")
    print("=" * 70)
    
    print(f"\nKey information for integration:")
    print(f"  - Expected input shape: {input_shape}")
    print(f"  - Expected features: {n_features}")
    print(f"  - Output shape: {model.output_shape}")
    print(f"  - Model type: {'LSTM/RNN' if any('lstm' in layer.name.lower() or 'gru' in layer.name.lower() for layer in model.layers) else 'Dense/CNN'}")
    
except ImportError as e:
    print(f"✗ Error: {e}")
    print("\nTensorFlow/Keras not installed. Install with:")
    print("  pip install tensorflow")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
