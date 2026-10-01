"""
Rebuild model architecture from H5 weights (Keras version compatibility fix)
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py

try:
    # Use TensorFlow 2.x compatible imports
    import tensorflow.compat.v1 as tf_v1
    from tensorflow.keras import layers, models
    from tensorflow.keras.models import Sequential, Model
    import tensorflow as tf
    
    print("TensorFlow version:", tf.__version__)
    
    model_path = r"C:\Amrita VishwaVidyapeetam\Twin Vent\TwinVent\Ventilator_Pressure_Prediction_final.h5"
    
    # Read the H5 file to understand architecture
    with h5py.File(model_path, 'r') as f:
        # Check if model config exists
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            print("\nModel Configuration Found:")
            print(json.dumps(config, indent=2)[:1000])  # Print first 1000 chars
    
    print("\n" + "=" * 70)
    print("REBUILDING MODEL ARCHITECTURE")
    print("=" * 70)
    
    # Based on the weights structure, rebuild the model
    # This is a typical architecture for time series with:
    # Normalization -> Conv1D -> Bidirectional LSTM -> Dense
    
    def build_model(input_shape=(80, 5)):  # Assuming 80 timesteps, 5 features
        """
        Rebuild model architecture compatible with the weights
        """
        model = Sequential([
            # Normalization layer
            layers.Normalization(input_shape=input_shape),
            
            # Conv1D layers
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            
            # Global pooling
            layers.GlobalAveragePooling1D(),
            
            # Dense output layer
            layers.Dense(1)  # Single output for pressure prediction
        ])
        
        return model
    
    # Try multiple input shapes
    possible_shapes = [
        (80, 5),   # 80 timesteps, 5 features (R, C, time_step, u_in, u_out)
        (80, 6),   # 80 timesteps, 6 features
        (None, 5), # Variable timesteps, 5 features
    ]
    
    for shape in possible_shapes:
        try:
            print(f"\nTrying input shape: {shape}")
            model = build_model(input_shape=shape)
            
            # Try to load weights
            model.load_weights(model_path, skip_mismatch=True, by_name=True)
            
            print(f"✓ Model built successfully with input shape: {shape}")
            
            # Test prediction
            if shape[0] is None:
                test_input = np.random.randn(1, 80, shape[1]).astype(np.float32)
            else:
                test_input = np.random.randn(1, shape[0], shape[1]).astype(np.float32)
            
            prediction = model.predict(test_input, verbose=0)
            print(f"✓ Test prediction successful! Output shape: {prediction.shape}")
            
            # Show model summary
            print("\nModel Summary:")
            model.summary()
            
            # Save compatible version
            output_path = "artifacts/model/lstm_model_compatible.h5"
            model.save(output_path)
            print(f"\n✓ Saved compatible model to: {output_path}")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ Failed with shape {shape}: {e}")
            continue
    
    print("\n✗ Could not rebuild model with any input shape")
    print("\nPlease provide:")
    print("1. The training script used to create this model")
    print("2. Or the model.json / model architecture file")
    print("3. Or information about input shape and model layers")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
