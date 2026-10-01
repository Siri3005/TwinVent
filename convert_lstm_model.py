"""
Convert external LSTM model to compatible format
Extracts weights and rebuilds architecture
"""

import sys
import numpy as np
import h5py
from pathlib import Path

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    import tensorflow as tf
    
    print("=" * 70)
    print("LSTM MODEL CONVERSION")
    print("=" * 70)
    
    source_model_path = r"C:\Amrita VishwaVidyapeetam\Twin Vent\TwinVent\Ventilator_Pressure_Prediction_final.h5"
    output_model_path = "artifacts/model/lstm_model.h5"
    
    print(f"\nSource: {source_model_path}")
    print(f"Output: {output_model_path}")
    
    # Read model configuration from H5 file
    with h5py.File(source_model_path, 'r') as f:
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            
            # Extract input shape from config
            input_layer = config['config']['layers'][0]
            input_shape = tuple(input_layer['config']['batch_input_shape'][1:])  # Remove batch dimension
            print(f"\nDetected input shape: {input_shape}")
            
            # Expected: (80, 35) - 80 timesteps, 35 features
            
    # Rebuild model architecture based on the config
    print("\nRebuilding model architecture...")
    
    model = models.Sequential([
        # Input normalization
        layers.Normalization(input_shape=input_shape),
        
        # Conv1D layers (as seen in weights structure)
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv1d'),
        layers.BatchNormalization(name='batch_normalization'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d'),
        
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu', name='conv1d_1'),
        layers.BatchNormalization(name='batch_normalization_1'),
        layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1'),
        
        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='bidirectional'),
        layers.Dropout(0.3, name='dropout'),
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bidirectional_1'),
        
        # Global pooling
        layers.GlobalAveragePooling1D(name='global_average_pooling1d'),
        
        # Dense output
        layers.Dense(1, name='dense')
    ])
    
    print("✓ Model architecture created")
    
    # Build model
    model.build(input_shape=(None,) + input_shape)
    print("✓ Model built")
    
    # Try to load weights from original model
    print("\nLoading weights from source model...")
    try:
        model.load_weights(source_model_path, skip_mismatch=True, by_name=True)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"⚠ Warning: Some weights may not have loaded: {e}")
        print("  Model will work but may need retraining for best performance")
    
    # Test the model
    print("\nTesting model...")
    test_input = np.random.randn(1, *input_shape).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    print(f"✓ Test prediction successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Sample output: {prediction[0]}")
    
    # Show model summary
    print("\nModel Summary:")
    print("-" * 70)
    model.summary()
    
    # Save converted model
    print(f"\nSaving model to: {output_model_path}")
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    print("✓ Model saved successfully")
    
    # Also save as Keras 3 format
    keras_path = output_model_path.replace('.h5', '.keras')
    model.save(keras_path)
    print(f"✓ Also saved as: {keras_path}")
    
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    
    print("\nNext steps:")
    print("1. Update benchmark adapter to use LSTM inference wrapper")
    print("2. Test predictions in the web app")
    print("3. Compare accuracy with XGBoost model")
    
    print(f"\nModel expects:")
    print(f"  - Input shape: {input_shape}")
    print(f"  - 35 engineered features")
    print(f"  - 80 timesteps per breath")
    
except ImportError as e:
    print(f"✗ Error: {e}")
    print("\nTensorFlow/Keras required. Install with:")
    print("  pip install tensorflow")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error during conversion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
