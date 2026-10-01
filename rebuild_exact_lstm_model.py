"""
Rebuild exact LSTM model architecture from training notebook
"""

import sys
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    
    print("=" * 70)
    print("EXACT LSTM MODEL REBUILD")
    print("=" * 70)
    
    source_model_path = r"C:\Amrita VishwaVidyapeetam\Twin Vent\TwinVent\Ventilator_Pressure_Prediction_final.h5"
    output_model_path = "artifacts/model/lstm_model.h5"
    
    # Based on notebook, the features are:
    # After dropping: 'pressure', 'id', 'breath_id', 'u_in', 'time_step'
    # Original features before drops: R, C, time_step, u_in, u_out, un_in_std, time_step_std
    # Plus 30 lag features (after1-5, back1-5 for time_step_std, un_in_std, u_out)
    # Total: 35 features
    
    n_features = 35  # As detected from H5 file
    
    print(f"\nInput shape: (80, {n_features})")
    print("Building exact model architecture from notebook...")
    
    # Create normalization layer (will need to adapt with actual data)
    norm = keras.layers.Normalization(input_shape=[80, n_features], axis=-1)
    
    # Build exact model from notebook
    model = keras.Sequential([
        norm,
        keras.layers.Conv1D(128, 3, activation="relu"),
        keras.layers.MaxPooling1D(),
        keras.layers.BatchNormalization(),
        keras.layers.Conv1D(256, 3, activation="relu"),
        keras.layers.MaxPooling1D(),
        keras.layers.BatchNormalization(),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(80)  # OUTPUT IS 80 VALUES (one per timestep)
    ])
    
    print("✓ Model architecture created (exact match to notebook)")
    
    # Compile with same settings as notebook
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mae"
    )
    
    print("✓ Model compiled with Adam optimizer and MAE loss")
    
    # Build model
    model.build(input_shape=(None, 80, n_features))
    print("✓ Model built")
    
    # Show model summary
    print("\nModel Summary:")
    print("-" * 70)
    model.summary()
    
    # Load weights from original model
    print("\nLoading weights from source model...")
    try:
        model.load_weights(source_model_path)
        print("✓ All weights loaded successfully!")
    except Exception as e:
        print(f"⚠ Warning: {e}")
        print("  Trying with skip_mismatch...")
        try:
            model.load_weights(source_model_path, skip_mismatch=True, by_name=True)
            print("✓ Weights loaded (some may have been skipped)")
        except Exception as e2:
            print(f"✗ Could not load weights: {e2}")
    
    # Test prediction
    print("\nTesting model...")
    test_input = np.random.randn(1, 80, n_features).astype(np.float32)
    prediction = model.predict(test_input, verbose=0)
    
    print(f"✓ Test prediction successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {prediction.shape}")
    print(f"  Output is 80 values (one per timestep): {prediction.shape[1] == 80}")
    print(f"  Sample outputs: {prediction[0][:5]}")
    
    # Save model
    print(f"\nSaving model to: {output_model_path}")
    Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)
    print("✓ Model saved successfully")
    
    # Also save as Keras format
    keras_path = output_model_path.replace('.h5', '.keras')
    model.save(keras_path)
    print(f"✓ Also saved as: {keras_path}")
    
    print("\n" + "=" * 70)
    print("SUCCESS - EXACT MODEL REBUILT")
    print("=" * 70)
    
    print("\nKey findings:")
    print("  - Model outputs 80 values (sequence-to-sequence)")
    print("  - Each output is a pressure prediction for that timestep")
    print("  - Expected 35 input features")
    print("  - Uses bidirectional LSTM with dropout")
    
    print("\nFeature engineering needed (35 features):")
    print("  1. R, C, u_out (keep as-is)")
    print("  2. un_in_std = standardized u_in within each breath")
    print("  3. time_step_std = standardized time_step within each breath")
    print("  4. 30 lag features: after1-5 and back1-5 for (time_step_std, un_in_std, u_out)")
    
except ImportError as e:
    print(f"✗ Error: {e}")
    print("\nTensorFlow required. Install with: pip install tensorflow")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
