#!/usr/bin/env python
"""
Test script to demonstrate ParamLake model weights capture and analysis.
"""

import os
import numpy as np
import tensorflow as tf
from paramlake import paramlake, ZarrModelAnalyzer

# Make sure we get reproducible results
tf.random.set_seed(42)
np.random.seed(42)

# Create a temporary directory for the example output
os.makedirs("test_output", exist_ok=True)
OUTPUT_PATH = "test_output/weights_test.zarr"

# Clean up any existing file to start fresh
if os.path.exists(OUTPUT_PATH):
    print(f"Removing existing output file: {OUTPUT_PATH}")
    import shutil
    shutil.rmtree(OUTPUT_PATH)

# Create a simple dataset
def create_dataset():
    # Generate synthetic data
    x = np.random.normal(0, 1, size=(1000, 10)).astype(np.float32)
    # Simple linear relationship with some noise
    w = np.random.normal(0, 1, size=(10, 1)).astype(np.float32)
    y = np.dot(x, w) + np.random.normal(0, 0.1, size=(1000, 1)).astype(np.float32)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(32)
    
    return train_dataset

# Define a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,), name='dense_1'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, name='dense_3')
    ])
    return model

# Use ParamLake to capture weights during training
@paramlake(
    output_path=OUTPUT_PATH,
    run_id="weights_test",
    capture_frequency=1,
    capture_weights=True,  # Enable weights capture
    capture_gradients=False,
    capture_optimizer_state=False
)
def train_model():
    # Create model and dataset
    model = create_model()
    train_dataset = create_dataset()
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Train for a few epochs to track weight changes
    history = model.fit(
        train_dataset,
        epochs=5,
        verbose=1
    )
    
    return model, history

def main():
    # Train model with ParamLake decorator
    print("\n=== Training model with ParamLake weights capture ===")
    model, history = train_model()
    
    # Analyze the stored weights
    print("\n=== Analyzing stored weights ===")
    analyzer = ZarrModelAnalyzer(OUTPUT_PATH, run_id="weights_test")
    
    # Get model summary which includes the number of steps
    model_summary = analyzer.get_model_summary()
    num_steps = model_summary.get('steps', 0)
    print(f"\nNumber of steps: {num_steps}")
    
    # Get layer names
    layer_names = analyzer.get_layer_names()
    print(f"Layer names: {layer_names}")
    
    # Get model architecture
    model_config = analyzer.get_run_metadata()
    print("\nModel Configuration:")
    print(model_config)
    
    # Get weights for each step using get_tensor_data
    print("\nWeight Analysis:")
    for step in range(num_steps):
        print(f"\nStep {step}:")
        
        for layer_name in layer_names:
            layer_info = analyzer.get_layer_info(layer_name)
            # Check if this layer has weights
            if "weights" in layer_info.get("tensor_types", []):
                weight_tensors = layer_info.get("tensors", {}).get("weights", [])
                
                if weight_tensors:
                    print(f"  Layer '{layer_name}' weight tensors: {weight_tensors}")
                    
                    for tensor_name in weight_tensors:
                        tensor_data = analyzer.get_tensor_data(layer_name, "weights", tensor_name, step=step)
                        print(f"    Tensor '{tensor_name}': shape={tensor_data.shape}, "
                              f"mean={tensor_data.mean():.6f}, "
                              f"std={tensor_data.std():.6f}")
    
    # Analyze weight changes between steps
    if num_steps >= 2:
        print("\nWeight changes between steps:")
        for layer_name in layer_names:
            layer_info = analyzer.get_layer_info(layer_name)
            # Check if this layer has weights
            if "weights" in layer_info.get("tensor_types", []):
                weight_tensors = layer_info.get("tensors", {}).get("weights", [])
                
                for tensor_name in weight_tensors:
                    initial_weights = analyzer.get_tensor_data(layer_name, "weights", tensor_name, step=0)
                    final_weights = analyzer.get_tensor_data(layer_name, "weights", tensor_name, step=num_steps-1)
                    
                    change = np.abs(final_weights - initial_weights).mean()
                    print(f"  {layer_name}/{tensor_name}: average absolute change = {change:.6f}")
    
    print("\nParamLake weights capture test completed successfully!")

if __name__ == "__main__":
    main() 