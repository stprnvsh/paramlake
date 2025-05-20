#!/usr/bin/env python
"""
Simple test script to demonstrate ParamLake optimizer state capture and loading.
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
OUTPUT_PATH = "test_output/optimizer_test.zarr"

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
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Use ParamLake to capture optimizer state during training
@paramlake(
    output_path=OUTPUT_PATH,
    run_id="optimizer_test",
    capture_frequency=1,
    capture_gradients=True,
    capture_optimizer_state=True  # Enable optimizer state capture
)
def train_model():
    # Create model and dataset
    model = create_model()
    train_dataset = create_dataset()
    
    # Create optimizer - Adam has momentum and variance states that should be captured
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Train for a few epochs to accumulate optimizer state
    history = model.fit(
        train_dataset,
        epochs=5,
        verbose=1
    )
    
    return model, history

def main():
    # Train model with ParamLake decorator
    print("\n=== Training model with ParamLake optimizer state capture ===")
    model, history = train_model()
    
    # Analyze the stored data
    print("\n=== Analyzing stored optimizer state ===")
    analyzer = ZarrModelAnalyzer(OUTPUT_PATH, run_id="optimizer_test")
    
    # Get optimizer configuration
    optimizer_config = analyzer.get_optimizer_config()
    print("\nOptimizer Configuration:")
    print(optimizer_config)
    
    # Get optimizer state for each step
    print("\nOptimizer States:")
    for step in range(5):  # We trained for 5 epochs
        optimizer_state = analyzer.get_optimizer_state(step=step)
        if optimizer_state:
            print(f"Step {step}: Found {len(optimizer_state)} optimizer state tensors")
            
            # Print shape and summary statistics of first few state tensors
            for i, state_tensor in enumerate(optimizer_state[:2]):  # Show only first 2 for brevity
                print(f"  State tensor {i}: shape={state_tensor.shape}, "
                      f"mean={state_tensor.mean():.6f}, "
                      f"std={state_tensor.std():.6f}")
            
            if len(optimizer_state) > 2:
                print(f"  ... and {len(optimizer_state) - 2} more state tensors")
        else:
            print(f"Step {step}: No optimizer state found")
    
    print("\nParamLake optimizer state capture test completed successfully!")

if __name__ == "__main__":
    main() 