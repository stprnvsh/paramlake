#!/usr/bin/env python
"""
Test script to demonstrate ParamLake gradients capture and analysis.
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
OUTPUT_PATH = "test_output/gradients_test.zarr"

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
    
    return train_dataset, x, y

# Define a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(10,), name='dense_1'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, name='dense_3')
    ])
    return model

# Custom callback to visualize gradients during training
class GradientDebugCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Training begins - setting up gradient visualization")
        
    def on_epoch_end(self, epoch, logs=None):
        # Use a batch of data to compute gradients for visualization
        sample_data = next(iter(self.model.dataset))
        x_sample, y_sample = sample_data
        
        # Compute gradients using GradientTape
        with tf.GradientTape() as tape:
            y_pred = self.model(x_sample, training=True)
            loss_value = self.model.compiled_loss(y_sample, y_pred)
        
        # Get gradients
        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        
        # Print gradient information
        print(f"\n==== Gradient visualization at epoch {epoch+1} ====")
        for i, (grad, var) in enumerate(zip(gradients, self.model.trainable_variables)):
            if grad is not None:
                print(f"Layer: {var.name}")
                print(f"  Shape: {grad.shape}")
                print(f"  Mean: {tf.reduce_mean(grad):.6f}")
                print(f"  Std: {tf.math.reduce_std(grad):.6f}")
                print(f"  Min: {tf.reduce_min(grad):.6f}")
                print(f"  Max: {tf.reduce_max(grad):.6f}")
                print(f"  L2 Norm: {tf.norm(grad):.6f}")
            else:
                print(f"Layer {var.name}: No gradient")

# Use ParamLake to capture gradients during training
@paramlake(
    output_path=OUTPUT_PATH,
    run_id="gradients_test",
    capture_frequency=1,  # Capture every epoch
    capture_weights=True, 
    capture_gradients=True,  # Enable gradients capture
    capture_optimizer_state=False,
    verbose=2  # Maximum verbosity for debugging
)
def train_model():
    # Create model and dataset
    model = create_model()
    train_dataset, x_data, y_data = create_dataset()
    
    # Store the dataset in the model for the callback
    model.dataset = train_dataset
    
    # Create optimizer with specific class name for debugging
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, name="adam_debug")
    
    # Define loss with name for debugging
    loss = tf.keras.losses.MeanSquaredError(name="mse_debug")
    
    # Compile model 
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae']
    )
    
    # Debug callback to visualize gradients
    debug_callback = GradientDebugCallback()
    
    # Train for a few epochs
    print("\nStarting model training with gradient capture...")
    history = model.fit(
        train_dataset,
        epochs=5,
        verbose=1,
        callbacks=[debug_callback]
    )
    
    return model, history

def main():
    # Train model with ParamLake decorator
    print("\n=== Training model with ParamLake gradients capture ===")
    model, history = train_model()
    
    # Analyze the stored gradients
    print("\n=== Analyzing stored gradients ===")
    analyzer = ZarrModelAnalyzer(OUTPUT_PATH, run_id="gradients_test")
    
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
    
    # Check tensor types for each layer
    print("\nAnalyzing tensor types by layer:")
    for layer_name in layer_names:
        layer_info = analyzer.get_layer_info(layer_name)
        tensor_types = layer_info.get("tensor_types", [])
        print(f"  Layer '{layer_name}' has tensor types: {tensor_types}")
        
        # For each tensor type, list available tensors
        for tensor_type in tensor_types:
            tensors = layer_info.get("tensors", {}).get(tensor_type, [])
            if tensors:
                print(f"    {tensor_type}: {tensors}")
    
    # Try to print the run_group structure
    print("\nAnalysis completed! Please check if gradients were captured in the output.")

if __name__ == "__main__":
    main() 