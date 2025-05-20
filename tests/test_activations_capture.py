#!/usr/bin/env python
"""
Test script to demonstrate ParamLake activations capture and analysis.
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
OUTPUT_PATH = "test_output/activations_test.zarr"

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

# Define a simple model with named layers for activation capture
def create_model():
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(16, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(8, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(1, name='dense_3')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Use ParamLake to capture activations during training
@paramlake(
    output_path=OUTPUT_PATH,
    run_id="activations_test",
    capture_frequency=1,
    capture_weights=False,
    capture_gradients=False,
    capture_optimizer_state=False,
    capture_activations=True,  # Enable activations capture
    capture_layers=['dense_1', 'dense_2', 'dense_3']  # Specify which layers to capture
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
    
    # Train for a few epochs
    history = model.fit(
        train_dataset,
        epochs=5,
        verbose=1
    )
    
    return model, history

def main():
    # Train model with ParamLake decorator
    print("\n=== Training model with ParamLake activations capture ===")
    model, history = train_model()
    
    # Analyze the stored activations
    print("\n=== Analyzing stored activations ===")
    analyzer = ZarrModelAnalyzer(OUTPUT_PATH, run_id="activations_test")
    
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
    
    # Get activations for each layer at different steps
    print("\nActivation Analysis:")
    for step in range(min(5, num_steps)):
        print(f"\nStep {step}:")
        
        for layer_name in layer_names:
            layer_info = analyzer.get_layer_info(layer_name)
            # Check if this layer has activations
            if "activations" in layer_info.get("tensor_types", []):
                activation_tensors = layer_info.get("tensors", {}).get("activations", [])
                
                if activation_tensors:
                    print(f"  Layer '{layer_name}' activation tensors: {activation_tensors}")
                    
                    for tensor_name in activation_tensors:
                        try:
                            tensor_data = analyzer.get_tensor_data(layer_name, "activations", tensor_name, step=step)
                            
                            # Calculate statistics on the activation outputs
                            act_mean = np.mean(tensor_data)
                            act_std = np.std(tensor_data)
                            act_min = np.min(tensor_data)
                            act_max = np.max(tensor_data)
                            active_percent = np.mean(tensor_data > 0) * 100  # For ReLU layers
                            
                            print(f"    Tensor '{tensor_name}': shape={tensor_data.shape}")
                            print(f"      Mean: {act_mean:.6f}, Std: {act_std:.6f}")
                            print(f"      Min: {act_min:.6f}, Max: {act_max:.6f}")
                            if 'dense_3' not in layer_name:  # Not applicable for output layer (no ReLU)
                                print(f"      Active neurons: {active_percent:.2f}%")
                        except Exception as e:
                            print(f"    Error getting activation data for {tensor_name}: {e}")
    
    # Analyze activation distribution changes over time for a specific layer
    if num_steps >= 2 and 'dense_1' in layer_names:
        print("\nTracking activation distribution changes for 'dense_1':")
        
        dense1_info = analyzer.get_layer_info('dense_1')
        if "activations" in dense1_info.get("tensor_types", []):
            activation_tensors = dense1_info.get("tensors", {}).get("activations", [])
            
            if activation_tensors:
                tensor_name = activation_tensors[0]  # Typically 'output' for activations
                
                for step in range(min(5, num_steps)):
                    try:
                        tensor_data = analyzer.get_tensor_data('dense_1', "activations", tensor_name, step=step)
                        active_percent = np.mean(tensor_data > 0) * 100
                        print(f"  Step {step}: Active neurons = {active_percent:.2f}%")
                    except Exception as e:
                        print(f"  Error analyzing activation distribution for step {step}: {e}")
    
    print("\nParamLake activations capture test completed successfully!")

if __name__ == "__main__":
    main() 