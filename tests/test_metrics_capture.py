#!/usr/bin/env python
"""
Test script to demonstrate ParamLake metrics collection and analysis.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from paramlake import paramlake, ZarrModelAnalyzer

# Make sure we get reproducible results
tf.random.set_seed(42)
np.random.seed(42)

# Create a temporary directory for the example output
os.makedirs("test_output", exist_ok=True)
OUTPUT_PATH = "test_output/metrics_test.zarr"

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

# Use ParamLake to track metrics during training
@paramlake(
    output_path=OUTPUT_PATH,
    run_id="metrics_test",
    capture_frequency=1,
    capture_weights=True,
    capture_gradients=True,
    capture_optimizer_state=True,
    metrics={
        "enabled": True,
        "capture_frequency": 1,
        "compute": ["l2", "mean", "var", "max", "min", "sparsity"],
        "advanced_compute": ["spectral_norm"]  # Experimental feature for matrices
    }
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

def plot_metric_evolution(analyzer, layer_name, tensor_type, tensor_name, metric_names):
    """Helper function to plot the evolution of metrics over time"""
    plt.figure(figsize=(12, 6))
    
    for metric_name in metric_names:
        try:
            # Get metric data with full path
            metric_path = f"{layer_name}/{tensor_type}/{tensor_name}/{metric_name}"
            metric_data = analyzer.get_metrics(metric_path)
            
            # Plot metric evolution
            plt.plot(metric_data, label=f"{metric_name}")
        except Exception as e:
            print(f"Error plotting metric {metric_name}: {e}")
    
    plt.title(f"Metrics Evolution for {layer_name}/{tensor_name}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot to a file
    plt.savefig(f"test_output/{layer_name}_{tensor_name}_metrics.png")
    plt.close()

def main():
    # Train model with ParamLake decorator
    print("\n=== Training model with ParamLake metrics collection ===")
    model, history = train_model()
    
    # Analyze the stored metrics
    print("\n=== Analyzing stored metrics ===")
    analyzer = ZarrModelAnalyzer(OUTPUT_PATH, run_id="metrics_test")
    
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
    
    # Available metrics
    print("\nAvailable metrics:")
    all_metrics = analyzer.get_metrics()
    metric_paths = list(all_metrics.keys())
    
    # Filter metrics by tensor type
    weight_metrics = [m for m in metric_paths if '/weights/' in m]
    gradient_metrics = [m for m in metric_paths if '/gradients/' in m]
    
    print(f"Found {len(weight_metrics)} weight metrics")
    print(f"Found {len(gradient_metrics)} gradient metrics")
    
    # Print a sample of metrics
    if weight_metrics:
        print("\nWeight Metrics (sample):")
        for metric_path in weight_metrics[:5]:  # Show first 5
            print(f"  {metric_path}")
            metric_data = analyzer.get_metrics(metric_path)
            print(f"    Shape: {metric_data.shape}, Final value: {metric_data[-1]:.6f}")
    
    if gradient_metrics:
        print("\nGradient Metrics (sample):")
        for metric_path in gradient_metrics[:5]:  # Show first 5
            print(f"  {metric_path}")
            metric_data = analyzer.get_metrics(metric_path)
            print(f"    Shape: {metric_data.shape}, Final value: {metric_data[-1]:.6f}")
    
    # Plot metrics evolution for a few key tensors
    if weight_metrics:
        # First dense layer kernel weights metrics
        try:
            plot_metric_evolution(
                analyzer, 
                'dense_1', 
                'weights', 
                'kernel', 
                ['l2', 'mean', 'var', 'max', 'min']
            )
            print("\nCreated plot for dense_1/weights/kernel metrics")
        except Exception as e:
            print(f"Error plotting weight metrics: {e}")
    
    if gradient_metrics:
        # First dense layer kernel gradients metrics
        try:
            plot_metric_evolution(
                analyzer, 
                'dense_1', 
                'gradients', 
                'kernel', 
                ['l2', 'mean', 'var', 'max', 'min']
            )
            print("Created plot for dense_1/gradients/kernel metrics")
        except Exception as e:
            print(f"Error plotting gradient metrics: {e}")
    
    # Analyze all metrics
    print("\nMetrics Analysis across layers:")
    metric_stats = {}
    
    for layer_name in layer_names:
        layer_info = analyzer.get_layer_info(layer_name)
        
        # Check if this layer has weights or gradients
        for tensor_type in ["weights", "gradients"]:
            if tensor_type in layer_info.get("tensor_types", []):
                tensors = layer_info.get("tensors", {}).get(tensor_type, [])
                
                for tensor_name in tensors:
                    # Check if metrics exist for this tensor
                    metrics_prefix = f"{layer_name}/{tensor_type}/{tensor_name}/"
                    tensor_metrics = [m for m in metric_paths if m.startswith(metrics_prefix)]
                    
                    if tensor_metrics:
                        print(f"  {layer_name}/{tensor_type}/{tensor_name} has {len(tensor_metrics)} metrics")
                        
                        # Compute basic statistics on each metric
                        for metric_path in tensor_metrics:
                            try:
                                metric_name = metric_path.split('/')[-1]
                                metric_data = analyzer.get_metrics(metric_path)
                                
                                # Compute trend
                                if len(metric_data) >= 2:
                                    trend = (metric_data[-1] - metric_data[0]) / (metric_data[0] + 1e-10)
                                    trend_desc = "increasing" if trend > 0.1 else "decreasing" if trend < -0.1 else "stable"
                                else:
                                    trend = 0
                                    trend_desc = "unknown"
                                
                                print(f"    {metric_name}: mean={np.mean(metric_data):.6f}, trend={trend_desc} ({trend:.2%})")
                                
                                # Store in stats
                                if layer_name not in metric_stats:
                                    metric_stats[layer_name] = {}
                                if tensor_type not in metric_stats[layer_name]:
                                    metric_stats[layer_name][tensor_type] = {}
                                if tensor_name not in metric_stats[layer_name][tensor_type]:
                                    metric_stats[layer_name][tensor_type][tensor_name] = {}
                                
                                metric_stats[layer_name][tensor_type][tensor_name][metric_name] = {
                                    "mean": float(np.mean(metric_data)),
                                    "min": float(np.min(metric_data)),
                                    "max": float(np.max(metric_data)),
                                    "trend": float(trend)
                                }
                            except Exception as e:
                                print(f"    Error analyzing {metric_path}: {e}")
    
    # Save the metrics stats
    try:
        import json
        stats_path = "test_output/metrics_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(metric_stats, f, indent=2)
        print(f"\nSaved metrics statistics to {stats_path}")
    except Exception as e:
        print(f"Error saving metrics stats: {e}")
    
    print("\nParamLake metrics collection test completed successfully!")

if __name__ == "__main__":
    main() 