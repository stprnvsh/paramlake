"""
Example using ParamLake with local Zarr storage.
"""

import os
import numpy as np
import tensorflow as tf
from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.factory import create_storage_manager
from paramlake.collectors.weight_collector import WeightCollector
from paramlake.collectors.activation_collector import ActivationCollector
from paramlake.collectors.gradient_collector import GradientCollector
from paramlake.storage.zarr_analyzer import ZarrModelAnalyzer

# Example model creation function
def create_model():
    """Create a simple CNN model for MNIST."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Load MNIST dataset
def load_mnist():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return (x_train, y_train), (x_test, y_test)

# Create a callback class for ParamLake weight collection
class ParamLakeCallback(tf.keras.callbacks.Callback):
    def __init__(self, config_dict, capture_frequency=2, include_layers=None, exclude_layers=None):
        super().__init__()
        self.config = ParamLakeConfig(config_dict)
        self.storage_manager = create_storage_manager(self.config)
        self.capture_frequency = capture_frequency
        
        # Create collectors
        self.weight_collector = WeightCollector(
            self.storage_manager,
            include_layers=include_layers,
            exclude_layers=exclude_layers
        )
        self.activation_collector = ActivationCollector(
            self.storage_manager,
            include_layers=include_layers,
            exclude_layers=exclude_layers
        )
        self.gradient_collector = GradientCollector(
            self.storage_manager,
            include_layers=include_layers,
            exclude_layers=exclude_layers
        )
        
        self.sample_data = None
        
    def on_train_begin(self, logs=None):
        # Initialize step counter
        self.storage_manager.set_step(0)
        
    def on_epoch_end(self, epoch, logs=None):
        # Only capture if epoch matches frequency
        if epoch % self.capture_frequency == 0:
            # Capture weights
            self.weight_collector.capture_model_weights(self.model, step=epoch)
            
            # If we have sample data, capture activations
            if self.sample_data is not None:
                self.activation_collector.capture_activations(self.model, self.sample_data, step=epoch)
            
            # Store metrics
            if logs:
                for name, value in logs.items():
                    self.storage_manager.store_metric(name, value, step=epoch)
                    
            # Increment step for next
            self.storage_manager.increment_step()
            
    def set_sample_data(self, sample_data):
        """Set sample data for activation collection."""
        self.sample_data = sample_data
        
    def close(self):
        """Close storage manager."""
        self.storage_manager.close()

# Main training function
def train_model_with_paramlake():
    """Train a model with ParamLake tracking."""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create ParamLake callback with config
    paramlake_callback = ParamLakeCallback(
        # Storage backend configuration
        config_dict={
            "storage_type": "zarr",
            "output_path": "mnist_training.zarr",
            "run_id": "test_run",
            # Chunking and compression settings
            "chunking": {
                "time_dimension": 10,
                "spatial_dimensions": "auto",
                "target_chunk_size": 1000000
            },
            "compression": {
                "algorithm": "zstd",
                "level": 3,
                "shuffle": 1
            },
        },
        capture_frequency=1,         # Capture parameters every epoch
        include_layers=["conv*", "dense*"],  # Only track conv and dense layers
    )
    
    # Set sample data for activation collection (using a small batch)
    paramlake_callback.set_sample_data(x_train[:5])
    
    # Create model
    model = create_model()
    
    try:
        # Train model
        model.fit(
            x_train[:1000],  # Using a subset for demonstration
            y_train[:1000],
            epochs=15,
            batch_size=32,
            validation_data=(x_test[:100], y_test[:100]),
            verbose=1,
            callbacks=[paramlake_callback]
        )
        
        return model
    finally:
        # Always close the callback to ensure data is properly saved
        paramlake_callback.close()

# Example of analyzing the data after training
def analyze_training():
    """Analyze the training data using ZarrModelAnalyzer."""
    # Create analyzer for the Zarr storage
    analyzer = ZarrModelAnalyzer("mnist_training.zarr")
    
    try:
        # Print model metadata
        metadata = analyzer.get_run_metadata()
        print(f"Model training metadata:")
        print(f"  Framework: {metadata.get('framework')}")
        print(f"  Version: {metadata.get('framework_version')}")
        print(f"  Timestamp: {metadata.get('timestamp')}")
        
        # Get layer names
        layer_names = analyzer.get_layer_names()
        print(f"\nTracked layers: {', '.join(layer_names)}")
        
        # Try to access the accuracy metrics
        try:
            metrics = analyzer.get_metrics()
            print("\nAvailable metrics:")
            for metric_name, values in metrics.items():
                print(f"  {metric_name}: {values[:5]}...")
        except Exception as e:
            print(f"Error getting metrics: {e}")
            
        # Get model summary
        summary = analyzer.get_model_summary()
        print(f"\nModel summary:")
        print(f"  Run ID: {summary.get('run_id')}")
        print(f"  Steps: {summary.get('steps')}")
        print(f"  Number of layers: {len(summary.get('layers', {}))}")
        
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        # Always close the analyzer
        analyzer.close()

if __name__ == "__main__":
    # Train the model
    print("Training model with ParamLake tracking...")
    model = train_model_with_paramlake()
    
    # Evaluate the model
    (_, _), (x_test, y_test) = load_mnist()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Analyze the training data
    print("\nAnalyzing training data...")
    analyze_training() 