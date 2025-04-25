"""
Example using ParamLake with Icechunk for S3 storage.
"""

import os
import numpy as np
import tensorflow as tf
from paramlake import paramlake, IcechunkModelAnalyzer

# Set up AWS credentials (for example purposes only - in practice use environment variables)
# IMPORTANT: Replace these with your own credentials or preferably set as environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_ACCESS_KEY"  # Replace with your own or use environment variables
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_SECRET_KEY"  # Replace with your own or use environment variables
os.environ["AWS_S3_ENDPOINT"] = "https://s3.amazonaws.com"

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

# Custom callback to print information about commits
class CommitInfoCallback(tf.keras.callbacks.Callback):
    def __init__(self, commit_frequency=5):
        super().__init__()
        self.commit_frequency = commit_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        # Epochs are zero-indexed in callbacks
        epoch_num = epoch + 1
        print(f"\nEpoch {epoch_num}: Current step = {epoch}, (epoch_num % commit_frequency) = {epoch_num % self.commit_frequency}")
        if epoch_num % self.commit_frequency == 0:
            print(f"Epoch {epoch_num}: This should trigger a commit (commit_frequency={self.commit_frequency})")
        else:
            epochs_until_commit = self.commit_frequency - (epoch_num % self.commit_frequency)
            print(f"Epoch {epoch_num}: {epochs_until_commit} more epochs until next commit")

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

# Main training function with ParamLake decorator
@paramlake(
    # Storage backend configuration
    storage_type="icechunk",
    storage_backend="s3",  # Explicitly set s3 as the backend
    bucket="paramlake",
    prefix="pranavsateesh7",  # Updated prefix to avoid conflicts
    region="us-east-1",
    endpoint_url="https://s3.amazonaws.com",
    create_repo=True,  # Create the repo if it doesn't exist
    run_id="test_run_2",
    
    # Icechunk configuration
    icechunk={
        "commit_frequency": 5,   # Commit changes every 5 epochs
        "tag_snapshots": True,   # Create tags for snapshots
        "verbose": True,         # Print additional information about commits
    },
    
    # ParamLake configuration
    capture_frequency=1,         # Capture parameters every epoch
    capture_gradients=True,
    capture_activations=True,
    include_layers=["conv*", "dense*"],  # Only track conv and dense layers
    
    # Chunking settings - but no compression as it's not compatible with Icechunk
    chunking={
        "time_dimension": 10,
        "spatial_dimensions": "auto",
        "target_chunk_size": 1000000
    },
)
def train_model(paramlake_callback=None):
    """Train a model with ParamLake tracking."""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create model
    model = create_model()
    
    # Train model - use a very small subset for quick testing
    model.fit(
        x_train[:100],  # Using a tiny subset for faster demonstration
        y_train[:100],
        epochs=20,       # Just a few epochs for testing
        batch_size=32,
        validation_data=(x_test[:20], y_test[:20]),
        verbose=1,
        callbacks=[CommitInfoCallback(commit_frequency=5)] + ([paramlake_callback] if paramlake_callback else [])
    )
    
    return model

# Example of analyzing the data after training
def analyze_training():
    """Analyze the training data using IcechunkModelAnalyzer."""
    # Create analyzer for the S3 storage
    analyzer = IcechunkModelAnalyzer({
        "type": "s3",
        "bucket": "paramlake",
        "prefix": "pranavsateesh7",  # Make sure this matches the prefix used above
        "region": "us-east-1",
        "endpoint_url": "https://s3.amazonaws.com"
    })
    
    try:
        # Print model metadata
        metadata = analyzer.get_run_metadata()
        print(f"Model training metadata:")
        print(f"  Framework: {metadata.get('framework')}")
        print(f"  Version: {metadata.get('framework_version')}")
        print(f"  Timestamp: {metadata.get('timestamp')}")
        
        # Get layer names
        layer_names = analyzer.get_layer_names()
        print(f"\nTracked layers ({len(layer_names)}):")
        for layer in layer_names:
            print(f"  - {layer}")
        
        # Try to access the accuracy metrics
        try:
            metrics = analyzer.get_metrics()
            print("\nAvailable metrics:")
            for metric_name, values in metrics.items():
                print(f"  {metric_name}: {values[:5]}...")
        except Exception as e:
            print(f"Error getting metrics: {e}")
            
        # Get training history
        try:
            history = analyzer.get_training_history()
            print(f"\nTraining history ({len(history)} snapshots):")
            for i, snapshot in enumerate(history[:5]):  # Show first 5 snapshots
                print(f"  {i+1}. ID: {snapshot['id'][:8]}..., Message: {snapshot['message']}")
            
            # With commit_frequency=5 and epochs=20, we should expect 4 snapshots
            expected_snapshots = 20 // 5
            print(f"\nExpected number of weight snapshots: {expected_snapshots} (every 5 epochs for 20 epochs)")
            
            # Count tensors by type for each layer
            tensor_counts = {}
            total_tensors = 0
            
            for layer_name in layer_names:
                try:
                    layer_info = analyzer.get_layer_info(layer_name)
                    tensor_types = layer_info.get("tensor_types", [])
                    
                    # Initialize counts for this layer
                    if layer_name not in tensor_counts:
                        tensor_counts[layer_name] = {}
                    
                    # Count tensors by type
                    for tensor_type in tensor_types:
                        try:
                            tensors = layer_info.get("tensors", {}).get(tensor_type, [])
                            tensor_counts[layer_name][tensor_type] = len(tensors)
                            total_tensors += len(tensors)
                            
                            # For the first layer, try to get tensor data to see time steps
                            if layer_name == layer_names[0] and len(tensors) > 0:
                                tensor_name = tensors[0]
                                try:
                                    tensor_data = analyzer.get_tensor_data(layer_name, tensor_type, tensor_name)
                                    if tensor_data is not None and len(tensor_data.shape) > 0:
                                        print(f"\nTime steps in {layer_name}/{tensor_type}/{tensor_name}: {tensor_data.shape[0]}")
                                except Exception as e:
                                    pass
                        except Exception as e:
                            print(f"Error counting tensors for {layer_name}/{tensor_type}: {e}")
                except Exception as e:
                    print(f"Error getting layer info for {layer_name}: {e}")
            
            # Print tensor counts by layer and type
            print(f"\nTensor counts by layer and type:")
            for layer_name, type_counts in tensor_counts.items():
                print(f"  {layer_name}:")
                for tensor_type, count in type_counts.items():
                    print(f"    - {tensor_type}: {count} tensors")
            
            print(f"\nTotal tensors stored: {total_tensors}")
            
        except Exception as e:
            print(f"Error getting training history: {e}")
            
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        # Always close the analyzer
        analyzer.close()

if __name__ == "__main__":
    # Train the model
    print("Training model with Icechunk S3 tracking...")
    model = train_model()
    
    # Evaluate the model
    (_, _), (x_test, y_test) = load_mnist()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Analyze the training data
    print("\nAnalyzing training data...")
    analyze_training() 