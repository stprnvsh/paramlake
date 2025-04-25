"""
Example of using ParamLake Zarr with TensorFlow.

This example demonstrates how to use the ParamLake Zarr decorator to capture
model parameters, gradients, and activations during training.
"""

import os
import numpy as np
import tensorflow as tf
from paramlake import paramlake, ZarrModelAnalyzer

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0

# Add channel dimension for Conv2D layers
x_train = np.expand_dims(x_train, -1)
x_val = np.expand_dims(x_val, -1)

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)

# Use a smaller subset for faster execution in this example
train_samples = 5000
val_samples = 1000
x_train = x_train[:train_samples]
y_train = y_train[:train_samples]
x_val = x_val[:val_samples]
y_val = y_val[:val_samples]

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")

# Create a CNN model for MNIST classification
def create_model():
    """Create a convolutional neural network for MNIST classification."""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name='input'),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv1'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        tf.keras.layers.BatchNormalization(name='bn1'),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        tf.keras.layers.BatchNormalization(name='bn2'),
        
        # Flatten layer
        tf.keras.layers.Flatten(name='flatten'),
        
        # Dense layers
        tf.keras.layers.Dense(128, activation='relu', name='dense1'),
        tf.keras.layers.Dropout(0.4, name='dropout1'),
        
        # Output layer
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Print model summary
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example 1: Applying ParamLake to a model directly
print("\nExample 1: Apply ParamLake directly to a model")
model = create_model()
model = paramlake(
    model,
    output_path="./data/mnist_direct.zarr",
    run_id="mnist_direct",
    capture_frequency=1,  # Capture every epoch
    capture_weights=True,
    capture_gradients=True,
    capture_activations=False
)

# Train the model (parameters will be captured automatically)
model.fit(
    x_train, y_train,
    epochs=5,  # Using fewer epochs for demonstration
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

# Example 2: Applying ParamLake as a function decorator
print("\nExample 2: Apply ParamLake as a function decorator")

@paramlake(
    output_path="./data/mnist_decorator.zarr",
    run_id="mnist_decorator",
    capture_frequency=1,  # Capture every epoch
    capture_weights=True,
    capture_gradients=True,
    capture_activations=False
)
def train_model():
    # Create and train the model
    model = create_model()
    history = model.fit(
        x_train, y_train,
        epochs=5,  # Using fewer epochs for demonstration
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1
    )
    return model, history

# Run the decorated function
model, history = train_model()

# Example 3: Using a YAML configuration file
print("\nExample 3: Using a YAML configuration file")

# Create a config file
config_yaml = """
output_path: "./data/mnist_yaml.zarr"
run_id: "mnist_yaml"
capture_frequency: 1
capture_weights: true
capture_gradients: true
capture_activations: true
chunking:
  time_dimension: 1
  target_chunk_size: 524288  # 512KB
"""

# Write the config to a file
os.makedirs("./data", exist_ok=True)
with open("./data/paramlake_config.yaml", "w") as f:
    f.write(config_yaml)

# Apply ParamLake with the config file
model = create_model()
model = paramlake(model, config="./data/paramlake_config.yaml")

# Train the model (parameters will be captured automatically)
model.fit(
    x_train, y_train,
    epochs=5,  # Using fewer epochs for demonstration
    batch_size=64,
    validation_data=(x_val, y_val),
    verbose=1
)

# Example 4: Analyzing the captured data
print("\nExample 4: Analyzing the captured data")

# Create an analyzer for the captured data
analyzer = ZarrModelAnalyzer("./data/mnist_direct.zarr")

# Print summary of the captured data
summary = analyzer.get_model_summary()
print("\nModel Summary:")
print(f"- Run ID: {summary['run_id']}")
print(f"- Framework: {summary['framework']} {summary['framework_version']}")
print(f"- ParamLake Version: {summary['paramlake_version']}")
print(f"- Steps: {summary['steps']}")
print(f"- Layers: {len(summary['layers'])}")

# Get information about a specific layer
layer_info = analyzer.get_layer_info("conv1")
print("\nLayer Information for 'conv1':")
print(f"- Type: {layer_info.get('type', 'Unknown')}")
print(f"- Tensor Types: {layer_info.get('tensor_types', [])}")
print(f"- Tensors: {layer_info.get('tensors', {})}")

# Get statistics for a specific layer's weights
stats = analyzer.get_layer_stats("conv1", "weights", "kernel")
print("\nWeight Statistics for 'conv1' kernel:")
print(f"- Mean: {stats['mean']}")
print(f"- L2 Norm: {stats['l2_norm']}")

# Print the shape of the captured tensors
tensor_data = analyzer.get_tensor_data("conv1", "weights", "kernel", step=0)
print(f"\nShape of conv1 kernel at step 0: {tensor_data.shape}")

# Example 5: Plotting the captured data (uncomment to see plots)
analyzer.plot_weight_evolution("conv1", "kernel")
analyzer.plot_gradient_norm_by_layer()

print("\nExamples complete. The captured data is stored in the ./data directory.") 