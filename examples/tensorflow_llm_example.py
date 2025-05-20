"""
Example of using ParamLake with a small LLM (Transformer model) in TensorFlow.

This example demonstrates how to track and analyze parameters when fine-tuning
a small transformer model on a text classification task.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from paramlake import paramlake, ZarrModelAnalyzer

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
VOCAB_SIZE = 8000  # Vocabulary size
EMBED_DIM = 256    # Embedding dimension
NUM_HEADS = 4      # Number of attention heads
FF_DIM = 512       # Hidden layer size in feed forward network
MAX_SEQ_LENGTH = 128  # Maximum sequence length
BATCH_SIZE = 32
EPOCHS = 5

def create_datasets():
    """Load and prepare the IMDb dataset for sentiment analysis."""
    print("Loading IMDb dataset...")
    
    # Load the IMDB dataset
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    
    # Pad sequences to ensure uniform length
    x_train = keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post"
    )
    x_val = keras.preprocessing.sequence.pad_sequences(
        x_val, maxlen=MAX_SEQ_LENGTH, padding="post", truncating="post"
    )
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    
    # Use a smaller subset for faster execution in this example
    train_subset = train_dataset.take(100)  # Take 100 batches
    val_subset = val_dataset.take(20)      # Take 20 batches
    
    return train_subset, val_subset, x_train, y_train, x_val, y_val

def create_transformer_model():
    """Create a small transformer-based model for text classification."""
    # Input layer
    inputs = layers.Input(shape=(MAX_SEQ_LENGTH,), name="input_token_ids")
    
    # Embedding layer
    embedding_layer = layers.Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBED_DIM, 
        name="token_embedding"
    )(inputs)
    
    # Positional encoding
    positions = tf.range(start=0, limit=MAX_SEQ_LENGTH, delta=1)
    position_embedding = layers.Embedding(
        input_dim=MAX_SEQ_LENGTH,
        output_dim=EMBED_DIM,
        name="position_embedding"
    )(positions)
    x = embedding_layer + position_embedding
    
    # Add multiple transformer blocks
    for i in range(2):  # 2 transformer blocks
        # Self-attention layer
        attn_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, 
            key_dim=EMBED_DIM // NUM_HEADS,
            name=f"attention_{i}"
        )(x, x)
        
        # Add & normalize (first residual connection)
        x = layers.Add(name=f"add_1_{i}")([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"norm_1_{i}")(x)
        
        # Feed-forward network
        ffn_output = layers.Dense(FF_DIM, activation="relu", name=f"dense_1_{i}")(x)
        ffn_output = layers.Dense(EMBED_DIM, name=f"dense_2_{i}")(ffn_output)
        
        # Add & normalize (second residual connection)
        x = layers.Add(name=f"add_2_{i}")([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6, name=f"norm_2_{i}")(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
    
    # Classification head
    x = layers.Dense(128, activation="relu", name="intermediate")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Print model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# Example 1: Tracking LLM parameters with ParamLake
print("\nExample 1: Tracking LLM parameters with ParamLake")

@paramlake(
    output_path="./data/llm_finetuning.zarr",
    run_id="llm_transformer",
    capture_frequency=1,  # Capture every epoch
    capture_weights=True,
    capture_gradients=True,
    capture_activations=True,
    # We can selectively track specific layers in the model
    include_layers=["*embedding*", "*attention*", "dense*"],
    # Chunking strategy for efficient storage
    chunking={
        "time_dimension": 1,  # Store each epoch separately
        # Auto-determine spatial dimensions for transformer tensors
        "spatial_dimensions": "auto",
        "target_chunk_size": 1048576  # 1MB target chunk size
    },
    
)
def train_transformer_model():
    """Train a transformer model with ParamLake tracking."""
    # Create datasets
    train_ds, val_ds, _, _, _, _ = create_datasets()
    
    # Create the transformer model
    model = create_transformer_model()
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1
    )
    
    return model, history

# Train the model with parameter tracking
print("Training transformer model with parameter tracking...")
model, history = train_transformer_model()

# Example 2: Analyzing the captured LLM parameters
print("\nExample 2: Analyzing the captured LLM parameters")

# Create an analyzer for the captured data
analyzer = ZarrModelAnalyzer("./data/llm_finetuning.zarr")

# Print summary of the captured data
summary = analyzer.get_model_summary()
print("\nModel Summary:")
print(f"- Run ID: {summary['run_id']}")
print(f"- Framework: {summary['framework']} {summary['framework_version']}")
print(f"- ParamLake Version: {summary['paramlake_version']}")
print(f"- Steps: {summary['steps']}")
print(f"- Layers: {len(summary['layers'])}")

# Show structure of attention layers
attn_layers = [layer for layer in summary['layers'] if 'attention' in layer]
print("\nAttention Layer Parameters:")
for layer in attn_layers:
    layer_info = analyzer.get_layer_info(layer)
    print(f"\n- Layer: {layer}")
    print(f"  Type: {layer_info.get('type', 'Unknown')}")
    
    # Show different tensor types for this layer
    tensor_types = layer_info.get('tensor_types', [])
    print(f"  Tensor Types: {tensor_types}")
    
    # Show individual tensors for each type
    for tensor_type in tensor_types:
        tensors = layer_info.get('tensors', {}).get(tensor_type, [])
        print(f"  {tensor_type}: {tensors}")
        
        # If there are weights, show their shapes
        if tensor_type == 'weights' and tensors:
            for tensor_name in tensors:
                try:
                    tensor = analyzer.get_tensor_data(layer, tensor_type, tensor_name, step=0)
                    print(f"    {tensor_name} shape: {tensor.shape}")
                except Exception as e:
                    print(f"    Error getting tensor data for {tensor_name}: {e}")

# Calculate parameter statistics for embedding layer
print("\nEmbedding Layer Statistics:")
try:
    # Assuming there is a token_embedding layer
    embedding_stats = analyzer.get_layer_stats("token_embedding", "weights", "embeddings")
    print(f"- Mean: {embedding_stats['mean']}")
    print(f"- Std Dev: {embedding_stats['std']}")
    print(f"- L2 Norm: {embedding_stats['l2_norm']}")
    
    # Show how embedding norms change across training
    print("\nEmbedding L2 Norm Evolution:")
    for step in range(EPOCHS):
        try:
            stats = analyzer.get_layer_stats("token_embedding", "weights", "embeddings", step=step)
            print(f"  Epoch {step}: {stats['l2_norm']:.6f}")
        except Exception:
            # Skip if this step wasn't captured
            pass
except Exception as e:
    print(f"Error analyzing embedding layer: {e}")

# Example 3: Gradient analysis for transformer model
print("\nExample 3: Gradient Analysis")

# Check if gradients were captured
try:
    attn_grad_norms = []
    ff_grad_norms = []
    
    # Collect gradient norms for attention and feed-forward layers
    for layer in summary['layers']:
        if 'attention' in layer:
            # Query, key, value weights in attention
            for tensor_name in ['query', 'key', 'value']:
                try:
                    grad_data = analyzer.get_tensor_data(layer, 'gradients', f'{tensor_name}/kernel', step=0)
                    grad_norm = np.linalg.norm(grad_data)
                    attn_grad_norms.append((layer, tensor_name, grad_norm))
                except Exception:
                    pass
        elif 'dense' in layer:
            try:
                grad_data = analyzer.get_tensor_data(layer, 'gradients', 'kernel', step=0)
                grad_norm = np.linalg.norm(grad_data)
                ff_grad_norms.append((layer, 'kernel', grad_norm))
            except Exception:
                pass
    
    # Print results if we found gradients
    if attn_grad_norms:
        print("\nAttention Gradient Norms (Epoch 0):")
        for layer, tensor, norm in attn_grad_norms:
            print(f"  {layer}/{tensor}: {norm:.6f}")
            
    if ff_grad_norms:
        print("\nFeed-Forward Gradient Norms (Epoch 0):")
        for layer, tensor, norm in ff_grad_norms:
            print(f"  {layer}/{tensor}: {norm:.6f}")
except Exception as e:
    print(f"Error analyzing gradients: {e}")

# Example 4: Activation distributions
print("\nExample 4: Activation Distributions")
try:
    # Check if activations were captured for attention layers
    activation_stats = []
    
    for layer in attn_layers:
        try:
            # Get activation outputs
            act_data = analyzer.get_tensor_data(layer, 'activations', 'output', step=0)
            if act_data is not None:
                act_mean = np.mean(act_data)
                act_std = np.std(act_data)
                activation_stats.append((layer, act_mean, act_std))
        except Exception:
            pass
    
    # Print results if we found activations
    if activation_stats:
        print("\nAttention Layer Activation Statistics (Epoch 0):")
        for layer, mean, std in activation_stats:
            print(f"  {layer}: mean={mean:.6f}, std={std:.6f}")
    else:
        print("\nNo attention layer activations found. Make sure capture_activations=True and sample inputs are provided.")
except Exception as e:
    print(f"Error analyzing activations: {e}")

print("\nLLM parameter tracking example complete!")
print("The captured LLM parameters are stored in ./data/llm_finetuning.zarr") 