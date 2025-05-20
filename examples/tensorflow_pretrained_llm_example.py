"""
Example of using ParamLake with a Pre-trained LLM in TensorFlow.

This example demonstrates how to track parameters when fine-tuning a
pre-trained language model (like BERT or GPT-2) on a specific task.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras
from tensorflow.keras import layers
from paramlake import paramlake, IcechunkModelAnalyzer

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Parameters
BATCH_SIZE = 32
EPOCHS = 3
MAX_SEQ_LENGTH = 128

# Use a small BERT model from TensorFlow Hub
BERT_MODEL_NAME = "small_bert/bert_en_uncased_L-4_H-512_A-8"  
TFHUB_HANDLE_ENCODER = f"https://tfhub.dev/tensorflow/{BERT_MODEL_NAME}/1"
TFHUB_HANDLE_PREPROCESS = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

def create_bert_dataset():
    """Create a dataset for BERT fine-tuning using GLUE SST-2 sentiment analysis."""
    print("Loading GLUE SST-2 dataset...")
    
    # Import GLUE benchmark for sentiment analysis
    import tensorflow_datasets as tfds
    # Load the SST-2 dataset (Stanford Sentiment Treebank)
    train_data, validation_data, test_data = tfds.load(
        name="glue/sst2",
        split=["train[:20%]", "validation", "test"],
        as_supervised=True
    )
    
    # Function to format the data properly for BERT
    def format_example(text, label):
        return {"text": text, "label": label}
    
    # Format the datasets
    train_dataset = train_data.map(format_example)
    validation_dataset = validation_data.map(format_example)
    
    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    
    # Use a smaller subset for faster execution in this example
    train_subset = train_dataset.take(50)  # Take 50 batches
    val_subset = validation_dataset.take(10)  # Take 10 batches
    
    return train_subset, val_subset, train_data, validation_data

def create_bert_model():
    """Create a BERT-based classification model from TensorFlow Hub."""
    print("Creating BERT model...")
    
    # Create text preprocessing layer
    text_input = keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(
        TFHUB_HANDLE_PREPROCESS, name="preprocessing"
    )
    encoder_inputs = preprocessing_layer(text_input)
    
    # Create BERT encoder layer
    encoder = hub.KerasLayer(
        TFHUB_HANDLE_ENCODER, trainable=True, name="BERT_encoder"
    )
    outputs = encoder(encoder_inputs)
    
    # Extract pooled output
    pooled_output = outputs["pooled_output"]  # [batch_size, 512]
    
    # Add classification head
    x = keras.layers.Dropout(0.1, name="dropout")(pooled_output)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)
    
    # Create model
    model = keras.Model(inputs=text_input, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    # Print model summary
    model.summary()
    
    return model

def analyze_bert_parameters(storage_path):
    """Analyze the parameters of a fine-tuned BERT model."""
    print("\nAnalyzing BERT parameters...")
    analyzer = ZarrModelAnalyzer(storage_path)
    
    try:
        # Print model summary
        summary = analyzer.get_model_summary()
        print(f"\nModel Summary:")
        print(f"- Run ID: {summary['run_id']}")
        print(f"- Framework: {summary['framework']} {summary['framework_version']}")
        print(f"- ParamLake Version: {summary['paramlake_version']}")
        print(f"- Steps: {summary['steps']}")
        print(f"- Layers: {len(summary['layers'])}")
        
        # Focus on attention and embedding layers
        attention_layers = [layer for layer in summary['layers'] if 'attention' in layer.lower()]
        
        # Find attention parameter key patterns
        print("\nAttention Layers:")
        for layer in attention_layers[:5]:  # Show first 5 attention layers
            print(f"- {layer}")
            
            # Try to get attention weights
            layer_info = analyzer.get_layer_info(layer)
            tensor_types = layer_info.get('tensor_types', [])
            
            print(f"  Tensor types: {tensor_types}")
            for tensor_type in tensor_types:
                tensors = layer_info.get('tensors', {}).get(tensor_type, [])
                print(f"  {tensor_type} tensors: {len(tensors)}")
                
                # Show first few tensor names
                if tensor_type == 'weights' and tensors:
                    for tensor_name in tensors[:3]:  # Show first 3 tensors
                        try:
                            tensor = analyzer.get_tensor_data(layer, tensor_type, tensor_name, step=0)
                            print(f"    {tensor_name} shape: {tensor.shape}")
                        except Exception as e:
                            print(f"    Error getting tensor data for {tensor_name}: {e}")
        
        # Calculate attention entropy (randomness) for the first attention layer
        print("\nAttention Analysis:")
        for i, layer in enumerate(attention_layers[:2]):  # Check first two attention layers
            try:
                # BERT stores attention matrices differently based on its implementation
                # Try common patterns
                attention_patterns = [
                    ('weights', 'query/kernel'),
                    ('weights', 'attention/key/kernel'),
                    ('weights', 'attention/query/kernel')
                ]
                
                for tensor_type, tensor_name in attention_patterns:
                    try:
                        attn_weights = analyzer.get_tensor_data(layer, tensor_type, tensor_name, step=0)
                        if attn_weights is not None:
                            # Calculate L2 norm of attention weights
                            attn_norm = np.linalg.norm(attn_weights)
                            print(f"  Layer {layer}, {tensor_name}: L2 Norm = {attn_norm:.6f}")
                            
                            # Try to get the same weights at the final step to compare
                            try:
                                final_weights = analyzer.get_tensor_data(
                                    layer, tensor_type, tensor_name, step=summary['steps']-1
                                )
                                if final_weights is not None:
                                    # Calculate change in weights during fine-tuning
                                    weight_diff = np.linalg.norm(final_weights - attn_weights)
                                    percent_change = (weight_diff / attn_norm) * 100
                                    print(f"    Weight change during fine-tuning: {weight_diff:.6f} ({percent_change:.2f}%)")
                            except Exception:
                                pass
                        break  # Found a valid pattern
                    except Exception:
                        continue
            except Exception as e:
                print(f"  Error analyzing attention layer {layer}: {e}")
                
        # Analyze gradients for classifier layer
        print("\nClassifier Layer Gradient Analysis:")
        try:
            classifier_layer = "classifier"
            if classifier_layer in summary['layers']:
                # Get gradient statistics across training
                for step in range(min(3, summary['steps'])):
                    try:
                        grad_data = analyzer.get_tensor_data(
                            classifier_layer, 'gradients', 'kernel', step=step
                        )
                        if grad_data is not None:
                            grad_norm = np.linalg.norm(grad_data)
                            grad_mean = np.mean(np.abs(grad_data))
                            grad_std = np.std(grad_data)
                            print(f"  Step {step}:")
                            print(f"    L2 Norm: {grad_norm:.6f}")
                            print(f"    Mean Abs: {grad_mean:.6f}")
                            print(f"    Std Dev: {grad_std:.6f}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"  Error analyzing classifier gradients: {e}")
            
        # Analyze metrics
        print("\nTraining Metrics:")
        try:
            metrics = analyzer.get_metrics()
            if metrics:
                metric_names = list(metrics.keys())
                print(f"  Available metrics: {metric_names}")
                
                # Show accuracy progression if available
                if 'accuracy' in metrics:
                    acc_values = metrics['accuracy']
                    print(f"  Accuracy progression: {acc_values}")
                
                # Show loss progression if available
                if 'loss' in metrics:
                    loss_values = metrics['loss']
                    print(f"  Loss progression: {loss_values}")
        except Exception as e:
            print(f"  Error analyzing metrics: {e}")
            
    except Exception as e:
        print(f"Analysis error: {e}")
    finally:
        # Always close the analyzer
        analyzer.close()

# Example 1: Fine-tuning BERT with ParamLake
print("\nExample 1: Fine-tuning BERT with ParamLake")

# Create data directory if it doesn't exist
os.makedirs("./data", exist_ok=True)

@paramlake(
    output_path="./data/bert_finetuning.zarr",
    run_id="bert_finetuning",
    capture_frequency=1,  # Capture every epoch
    capture_weights=True,
    capture_gradients=True,
    capture_activations=False,  # Activations can be very large for LLMs
    # We focus on specific layers to track
    include_layers=["*attention*", "*dense*", "*classifier*", "*dropout*"],
    exclude_layers=["*intermediate*"],  # Skip some large intermediate layers
    # Chunking strategy for efficient storage
    chunking={
        "time_dimension": 1,  # Store each epoch separately
        "spatial_dimensions": "auto",
        "target_chunk_size": 2097152  # 2MB target chunk size
    },
    # Compression settings
    compression={
        "algorithm": "zstd",
        "level": 3,
        "shuffle": True
    }
)
def finetune_bert():
    """Fine-tune BERT with ParamLake tracking."""
    # Create datasets
    train_ds, val_ds, _, _ = create_bert_dataset()
    
    # Create the BERT model
    model = create_bert_model()
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1
    )
    
    return model, history

# Example 2: Analyzing fine-tuning with Icechunk for cloud storage
print("\nExample 2: Fine-tuning BERT with ParamLake + Icechunk")
# For cloud storage, we can use Icechunk
from paramlake import ZarrModelAnalyzer

# Define a dataset sample to capture activations
def get_sample_input():
    """Get a sample input for activation capture."""
    # Import GLUE benchmark for sentiment analysis
    import tensorflow_datasets as tfds
    # Get a single batch
    sample_data = next(iter(tfds.load(
        name="glue/sst2",
        split="validation[:1]",
        as_supervised=True
    )))
    # Format it for the model
    return {"text": sample_data[0]}

@paramlake(
    # Use Zarr for local storage
    output_path="./data/bert_activation_analysis.zarr",
    run_id="bert_activation_analysis",
    capture_frequency=1,
    capture_weights=True,
    capture_gradients=True,
    capture_activations=True,  # Capture activations for analysis
    activation_sample_inputs=get_sample_input,  # Provide sample input
    # Focus on attention layers for activation analysis
    include_layers=["*attention*"],
    # Chunking and compression for attention outputs
    chunking={
        "time_dimension": 1,
        "spatial_dimensions": "auto",
        "target_chunk_size": 1048576  # 1MB chunks
    },
    compression={
        "algorithm": "zstd",
        "level": 4,
        "shuffle": True
    }
)
def analyze_bert_activations():
    """Analyze BERT activations during fine-tuning."""
    # Create a smaller dataset since we're capturing more data
    train_ds, val_ds, _, _ = create_bert_dataset()
    
    # Take even smaller subsets for activation analysis
    train_ds = train_ds.take(10)
    val_ds = val_ds.take(2)
    
    # Create the BERT model
    model = create_bert_model()
    
    # Train for fewer epochs since activation data is large
    history = model.fit(
        train_ds,
        epochs=1,  # Just one epoch for activation analysis
        validation_data=val_ds,
        verbose=1
    )
    
    return model, history

# Main execution
if __name__ == "__main__":
    # Fine-tune BERT with parameter tracking
    print("Fine-tuning BERT with parameter tracking...")
    model, history = finetune_bert()
    
    # Analyze the fine-tuned parameters
    analyze_bert_parameters("./data/bert_finetuning.zarr")
    
    # Optionally run activation analysis
    # Uncomment this if you want to run the activation analysis
    # print("\nRunning BERT activation analysis...")
    # model, history = analyze_bert_activations()
    
    print("\nBERT parameter tracking example complete!")
    print("The captured parameters are stored in ./data/bert_finetuning.zarr") 