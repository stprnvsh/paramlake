# ParamLake

A comprehensive solution for tracking, storing, and analyzing deep learning model parameters, gradients, and activations during training. ParamLake uses advanced storage technologies for efficient management of model data.

## Features

- **Minimal Code Changes**: Simply add a decorator to your training function
- **Automatic Gradient Capture**: Multiple methods for tracking gradients without manual instrumentation
- **Comprehensive Data Collection**: Capture trainable weights, non-trainable variables, gradients, and activations
- **Optimized Storage**: Specialized chunk sizes and compression strategies for different tensor types
- **Efficient Analysis**: Tools for analyzing and visualizing model parameters and gradients
- **Transactional Storage**: Supports Icechunk for cloud-native transactional tensor storage
- **Framework Agnostic Design**: Core schema designed to work across TensorFlow, PyTorch, and JAX (TensorFlow implementation provided)
- **Flexible Configuration**: YAML-based configuration for customizing what and how data is collected
- **Production Ready**: Optimized for minimal training overhead while providing comprehensive parameter tracking

## Installation

```bash
# Basic installation
pip install paramlake

# With Icechunk support (for transactional cloud storage)
pip install paramlake icechunk

# With visualization support
pip install paramlake matplotlib
```

## Quick Start

```python
import tensorflow as tf
from paramlake import paramlake

# 1. Optional: Configure via YAML
# config.yaml:
# capture_frequency: 1  # every epoch
# compression:
#   algorithm: blosc_zstd
#   level: 3
# output_path: "model_data.zarr"
# gradients:
#   enabled: true
#   auto_tracking: true
#   track_method: "auto"  # Can be "auto", "train_step", "optimizer", or "callback"

# 2. Add the decorator to your training function
@paramlake(config="config.yaml")  # or inline config: @paramlake(capture_frequency=5)
def train_model():
    # Define and train your model as usual
    model = tf.keras.Sequential([...])
    model.compile(...)
    model.fit(...)

# 3. Call your training function - ParamLake will automatically log parameters and gradients
train_model()

# 4. Analyze the data
from paramlake import ZarrModelAnalyzer

analyzer = ZarrModelAnalyzer("model_data.zarr")
analyzer.plot_weight_evolution("dense_1/kernel")

# 5. Analyze gradient behavior
analyzer.plot_gradient_norm_by_layer()
gradient_stats = analyzer.analyze_gradient_statistics()
print(f"Gradient coverage: {gradient_stats['summary']['gradient_coverage']:.2%}")
```

## Automatic Gradient Capture

ParamLake provides multiple methods to automatically capture gradients during training:

```python
# Configure gradient tracking method in the decorator
@paramlake(
    gradients={
        "enabled": True,
        "auto_tracking": True,
        "track_method": "auto"  # Automatically select the best method
    }
)
def train_model():
    model = create_model()
    model.compile(...)
    model.fit(...)
    return model

# Alternatively, use a configuration file
@paramlake(config="paramlake_config.yaml")
def train_model():
    # ParamLake will handle gradient tracking based on config file settings
    model = create_model()
    model.compile(...)
    model.fit(...)
    return model
```

The available gradient tracking methods are:
- **"auto"**: Automatically detect and use the best method for the model
- **"train_step"**: Override the model's train_step method
- **"optimizer"**: Override the optimizer's apply_gradients method
- **"callback"**: Use a callback-based approach with GradientTape

## Configuration Options

ParamLake can be configured through a YAML file or by passing parameters directly to the decorator:

```yaml
# Basic options
output_path: "model_data.zarr"  # Where to store the dataset
capture_frequency: 5  # Capture every 5 steps/epochs
capture_gradients: true  # Whether to capture gradients
capture_activations: false  # Whether to capture activations

# Gradient options
gradients:
  enabled: true
  auto_tracking: true
  track_method: "auto"  # "auto", "train_step", "optimizer", or "callback"

# Layer filtering
include_layers: ["dense*", "conv*"]  # Only include layers matching patterns
exclude_layers: ["batch_normalization*"]  # Exclude specific layers

# Storage optimization
compression:
  algorithm: blosc_zstd  # Compression algorithm: blosc, zstd, lz4, etc.
  level: 3  # Compression level (higher = more compression but slower)
  shuffle: true  # Whether to shuffle data before compression

# Gradient-specific compression
gradient_compression:
  algorithm: blosc_zstd
  level: 5  # Higher compression for gradients
  shuffle: true

# Chunking strategy
chunking:
  time_dimension: 1  # Number of time steps per chunk
  spatial_dimensions: auto  # Automatic sizing based on tensor shape
  target_chunk_size: 1048576  # Target chunk size in bytes (1MB)
  gradient_chunk_size: 524288  # Smaller chunks for gradients (512KB)
```

## Cloud Storage with Icechunk

ParamLake supports [Icechunk](https://icechunk.io), a transactional storage engine for tensor data designed for cloud object storage. This provides:

- **Transactional Consistency**: Prevent data corruption when multiple processes write to the store
- **Version Control**: Track model parameters across different training runs with branches and tags
- **Time Travel**: Go back to previous states of model parameters for comparison
- **Cloud Optimization**: Optimized for S3, GCS, and Azure blob storage

### Using ParamLake with Icechunk

```python
import tensorflow as tf
from paramlake import paramlake

# Configure S3 storage with Icechunk backend
@paramlake(
    storage_backend="icechunk",
    storage_type="s3",
    bucket="paramlake",
    prefix="mnist_training",
    region="us-east-1",
    create_repo=True,
    icechunk={
        "commit_frequency": 5,  # Commit changes every 5 epochs
        "tag_snapshots": True,  # Create tags for snapshots
    },
    capture_frequency=1,
    gradients={
        "enabled": True,
        "auto_tracking": True
    }
)
def train_model():
    # Train your model as usual
    model = tf.keras.Sequential([...])
    model.compile(...)
    model.fit(...)
    return model

# Analyze data with IcechunkModelAnalyzer
from paramlake import IcechunkModelAnalyzer

analyzer = IcechunkModelAnalyzer({
    "type": "s3",
    "bucket": "paramlake",
    "prefix": "mnist_training"
})

# Analyze snapshots, compare runs, gradient behavior, etc.
analyzer.plot_weight_evolution("dense/kernel")
analyzer.plot_gradient_norm_by_layer()
gradient_stats = analyzer.analyze_gradient_statistics()
```

## Analyzing the Data

ParamLake provides utilities for analyzing the collected data:

```python
from paramlake import ZarrModelAnalyzer

analyzer = ZarrModelAnalyzer("model_data.zarr")

# Get layer statistics over time
stats = analyzer.get_layer_stats("dense_1/kernel")

# Plot weight evolution
analyzer.plot_weight_evolution("dense_1/kernel")

# Analyze gradients
gradient_stats = analyzer.analyze_gradient_statistics()
for layer_name, layer_stats in gradient_stats["layer_stats"].items():
    for tensor_name, tensor_stats in layer_stats.items():
        print(f"{layer_name}/{tensor_name}:")
        print(f"  Mean gradient magnitude: {tensor_stats['mean_abs']:.6f}")
        print(f"  Max gradient magnitude: {tensor_stats['max']:.6f}")
        print(f"  Zero fraction: {tensor_stats['zero_fraction']:.2%}")

# Plot gradient norms
analyzer.plot_gradient_norm_by_layer()

# Compare two training runs
analyzer.compare_runs("run1.zarr", "run2.zarr")
```

For Icechunk storage, use the IcechunkModelAnalyzer:

```python
from paramlake import IcechunkModelAnalyzer

# Analyze S3 storage
analyzer = IcechunkModelAnalyzer({
    "type": "s3", 
    "bucket": "my-bucket", 
    "prefix": "my-training-run"
})

# Get training history (snapshots)
history = analyzer.get_training_history()

# Compare snapshots
analyzer.plot_snapshot_comparison(
    other_snapshot_id="H5CCPE350FJV69V9D0HG",
    layer_name="dense/kernel"
)

# Analyze gradients across snapshots
for snapshot in history[:3]:  # Look at the latest 3 snapshots
    temp_analyzer = IcechunkModelAnalyzer({
        "type": "s3", 
        "bucket": "my-bucket", 
        "prefix": "my-training-run"
    }, snapshot_id=snapshot["id"])
    
    # Get gradient statistics
    grad_stats = temp_analyzer.analyze_gradient_statistics()
    print(f"Snapshot {snapshot['id']}: Gradient coverage {grad_stats['summary']['gradient_coverage']:.2%}")
```

## Extensibility

ParamLake is designed to be framework-agnostic. While the current implementation focuses on TensorFlow, the schema and storage mechanism are designed to support other frameworks like PyTorch and JAX.

## License

MIT License