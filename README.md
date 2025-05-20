<div align="center">
  
# 🚀 ParamLake

<img src="https://img.shields.io/badge/Version-0.1.1-blue.svg" alt="Version"/>
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
<img src="https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9%20|%203.10-blue.svg" alt="Python versions"/>

**Track, Store, and Analyze AI Model Internals with Precision**

*Version control for your model's soul - capture every weight, gradient, and activation during training*

[Installation](#-installation) • [Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Examples](#-examples)
</div>

---

<p align="center">
  <i>ParamLake records the invisible journey of your deep learning model through training, creating a time machine for AI experimentation.</i>
</p>

<div align="center">
  <img width="80%" src="https://img.shields.io/badge/%F0%9F%93%8A%20Visualization%20Diagram%20Placeholder-Add%20your%20diagram%20here-lightgrey" alt="Visualization Diagram Placeholder"/>
</div>

## 🌟 Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Comprehensive Tracking</h3>
      <ul>
        <li>Capture <b>weights</b>, <b>gradients</b>, <b>activations</b> & <b>optimizer states</b></li>
        <li>Minimal code changes - just add a decorator!</li>
        <li>Automatic gradient capture without instrumentation</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Advanced Metrics</h3>
      <ul>
        <li>Automatic tensor statistics (L2 norm, sparsity, etc.)</li>
        <li>Performance impact tracking</li>
        <li>Training flow visualization</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>💾 Optimized Storage</h3>
      <ul>
        <li>Tensor-specialized chunking & compression</li>
        <li>Cloud-native with Icechunk integration</li>
        <li>Git-like versioning for AI models</li>
      </ul>
    </td>
    <td>
      <h3>🔍 Deep Analysis</h3>
      <ul>
        <li>Visualize weight evolution & gradient behavior</li>
        <li>Compare training runs with precision</li>
        <li>Time-travel through model versions</li>
      </ul>
    </td>
  </tr>
</table>

- **Framework Agnostic Core** - Architecture designed for TensorFlow, PyTorch, and JAX
- **Production Ready** - Minimal training overhead with asynchronous writing and memory adaptation
- **Flexible Configuration** - YAML-based settings to customize everything
- **Checkpointing** - Save and restore from any point in training history

## 💻 Installation

Choose the installation that fits your needs:

```bash
# 🔋 Basic installation
pip install paramlake

# ☁️ With Icechunk support (for transactional cloud storage)
pip install paramlake icechunk

# 📈 With visualization support
pip install paramlake matplotlib
```

## 🚀 Quick Start

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
# metrics:
#   enabled: true
#   capture_frequency: 1
#   compute: ["l2", "mean", "var", "max", "min", "sparsity"]
#   advanced_compute: ["spectral_norm"]  # For matrix tensors
# capture_optimizer_state: true # Enable optimizer state capture

# 2. Add the decorator to your training function
@paramlake(config="config.yaml")  # or inline config: @paramlake(capture_frequency=5, capture_optimizer_state=True)
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

# 6. Analyze tensor metrics
kernel_metrics = analyzer.get_tensor_metrics("dense_1", "weights", "kernel", "l2")
print(f"L2 norm evolution: {kernel_metrics}")
analyzer.plot_tensor_metrics("dense_1", "weights", "kernel", ["l2", "mean", "var"])

# 7. Analyze optimizer state (if captured)
optimizer_config = analyzer.get_optimizer_config()
if optimizer_config:
    print(f"Optimizer Config: {optimizer_config}")
optimizer_state_step_0 = analyzer.get_optimizer_state(step=0)
if optimizer_state_step_0:
    print(f"Optimizer state at step 0: {len(optimizer_state_step_0)} tensors")
```

## 📊 Metrics Collection

<div align="center">
  <img width="70%" src="https://img.shields.io/badge/%F0%9F%93%88%20Metrics%20visualization%20placeholder-Add%20your%20metrics%20chart%20here-lightgrey" alt="Metrics Visualization Placeholder"/>
</div>

ParamLake automatically computes and stores key statistics about your model's tensors during training:

```python
# Configure metrics collection in your config
@paramlake(
    metrics={
        "enabled": True,
        "capture_frequency": 1,  # Every step
        "compute": ["l2", "mean", "var", "max", "min", "sparsity"],
        "advanced_compute": ["spectral_norm"]  # For matrix tensors (2D)
    }
)
def train_model():
    # Your training code as usual
    model = create_model()
    model.compile(...)
    model.fit(...)
    return model

# Later, analyze the metrics
analyzer = ZarrModelAnalyzer("model_data.zarr")

# Get metrics for a specific tensor
l2_norms = analyzer.get_tensor_metrics("dense_1", "weights", "kernel", "l2")
print(f"L2 norm evolution: {l2_norms}")

# Plot multiple metrics for a tensor over time
analyzer.plot_tensor_metrics(
    "dense_1", 
    "weights", 
    "kernel", 
    ["l2", "mean", "var", "max", "min"]
)

# Get summary statistics across all metrics
metrics_stats = analyzer.get_metrics()
```

<details>
<summary><b>📋 Available metrics include:</b></summary>

- **l2**: L2 norm (magnitude) of the tensor
- **mean**: Mean value of the tensor
- **var**: Variance of the tensor
- **max**: Maximum value in the tensor
- **min**: Minimum value in the tensor
- **sparsity**: Fraction of zero values in the tensor
- **spectral_norm**: Largest singular value (for 2D matrices only)
</details>

## 🔄 Checkpointing and Model Versioning

<div align="center">
  <img width="70%" src="https://img.shields.io/badge/%F0%9F%94%84%20Version%20control%20diagram%20placeholder-Add%20your%20version%20flow%20diagram%20here-lightgrey" alt="Version Control Diagram Placeholder"/>
</div>

ParamLake provides Git-like functionality for AI models, allowing you to save checkpoints and resume training from any point:

```python
import tensorflow as tf
from paramlake import save_checkpoint, load_checkpoint, list_checkpoints
from paramlake.storage.factory import create_storage_manager
from paramlake.utils.config import ParamLakeConfig

# 1. Create a model and train it for a few epochs
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=5)

# 2. Create a storage manager
config = ParamLakeConfig({
    "output_path": "model_checkpoints.zarr",
    "run_id": "my_training_run"
})
storage = create_storage_manager(config)

# 3. Save a checkpoint after initial training
checkpoint_id = save_checkpoint(
    model=model,
    storage_manager=storage,
    step=5,
    name="initial_training",
    description="Model after 5 epochs"
)
print(f"Saved checkpoint: {checkpoint_id}")

# 4. Train the model further
model.fit(x_train, y_train, epochs=5, initial_epoch=5)

# 5. Save another checkpoint
checkpoint_id_2 = save_checkpoint(
    model=model,
    storage_manager=storage,
    step=10,
    name="continued_training",
    description="Model after 10 epochs"
)

# 6. List all available checkpoints
checkpoints = list_checkpoints(storage)
for i, checkpoint in enumerate(checkpoints):
    print(f"Checkpoint {i+1}:")
    print(f"  ID: {checkpoint.get('id')}")
    print(f"  Name: {checkpoint.get('name')}")
    print(f"  Step: {checkpoint.get('step')}")

# 7. Create a new model and restore from a checkpoint
new_model = tf.keras.Sequential([...])
new_model.compile(optimizer='adam', loss='mse')  # Same architecture required

# 8. Load weights and optimizer state from checkpoint
metadata = load_checkpoint(
    model=new_model,
    storage_manager=storage,
    checkpoint_id=checkpoint_id,  # Can also use step=5 to load by step number
    include_optimizer=True,
    recompile=True
)

# 9. Continue training from where you left off
new_model.fit(x_train, y_train, epochs=5, initial_epoch=5)
```

With IceChunk storage, checkpoints are integrated with snapshot functionality for seamless version control of model training.

## 📈 Automatic Gradient Capture

<div align="center">
  <img width="70%" src="https://img.shields.io/badge/%F0%9F%93%89%20Gradient%20capture%20visualization-Add%20gradient%20flow%20diagram%20here-lightgrey" alt="Gradient Capture Visualization Placeholder"/>
</div>

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

<details>
<summary><b>🛠️ Available gradient tracking methods:</b></summary>

- **"auto"**: Automatically detect and use the best method for the model
- **"train_step"**: Override the model's train_step method
- **"optimizer"**: Override the optimizer's apply_gradients method
- **"callback"**: Use a callback-based approach with GradientTape
</details>

## ⚙️ Configuration Options

<div align="center">
  <code>📝 YAML Configuration</code>
</div>

ParamLake can be configured through a YAML file or by passing parameters directly to the decorator:

<details open>
<summary><b>Sample Configuration</b></summary>

```yaml
# Basic options
output_path: "model_data.zarr"  # Where to store the dataset
capture_frequency: 5  # Capture every 5 steps/epochs
capture_gradients: true  # Whether to capture gradients
capture_activations: false  # Whether to capture activations
capture_optimizer_state: true # Whether to capture optimizer state and configuration

# Gradient options
gradients:
  enabled: true
  auto_tracking: true
  track_method: "auto"  # "auto", "train_step", "optimizer", or "callback"

# Metrics options
metrics:
  enabled: true
  capture_frequency: 1
  compute: ["l2", "mean", "var", "max", "min", "sparsity"]
  advanced_compute: ["spectral_norm"]  # For matrix tensors

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
</details>

## ☁️ Cloud Storage with Icechunk

<div align="center">
  <img width="70%" src="https://img.shields.io/badge/%E2%98%81%EF%B8%8F%20Cloud%20storage%20visualization-Add%20cloud%20architecture%20diagram%20here-lightgrey" alt="Cloud Storage Visualization Placeholder"/>
</div>

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
    },
    metrics={
        "enabled": True,
        "compute": ["l2", "mean", "var", "max", "min", "sparsity"]
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

# Analyze metrics from tensors
analyzer.plot_tensor_metrics("dense", "weights", "kernel", ["l2", "mean"])

# Analyze optimizer state with Icechunk
optimizer_config = analyzer.get_optimizer_config()
if optimizer_config:
    print(f"Optimizer Config for snapshot {analyzer.snapshot_id}: {optimizer_config}")
```

## 🔍 Analyzing the Data

<div align="center">
  <img width="70%" src="https://img.shields.io/badge/%F0%9F%94%8D%20Analysis%20visualization-Add%20analysis%20dashboard%20image%20here-lightgrey" alt="Analysis Visualization Placeholder"/>
</div>

ParamLake provides powerful utilities for analyzing the collected data:

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

# Analyze tensor metrics
l2_norm = analyzer.get_tensor_metrics("dense_1", "weights", "kernel", "l2")
print(f"L2 norm evolution: {l2_norm}")

# Plot multiple metrics
analyzer.plot_tensor_metrics("dense_1", "weights", "kernel", ["l2", "mean", "var"])

# Compare two training runs
analyzer.compare_runs("run1.zarr", "run2.zarr")

# Retrieve optimizer configuration and state
optimizer_config = analyzer.get_optimizer_config()
if optimizer_config:
    print(f"Optimizer Configuration: {optimizer_config}")

# Get optimizer state for a specific step (e.g., step 10)
optimizer_state_step_10 = analyzer.get_optimizer_state(step=10)
if optimizer_state_step_10:
    print(f"Optimizer has {len(optimizer_state_step_10)} state tensors at step 10.")
    # You can then inspect individual tensors: optimizer_state_step_10[0]
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

# Analyze tensor metrics across snapshots
for snapshot in history[:3]:  # Look at the latest 3 snapshots
    temp_analyzer = IcechunkModelAnalyzer({
        "type": "s3", 
        "bucket": "my-bucket", 
        "prefix": "my-training-run"
    }, snapshot_id=snapshot["id"])
    
    # Get metric statistics
    l2_norms = temp_analyzer.get_tensor_metrics("dense", "weights", "kernel", "l2")
    print(f"Snapshot {snapshot['id']}: L2 norm evolution: {l2_norms}")
    
    # Get gradient statistics
    grad_stats = temp_analyzer.analyze_gradient_statistics()
    print(f"Snapshot {snapshot['id']}: Gradient coverage {grad_stats['summary']['gradient_coverage']:.2%}")
    
    # Get optimizer config for this snapshot
    opt_config_snap = temp_analyzer.get_optimizer_config()
    if opt_config_snap:
        print(f"Snapshot {snapshot['id']}: Optimizer Config: {opt_config_snap}")
```

## 🔌 Extensibility

ParamLake is designed to be framework-agnostic. While the current implementation focuses on TensorFlow, the schema and storage mechanism are designed to support other frameworks like PyTorch and JAX.

## 📜 License

[MIT License](LICENSE)

---

<div align="center">
  <p>
    <a href="https://github.com/yourusername/paramlake/issues">Report Bug</a> •
    <a href="https://github.com/yourusername/paramlake/issues">Request Feature</a> •
    <a href="https://github.com/yourusername/paramlake/stargazers">⭐ Star Us</a>
  </p>
  <p>
    <i>Made with ❤️ for the ML community</i>
  </p>
</div>