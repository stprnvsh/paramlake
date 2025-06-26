# ParamLake Installation Guide

ParamLake is designed to be framework-agnostic, supporting multiple machine learning frameworks through optional dependencies. This allows you to install only the frameworks you need, reducing installation time and dependency conflicts.

## Basic Installation

### Core Package (Framework-Agnostic)

```bash
pip install paramlake
```

This installs only the core ParamLake functionality with minimal dependencies:
- `zarr>=2.12.0` - Array storage backend
- `numcodecs>=0.10.0` - Compression codecs
- `icechunk>=0.1.0` - Version control backend  
- `numpy>=1.19.0` - Numerical computing
- `pyyaml>=6.0` - Configuration files

## Framework-Specific Installations

### TensorFlow Support

```bash
pip install "paramlake[tf]"
```

This adds TensorFlow support:
- `tensorflow>=2.6.0`

### PyTorch Support

```bash
pip install "paramlake[torch]"
```

This adds PyTorch support:
- `torch>=1.8.0`
- `torchvision>=0.9.0`

### JAX Support

```bash
pip install "paramlake[jax]"
```

This adds JAX support:
- `jax[cpu]>=0.3.0`
- `jaxlib>=0.3.0`
- `flax>=0.6.0`

### HuggingFace Transformers Support

```bash
pip install "paramlake[hf]"
```

This adds HuggingFace ecosystem support:
- `transformers>=4.20.0`
- `datasets>=2.0.0`
- `tokenizers>=0.12.0`

## Combined Installations

### Multiple Frameworks

You can install support for multiple frameworks at once:

```bash
# TensorFlow + PyTorch
pip install "paramlake[tf,torch]"

# All ML frameworks
pip install "paramlake[tf,torch,jax]"

# ML frameworks + HuggingFace
pip install "paramlake[tf,torch,hf]"
```

### Common Combinations

```bash
# Data science stack
pip install "paramlake[tf,scientific,visualize]"

# Research environment
pip install "paramlake[tf,torch,jax,hf,visualize]"

# Production deployment
pip install "paramlake[torch,distributed]"
```

## Additional Optional Features

### Scientific Computing

```bash
pip install "paramlake[scientific]"
```

Adds scientific computing libraries:
- `scipy>=1.7.0`
- `scikit-learn>=1.0.0`

### Distributed Computing

```bash
pip install "paramlake[distributed]"
```

Adds distributed computing support:
- `dask[array]>=2022.0.0`
- `xarray>=0.19.0`

### Visualization

```bash
pip install "paramlake[visualize]"
```

Adds plotting and visualization:
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`

### Development Tools

```bash
pip install "paramlake[dev]"
```

Adds development and testing tools:
- `pytest>=7.0.0`
- `black>=22.0.0`
- `isort>=5.10.0`
- `mypy>=0.950`

### Everything

```bash
pip install "paramlake[all]"
```

Installs all optional dependencies.

## UV Package Manager

If you're using [uv](https://github.com/astral-sh/uv) for faster package management:

```bash
# Basic installation
uv add paramlake

# Framework-specific
uv add "paramlake[torch]"
uv add "paramlake[tf]"
uv add "paramlake[jax]"

# Combined
uv add "paramlake[tf,torch]"
uv add "paramlake[all]"
```

## Framework Detection

ParamLake automatically detects which frameworks are available at runtime:

```python
from paramlake.utils.framework_utils import framework_info

# Check what's available
info = framework_info()
print("Available frameworks:", info["available"])
print("Versions:", info["versions"])
```

## Usage Examples

### Framework-Agnostic Usage

```python
from paramlake import Repo
from paramlake.utils.framework_utils import detect_model_framework

# Works with any supported framework
repo = Repo("my-model-repo")

# Framework is auto-detected
model = create_your_model()  # TensorFlow, PyTorch, or JAX
framework = detect_model_framework(model)
print(f"Detected framework: {framework}")

# Universal operations
repo.commit(model, "Initial commit")
```

### Framework-Specific Usage

```python
# TensorFlow
import tensorflow as tf
model = tf.keras.Sequential([...])

# PyTorch  
import torch.nn as nn
model = nn.Sequential([...])

# JAX/Flax
import flax.linen as nn
model = nn.Dense(features=10)

# All work the same way with ParamLake
repo.commit(model, "Model checkpoint")
```

## Migration from All-Inclusive Installation

If you previously installed ParamLake with all dependencies and want to switch to optional dependencies:

```bash
# Uninstall old version
pip uninstall paramlake

# Install with specific frameworks
pip install "paramlake[torch]"  # or whatever you need
```

## Troubleshooting

### Missing Framework Error

If you see an error like:
```
FrameworkError: TensorFlow is required for this operation. 
Install with: pip install 'paramlake[tf]' or pip install tensorflow
```

Install the missing framework:
```bash
pip install "paramlake[tf]"
```

### Import Warnings

ParamLake will issue warnings if optional frameworks are missing but not required:
```
UserWarning: tensorflow is not available. Some functionality may be limited. 
Install with: pip install 'paramlake[tf]'
```

These warnings are informational and can be ignored if you don't need that framework.

### Framework Detection Issues

Check available frameworks:
```python
from paramlake.utils.framework_utils import get_available_frameworks

print("Available:", get_available_frameworks())
```

## Performance Benefits

Using optional dependencies provides several benefits:

1. **Faster Installation**: Only install what you need
2. **Smaller Environment**: Reduced disk space and memory usage
3. **Fewer Conflicts**: Avoid dependency version conflicts
4. **Environment Isolation**: Different projects can use different frameworks

## Recommended Installations

### For Researchers
```bash
pip install "paramlake[tf,torch,hf,visualize]"
```

### For Production
```bash
pip install "paramlake[torch]"  # or your specific framework
```

### For Experimentation
```bash
pip install "paramlake[all]"
```

### For CI/CD
```bash
pip install "paramlake[torch,dev]"  # framework + testing tools
``` 