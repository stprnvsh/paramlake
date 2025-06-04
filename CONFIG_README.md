# ParamLake Configuration Guide

## 🔧 **Quick Start Configuration**

ParamLake supports multiple configuration methods and storage backends. This guide covers all configuration options with working examples.

---

## 📋 **Table of Contents**

1. [Basic Configuration](#basic-configuration)
2. [Storage Backends](#storage-backends)
3. [Tracking Configuration](#tracking-configuration)
4. [AWS Setup](#aws-setup)
5. [Configuration Methods](#configuration-methods)
6. [Working Examples](#working-examples)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 **Basic Configuration**

### Minimal Configuration
```python
from paramlake.utils.config import ParamLakeConfig

# Minimal local configuration
config = ParamLakeConfig({
    'storage_type': 'zarr',           # or 'icechunk'
    'output_path': './model_data',    # Local path
    'capture_weights': True
})
```

### Production Configuration
```python
# Full production configuration
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'your-bucket-name',
    'prefix': 'experiments/mnist',
    'region': 'us-east-1',
    'create_repo': True,
    'capture_weights': True,
    'capture_gradients': False,       # Disable for performance
    'capture_optimizer_state': False,
    'capture_frequency': 5,           # Every 5 epochs
    'compression': {
        'algorithm': 'none',          # Disable for compatibility
        'level': 0,
        'shuffle': False
    }
})
```

---

## 🗄️ **Storage Backends**

### 1. **Local Zarr Storage** (Development)
```python
config = ParamLakeConfig({
    'storage_type': 'zarr',
    'output_path': './experiments/model_data.zarr',
    'create_repo': True,
    'compression': {
        'algorithm': 'blosc_lz4',
        'level': 3,
        'shuffle': True
    }
})
```

### 2. **IceChunk Local Storage**
```python
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 'local',
    'output_path': './icechunk_data',
    'create_repo': True,
    'icechunk': {
        'commit_frequency': 10,
        'tag_snapshots': True,
        'verbose': False
    }
})
```

### 3. **IceChunk S3 Storage** (Production)
```python
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'your-ml-experiments',
    'prefix': 'team/mnist-experiments',
    'region': 'us-east-1',
    'endpoint_url': 'https://s3.amazonaws.com',
    'create_repo': True,
    'icechunk': {
        'commit_frequency': 5,
        'tag_snapshots': True,
        'auto_commit': True,
        'verbose': False
    }
})
```

### 4. **IceChunk Google Cloud Storage**
```python
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 'gcs',
    'bucket': 'your-gcs-bucket',
    'prefix': 'ml-experiments',
    'create_repo': True
})
```

---

## 📊 **Tracking Configuration**

### Weight Tracking
```python
config = ParamLakeConfig({
    'capture_weights': True,
    'capture_non_trainable': False,   # Only trainable weights
    'include_layers': None,           # All layers (or specify list)
    'exclude_layers': ['dropout'],    # Skip dropout layers
    'capture_frequency': 5            # Every 5 epochs
})
```

### Gradient Tracking
```python
config = ParamLakeConfig({
    'capture_gradients': True,
    'gradients': {
        'enabled': True,
        'capture_frequency': 1,       # Every epoch
        'include_layers': ['dense', 'conv'],
        'compression': {
            'algorithm': 'none',      # Recommended for gradients
            'level': 0
        }
    }
})
```

### Metrics Configuration
```python
config = ParamLakeConfig({
    'metrics': {
        'enabled': True,
        'capture_frequency': 1,
        'compute': ['l2', 'mean', 'std', 'sparsity'],
        'include_layers': None,       # All layers
        'exclude_layers': []
    }
})
```

### Optimizer State Tracking
```python
config = ParamLakeConfig({
    'capture_optimizer_state': True,
    'optimizer': {
        'capture_frequency': 10,      # Less frequent (large data)
        'include_state_vars': ['momentum', 'variance']
    }
})
```

---

## 🔑 **AWS Setup**

### Method 1: Environment Variables (Recommended)
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
```

### Method 2: In Code (Development Only)
```python
import os

os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'
os.environ['AWS_REGION'] = 'us-east-1'
```

### Method 3: AWS Credentials File
```bash
# ~/.aws/credentials
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key

# ~/.aws/config
[default]
region = us-east-1
```

### S3 Bucket Setup
```python
# Ensure your bucket exists and has proper permissions
import boto3

s3 = boto3.client('s3')
bucket_name = 'your-paramlake-bucket'

# Create bucket if it doesn't exist
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"✅ Created bucket: {bucket_name}")
except:
    print(f"✅ Bucket {bucket_name} already exists")
```

---

## ⚙️ **Configuration Methods**

### 1. **Direct Dictionary**
```python
from paramlake.utils.config import ParamLakeConfig

config_dict = {
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'my-experiments',
    'capture_weights': True
}

config = ParamLakeConfig(config_dict)
```

### 2. **YAML Configuration File**
```yaml
# config.yaml
storage_type: icechunk
storage_backend: s3
bucket: my-ml-experiments
prefix: team-alpha/mnist
region: us-east-1
create_repo: true

# Tracking settings
capture_weights: true
capture_gradients: false
capture_optimizer_state: false
capture_frequency: 5

# Compression (disable for icechunk compatibility)
compression:
  algorithm: none
  level: 0
  shuffle: false

# IceChunk specific settings
icechunk:
  commit_frequency: 10
  tag_snapshots: true
  auto_commit: true
  verbose: false
```

Load YAML config:
```python
import yaml
from paramlake.utils.config import ParamLakeConfig

with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

config = ParamLakeConfig(config_dict)
```

### 3. **Using @paramlake Decorator**
```python
from paramlake import paramlake

@paramlake(
    storage_type='icechunk',
    storage_backend='s3',
    bucket='my-experiments',
    capture_weights=True,
    capture_frequency=5
)
def train_model():
    # Your training code here
    model = create_model()
    model.fit(X, y, epochs=50)
    return model
```

### 4. **Using Repo Class**
```python
from paramlake import Repo

config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'my-experiments'
})

repo = Repo('./local_cache', config=config)

# Use repo.track() decorator
@repo.track(capture_frequency=10)
def train_model():
    # Training code
    pass
```

---

## 🎯 **Working Examples**

### Example 1: Local Development
```python
from paramlake import paramlake
import tensorflow as tf

@paramlake(
    storage_type='zarr',
    output_path='./experiments/model_v1.zarr',
    capture_weights=True,
    capture_frequency=1
)
def train_locally():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
    
    return model
```

### Example 2: Cloud Production
```python
import os
from paramlake import paramlake
from datetime import datetime

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret'
os.environ['AWS_REGION'] = 'us-east-1'

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

@paramlake(
    storage_type='icechunk',
    storage_backend='s3',
    bucket='ml-experiments',
    prefix=f'mnist-cnn-{timestamp}',
    region='us-east-1',
    create_repo=True,
    capture_weights=True,
    capture_frequency=5,
    compression={'algorithm': 'none', 'level': 0, 'shuffle': False},
    icechunk={
        'commit_frequency': 10,
        'tag_snapshots': True,
        'verbose': False
    }
)
def train_production():
    model = create_cnn_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_val, y_val),
        batch_size=128
    )
    
    return model, history
```

### Example 3: Multi-Branch Experiment
```python
from paramlake import Repo
from paramlake.utils.config import ParamLakeConfig

config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'experiments',
    'prefix': 'optimizer-comparison',
    'create_repo': True
})

repo = Repo('./cache', config=config)

# Experiment 1: Adam optimizer
repo.create_branch('adam_experiment')

@repo.track(capture_frequency=5)
def train_with_adam():
    model = create_model()
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50)
    return model

# Experiment 2: SGD optimizer  
repo.create_branch('sgd_experiment')

@repo.track(capture_frequency=5)
def train_with_sgd():
    model = create_model()
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X, y, epochs=50)
    return model
```

---

## 🔧 **Performance Configuration**

### High-Performance Setup
```python
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    
    # Reduce capture frequency for large models
    'capture_frequency': 10,          # Every 10 epochs instead of 1
    
    # Disable expensive collectors
    'capture_gradients': False,       # Gradients are large
    'capture_optimizer_state': False, # Optimizer state is large
    'capture_activations': False,     # Activations are very large
    
    # Focus on weights and basic metrics
    'capture_weights': True,
    'metrics': {
        'enabled': True,
        'compute': ['l2', 'mean'],    # Only essential metrics
        'capture_frequency': 5
    },
    
    # Disable compression for speed
    'compression': {
        'algorithm': 'none',
        'level': 0,
        'shuffle': False
    },
    
    # Optimize IceChunk settings
    'icechunk': {
        'commit_frequency': 20,       # Less frequent commits
        'tag_snapshots': False,       # Disable auto-tagging
        'verbose': False
    }
})
```

### Memory-Optimized Setup
```python
config = ParamLakeConfig({
    'storage_type': 'zarr',           # Zarr for lower memory usage
    'capture_frequency': 20,          # Very infrequent capture
    'capture_weights': True,
    'capture_gradients': False,
    'capture_optimizer_state': False,
    'exclude_layers': ['large_layer1', 'large_layer2'],  # Skip large layers
    'compression': {
        'algorithm': 'blosc_lz4',     # Good compression ratio
        'level': 6,
        'shuffle': True
    }
})
```

---

## 🚨 **Troubleshooting**

### Common Issues and Solutions

#### 1. **IceChunk Import Error**
```bash
# Install icechunk
pip install icechunk

# Or for conda
conda install -c conda-forge icechunk
```

#### 2. **AWS Credentials Error**
```python
# Test your AWS setup
import boto3

try:
    s3 = boto3.client('s3')
    s3.list_buckets()
    print("✅ AWS credentials working")
except Exception as e:
    print(f"❌ AWS error: {e}")
```

#### 3. **Compression Errors with IceChunk**
```python
# Always disable compression for IceChunk
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'compression': {
        'algorithm': 'none',  # Must be 'none' for IceChunk
        'level': 0,
        'shuffle': False
    }
})
```

#### 4. **Memory Issues**
```python
# Reduce capture frequency and disable heavy collectors
config = ParamLakeConfig({
    'capture_frequency': 10,          # Capture less often
    'capture_gradients': False,       # Disable gradients
    'capture_optimizer_state': False, # Disable optimizer state
    'exclude_layers': ['very_large_layer']  # Skip large layers
})
```

#### 5. **Storage Path Issues**
```python
# Ensure directory exists for local storage
import os

output_path = './experiments/model_data.zarr'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

config = ParamLakeConfig({
    'storage_type': 'zarr',
    'output_path': output_path
})
```

---

## ✅ **Configuration Validation**

### Test Your Configuration
```python
def test_config(config):
    """Test if configuration is valid."""
    
    print("🧪 Testing ParamLake configuration...")
    
    try:
        from paramlake.storage.factory import create_storage_manager
        
        # Test storage manager creation
        storage_manager = create_storage_manager(config)
        print("✅ Storage manager created successfully")
        
        # Test basic operations
        storage_manager.set_step(0)
        print("✅ Storage manager operations working")
        
        # Clean up
        storage_manager.close()
        print("✅ Configuration test passed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()

# Test your configuration
config = ParamLakeConfig({
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'your-bucket',
    'create_repo': True
})

test_config(config)
```

---

## 📖 **Additional Resources**

- **IceChunk Documentation**: https://icechunk.io/
- **Zarr Documentation**: https://zarr.readthedocs.io/
- **AWS S3 Setup**: https://docs.aws.amazon.com/s3/
- **Project Status**: See `PROJECT_STATUS.md` for current features
- **Examples**: Check `mnist_training_comparison_demo.py` for working examples

---

## 🎯 **Quick Configuration Templates**

### Template 1: Local Development
```python
LOCAL_CONFIG = {
    'storage_type': 'zarr',
    'output_path': './experiments/model.zarr',
    'capture_weights': True,
    'capture_frequency': 1
}
```

### Template 2: Cloud Production
```python
CLOUD_CONFIG = {
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'your-bucket',
    'prefix': 'experiments',
    'region': 'us-east-1',
    'create_repo': True,
    'capture_weights': True,
    'capture_frequency': 5,
    'compression': {'algorithm': 'none', 'level': 0, 'shuffle': False}
}
```

### Template 3: High-Performance
```python
PERFORMANCE_CONFIG = {
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'your-bucket',
    'capture_frequency': 10,
    'capture_weights': True,
    'capture_gradients': False,
    'capture_optimizer_state': False,
    'compression': {'algorithm': 'none', 'level': 0, 'shuffle': False}
}
```

---

**🎉 You're ready to configure ParamLake for your ML experiments!**

For more examples and advanced usage, see the complete demos in the project repository. 