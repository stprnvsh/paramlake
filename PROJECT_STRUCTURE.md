# ParamLake Framework Project Structure

This document provides an overview of the ParamLake framework structure and its components.

## Project Organization

```
paramlake/
├── paramlake/            # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── default_config.yaml    # Default configuration
│   ├── collectors/            # Data collection modules
│   │   ├── __init__.py        # Package initialization
│   │   ├── weight_collector.py     # Collect weights and parameters
│   │   ├── gradient_collector.py   # Collect gradients during training
│   │   └── activation_collector.py # Collect layer activations
│   ├── decorators/            # Decorator implementation
│   │   ├── __init__.py        # Package initialization
│   │   └── model_decorator.py # Main @paramlake decorator
│   ├── schema/                # Schema definitions
│   │   ├── __init__.py        # Package initialization
│   │   └── schema.py          # Framework-independent schema
│   ├── storage/               # Storage implementation
│   │   ├── __init__.py        # Package initialization
│   │   ├── factory.py         # Storage manager factory
│   │   ├── storage_interface.py # Base storage interface
│   │   ├── zarr_manager.py    # Zarr storage manager
│   │   ├── zarr_analyzer.py   # Zarr data analyzer
│   │   ├── icechunk_manager.py # Icechunk storage manager
│   │   └── icechunk_analyzer.py # Icechunk data analyzer
│   └── utils/                 # Utility functions
│       ├── __init__.py        # Package initialization
│       └── config.py          # Configuration utilities
├── examples/                  # Example scripts
│   └── tensorflow_example.py  # TensorFlow example
├── README.md                  # Project README
├── LICENSE                    # MIT License
├── PROJECT_STRUCTURE.md       # This file
├── pyproject.toml             # Project metadata
└── setup.py                   # Installation script
```

## Main Components

### Collectors

Collectors are responsible for gathering data from models during training:

- **Weight Collector**: Captures trainable and non-trainable weights from model layers
  - Handles memory monitoring and adaptive collection to reduce overhead
  - Supports layer filtering by name or type
  - Estimates tensor size to optimize storage
  - Implements memory pressure monitoring to dynamically adjust collection frequency

- **Gradient Collector**: Captures gradients for trainable weights
  - Maps variables to their respective layers for efficient gradient tracking
  - Supports multiple gradient capture methods (train_step override, optimizer override, callback)
  - Automatically selects the best method for the model architecture
  - Implements restoration of original methods post-training
  - Provides diagnostic information about gradient magnitudes

- **Activation Collector**: Captures intermediate layer outputs (activations)
  - Supports multiple neural network architectures (Functional API, Sequential, Subclassed)
  - Can generate sample inputs to capture activations
  - Handles capturing activations from any layer in the model

### Decorators

The decorator system provides a simple interface for users:

- **Model Decorator**: Main decorator (`@paramlake`) that can be applied to models or training functions
  - Automatically detects and instruments models
  - Provides a Keras callback for integration with standard training loops
  - Wraps training steps for custom training loops
  - Manages the collector lifecycle (initialization, collection, finalization)
  - Configures automatic gradient tracking based on user settings
  - Ensures clean restoration of original methods after training

### Schema

Schema definitions for standardizing data across frameworks:

- **Schema**: Framework-independent schema for layers, tensors, and other data
  - Defines standard mappings for different layer types across frameworks
  - Normalizes tensor names across frameworks (TensorFlow, PyTorch, JAX)
  - Provides the data structure specification for ParamLake data

### Storage

Storage components handle efficient storage and retrieval of data:

- **Storage Interface**: Abstract interface for storage managers, ensuring consistent API
  - Defines methods for creating layer groups, storing tensors, and managing metadata
  - Provides a unified API for different storage backends

- **Storage Factory**: Creates appropriate storage manager based on configuration
  - Dynamically selects between Zarr and Icechunk based on configuration
  - Handles fallbacks when optional dependencies are not available

- **Zarr Manager**: Manages writing data to Zarr stores with appropriate chunking and compression
  - Implements optimized chunking strategies for different tensor shapes and tensor types
  - Supports compression algorithms (blosc, zstd, lz4)
  - Handles both synchronous and asynchronous writes
  - Monitors memory usage for adaptive collection strategies
  - Provides specialized gradient storage with optimized chunking

- **Zarr Analyzer**: Provides tools for analyzing and visualizing the stored data
  - Computes statistics on weights, gradients, and activations
  - Generates visualizations of parameter evolution and gradient behavior
  - Supports comparing multiple training runs
  - Implements lazy loading to efficiently handle large datasets

- **Icechunk Manager**: Manages writing data to Icechunk repositories with transactional semantics
  - Supports cloud storage backends (S3, GCS, Azure)
  - Implements optimized chunking strategies for different tensor types
  - Handles transactions and commits for ensuring data consistency
  - Provides metadata tracking and tagging for model versions
  - Includes gradient-specific storage optimizations

- **Icechunk Analyzer**: Provides tools for analyzing data stored in Icechunk repositories
  - Supports efficient querying of model histories and snapshots
  - Generates visualizations of parameter and gradient evolution
  - Enables comparisons between different snapshots and training runs
  - Implements analysis of gradient statistics across model layers

### Utils

Utility functions for configuration and other common tasks:

- **Config**: Handles loading and validating configuration from YAML files or inline parameters
  - Implements deep dictionary merging for configuration hierarchies
  - Provides validation and defaults for all configuration options
  - Supports gradient-specific configuration settings
  - Enables specialized compression and chunking for different tensor types

## Data Flow

1. The user applies the `@paramlake` decorator to a model or training function
2. When training starts, the decorator initializes the storage manager and collectors
3. During training, at specified intervals:
   - The weight collector captures layer weights
   - The gradient collector automatically captures gradients using the appropriate method
   - The activation collector captures activations (if enabled)
4. All data is stored with optimized chunking and compression
5. After training, the user can analyze the data using the appropriate analyzer

## Configuration Options

Configuration can be provided via YAML files or inline parameters:

- **Basic Options**: Output path, run ID, capture frequency, what to capture
- **Gradient Options**: Auto tracking, tracking method, specialized storage settings
- **Layer Filtering**: Include/exclude specific layers or layer types
- **Compression**: Algorithm, level, and tensor-type specific compression settings
- **Chunking**: How to chunk the data for efficient storage and retrieval
- **Activation Capture**: Sample input configuration for activation capture
- **Performance Options**: Async writes, buffer size, adaptive collection, memory thresholds
- **Cloud Storage**: Configuration for S3, GCS, and Azure storage backends
- **Icechunk Options**: Commit frequency, tagging, and other Icechunk-specific settings

## Advanced Features

- **Automatic Gradient Capture**: Multiple methods for capturing gradients without manual instrumentation
- **Memory-Adaptive Collection**: Automatically adjusts collection frequency based on memory pressure
- **Async Storage**: Supports asynchronous writes to reduce training overhead
- **Cloud-Native Storage**: Optimized for cloud object storage with Icechunk
- **Transactional Consistency**: Ensures data integrity with atomic commits in Icechunk
- **Framework Compatibility**: Core schema designed to support TensorFlow, PyTorch, and JAX
- **Efficient Data Analysis**: Lazy loading and caching for efficient analysis of large datasets
- **Gradient Analysis Tools**: Built-in support for analyzing and visualizing gradient behavior
- **Version Control**: History tracking and snapshot management with Icechunk 