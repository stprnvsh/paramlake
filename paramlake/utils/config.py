"""
Configuration utilities for ParamLake Zarr.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(config: Optional[Union[str, Dict[str, Any]]] = None, **kwargs: Any) -> 'ParamLakeConfig':
    """
    Load and create a ParamLakeConfig from a file path, dictionary, or keyword arguments.
    
    Args:
        config: Path to YAML config file or dictionary with config values
        **kwargs: Additional config overrides passed directly
        
    Returns:
        ParamLakeConfig instance
    """
    return ParamLakeConfig(config, **kwargs)


class ParamLakeConfig:
    """Configuration manager for ParamLake Zarr."""

    DEFAULT_CONFIG = {
        # Basic options
        "output_path": "paramlake_data.zarr",
        "run_id": None,  # If None, will generate a timestamp-based ID
        "capture_frequency": 1,  # Capture every N steps/epochs
        "capture_gradients": True,
        "capture_weights": True,
        "capture_non_trainable": True,
        "capture_activations": False,
        
        # Layer filtering
        "include_layers": None,  # None means include all
        "exclude_layers": None,  # None means exclude none
        "include_types": None,   # None means include all types
        
        # Gradient options
        "gradients": {
            "enabled": True,  # Whether to capture gradients
            "auto_tracking": True,  # Whether to use automatic gradient tracking
            "track_method": "auto",  # "auto", "train_step", "optimizer", or "callback"
        },
        
        # Compression options
        "compression": {
            "algorithm": "blosc_zstd",
            "level": 3,
            "shuffle": True,
        },
        
        # Gradient-specific compression (overrides main compression if specified)
        "gradient_compression": {
            "algorithm": "blosc_zstd",  # Usually good for gradients
            "level": 5,  # Higher compression for gradients
            "shuffle": True,
        },
        
        # Chunking strategy
        "chunking": {
            "time_dimension": 1,  # Chunk every timestep separately
            "spatial_dimensions": "auto",  # Auto-determine based on tensor shape
            "target_chunk_size": 1048576,  # 1MB target chunk size
            "gradient_chunk_size": 524288,  # 512KB for gradients (smaller chunks)
        },
        
        # Activation capture options (only used if capture_activations is True)
        "activations": {
            "sample_batch": None,  # Path to sample input batch or None
            "sample_batch_size": 1,  # Number of samples to use
        },
        
        # Performance options
        "async_writes": False,  # Enable asynchronous writes
        "buffer_size": 100,     # Size of the async write queue
        "adaptive_collection": False,  # Enable adaptive collection frequency
        "memory_threshold": 80,  # Memory usage threshold percentage
        "memory_check_interval": 10,  # How often to check memory (seconds)
        
        # Storage backend options
        "storage_backend": "zarr",  # Options: "zarr", "icechunk"
        "storage_type": "local",    # Options: "local", "s3", "gcs", "azure"
        
        # Cloud storage options
        "bucket": None,          # S3/GCS bucket name
        "prefix": None,          # Storage prefix 
        "region": "us-east-1",   # Cloud region
        "account": None,         # Azure account name
        "container": None,       # Azure container name
        
        # Icechunk specific options
        "create_repo": False,    # Create repo if it doesn't exist
        "icechunk": {
            "commit_frequency": 10,  # Commit every N steps
            "tag_snapshots": False,  # Create tags for important snapshots
        },
        
        # Debug options
        "verbose": False,  # Enable verbose logging
    }

    def __init__(
        self,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize configuration with defaults and user overrides.
        
        Args:
            config: Path to YAML config file or dictionary with config values
            **kwargs: Additional config overrides passed directly to the decorator
        """
        # Start with default config
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from YAML file if provided
        if isinstance(config, str):
            yaml_config = self._load_yaml_config(config)
            if yaml_config:
                self._update_config(yaml_config)
        # Or update from dict
        elif isinstance(config, dict):
            self._update_config(config)
            
        # Override with any kwargs passed directly to the decorator
        if kwargs:
            self._update_config(kwargs)
            
        # Validate the final configuration
        self._validate_config()

    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = os.path.expanduser(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")

    def _update_config(self, updates: Dict[str, Any]) -> None:
        """Recursively update configuration with new values."""
        for key, value in updates.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(self.config[key], value)
            else:
                # Direct update for non-dict values or new keys
                self.config[key] = value

    def _update_nested_dict(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update nested dictionary values."""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value

    def _validate_config(self) -> None:
        """Validate the configuration values."""
        # Ensure output_path is valid
        if not self.config["output_path"]:
            self.config["output_path"] = "paramlake_data.zarr"

        # Generate run_id if not provided
        if not self.config["run_id"]:
            from datetime import datetime
            self.config["run_id"] = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Ensure capture_frequency is positive
        if self.config["capture_frequency"] <= 0:
            self.config["capture_frequency"] = 1
            
        # Validate storage backend options
        if self.config["storage_backend"] == "icechunk":
            # Ensure cloud storage parameters are set if using cloud storage
            if self.config["storage_type"] == "s3" and not self.config["bucket"]:
                raise ValueError("S3 bucket must be specified when using S3 storage")
                
            if self.config["storage_type"] == "gcs" and not self.config["bucket"]:
                raise ValueError("GCS bucket must be specified when using GCS storage")
                
            if self.config["storage_type"] == "azure" and (not self.config["account"] or not self.config["container"]):
                raise ValueError("Azure account and container must be specified when using Azure storage")
        else:
            # For Zarr storage, create the output directory if it doesn't exist
            output_dir = os.path.dirname(self.config["output_path"])
            if output_dir and not os.path.exists(output_dir):
                Path(output_dir).mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value."""
        return self.config[key]
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self.config.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return self.config.copy() 