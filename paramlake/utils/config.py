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
        "capture_optimizer_state": True,  # Added new option
        
        # Repository connection and discovery options
        "repository": {
            "auto_discovery": True,  # Automatically search for repositories in parent directories
            "discovery_depth": 10,  # Maximum depth to search for repositories
            "connect_existing": False,  # Flag to indicate connecting to existing repo
            "validate_on_connect": True,  # Validate repository health when connecting
            "create_repo": False,  # Whether to create a new repository
            "allow_missing_metadata": False,  # Allow connection even if some metadata is missing
            "repair_on_connect": False,  # Attempt to repair issues when connecting
            "backup_before_repair": True,  # Create backup before attempting repairs
            "connection_timeout": 30,  # Timeout for repository connection in seconds
        },
        
        # Repository metadata and tracking
        "metadata": {
            "track_connection_history": True,  # Track when and how repository was accessed
            "store_user_info": True,  # Store user information in repository metadata
            "store_environment_info": True,  # Store environment information
            "connection_log_file": None,  # File to log connection attempts (None for no logging)
            "last_accessed_file": ".last_accessed",  # File to track last access time
            "user_preferences_file": ".user_preferences",  # User-specific preferences
        },
        
        # Multi-repository management
        "multi_repo": {
            "enabled": False,  # Enable multi-repository features
            "default_repo": None,  # Default repository name or path
            "repo_registry": {},  # Registry of known repositories
            "auto_switch": False,  # Automatically switch to most appropriate repo
            "sync_preferences": True,  # Sync preferences across repositories
        },
        
        # Repository search and indexing
        "discovery": {
            "search_patterns": [
                "*.zarr",
                "*.paramlake",
                ".icechunk",
                "icechunk.json",
                ".zarray",
                ".zgroup",
                "zarr.json",
                ".paramlake",
                "paramlake.json"
            ],
            "exclude_patterns": [
                "node_modules",
                ".git",
                "__pycache__",
                "*.tmp",
                "*.temp"
            ],
            "max_search_time": 5,  # Maximum time to spend searching (seconds)
            "cache_search_results": True,  # Cache search results
            "search_cache_ttl": 3600,  # Search cache time-to-live (seconds)
        },
        
        # Connection validation and health checks
        "validation": {
            "check_storage_integrity": True,  # Check storage backend integrity
            "check_metadata_consistency": True,  # Check metadata consistency
            "check_branch_validity": True,  # Validate branch references
            "check_tag_validity": True,  # Validate tag references
            "check_snapshot_integrity": False,  # Deep validation of snapshots (expensive)
            "max_validation_time": 10,  # Maximum time for validation (seconds)
            "validation_log_level": "WARNING",  # Log level for validation messages
        },
        
        # Repository repair and recovery
        "repair": {
            "auto_repair_minor_issues": False,  # Automatically fix minor issues
            "backup_before_repair": True,  # Always backup before repairs
            "repair_log_file": None,  # Log file for repair operations
            "max_repair_attempts": 3,  # Maximum repair attempts
            "repair_strategies": ["metadata", "references", "indexes"],  # Repair strategies to try
        },
        
        # Workspace and project management
        "workspace": {
            "enabled": False,  # Enable workspace features
            "workspace_file": ".paramlake/workspace.json",  # Workspace configuration file
            "project_roots": [],  # List of project root directories
            "auto_detect_projects": True,  # Automatically detect project boundaries
            "share_repos_across_projects": True,  # Allow sharing repos across projects
        },
        
        # Repository aliases and shortcuts
        "aliases": {
            "enabled": True,  # Enable repository aliases
            "alias_file": ".paramlake/aliases.json",  # File to store aliases
            "auto_create_aliases": True,  # Automatically create aliases for frequently used repos
            "global_aliases": {},  # Global aliases available across all projects
        },
        
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
        
        # Metrics collection options
        "metrics": {
            "enabled": True,  # Whether to collect metrics
            "capture_frequency": 1,  # Compute metrics every N steps
            "compute": ["l2", "mean", "var", "max", "min", "sparsity"],  # Basic metrics
            "advanced_compute": [],  # Optional advanced metrics (spectral_norm, etc.)
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
        "endpoint_url": None,    # Custom endpoint URL (e.g., for S3-compatible services)
        
        # Icechunk specific options
        "create_repo": False,    # Create repo if it doesn't exist
        "icechunk": {
            "commit_frequency": 10,  # Commit every N steps
            "tag_snapshots": False,  # Create tags for important snapshots
            "auto_create_branches": False,  # Automatically create branches when needed
            "verbose": False,  # Enable verbose Icechunk logging
        },
        
        # Git-like version control features
        "git_features": {
            "enabled": True,  # Enable Git-like version control features
            "default_branch": "main",  # Default branch name
            "conflict_resolution": "auto",  # Default conflict resolution strategy
            "auto_commit": False,  # Automatically commit after training epochs
            "auto_tag": False,  # Automatically create tags for important milestones
            "auto_commit_frequency": 10,  # Auto-commit every N epochs
            "auto_tag_frequency": 50,  # Auto-tag every N epochs
            "remotes": {},  # Remote repository configurations
            "merge_strategy": "auto",  # Default merge strategy for branches
            "rebase_strategy": "detect",  # Default rebase strategy
            "tag_prefix": "v",  # Prefix for auto-generated tags
            "branch_protection": {
                "main": True,  # Protect main branch from deletion
                "patterns": ["release/*", "hotfix/*"],  # Protected branch patterns
            },
        },
        
        # Advanced Icechunk configuration
        "advanced_icechunk": {
            # Storage settings
            "storage_class": "STANDARD",  # S3 storage class
            "metadata_storage_class": "STANDARD",  # Metadata storage class
            "chunks_storage_class": "STANDARD",  # Chunk data storage class
            
            # Concurrency settings
            "max_concurrent_requests": 10,  # Max concurrent requests for object operations
            "ideal_request_size": 1000000,  # Ideal size for concurrent requests (1MB)
            
            # Caching configuration
            "cache_snapshot_nodes": 100,  # Number of snapshot nodes to cache
            "cache_chunk_refs": 100,  # Number of chunk references to cache
            "cache_transaction_changes": 100,  # Number of transaction changes to cache
            "cache_bytes_attributes": 10000,  # Bytes for attribute caching
            "cache_bytes_chunks": 1000000,  # Bytes for chunk caching (1MB)
            
            # Compression settings (overrides main compression for Icechunk)
            "compression_algorithm": "zstd",  # zstd, lz4
            "compression_level": 3,  # Compression level (1-22 for zstd)
        },
        
        # Branch and merge configuration
        "branch_config": {
            "default_merge_strategy": "auto",  # auto, ours, theirs, manual
            "allow_fast_forward": True,  # Allow fast-forward merges
            "require_merge_message": False,  # Require message for merge commits
            "auto_delete_merged_branches": False,  # Auto-delete branches after merge
            "experimental_branch_prefix": "experiment/",  # Prefix for experimental branches
            "feature_branch_prefix": "feature/",  # Prefix for feature branches
            "hotfix_branch_prefix": "hotfix/",  # Prefix for hotfix branches
        },
        
        # Remote repository configuration
        "remotes": {
            # Example remote configuration
            # "origin": {
            #     "type": "s3",  # s3, gcs, azure, local
            #     "bucket": "shared-models",
            #     "prefix": "team-repo",
            #     "region": "us-east-1",
            #     "credentials": "from_env"  # from_env, explicit, profile
            # }
        },
        
        # Conflict resolution configuration
        "conflict_resolution": {
            "default_strategy": "detect",  # detect, auto, ours, theirs
            "auto_resolve_simple": True,  # Auto-resolve simple conflicts
            "backup_conflicts": True,  # Create backup before resolving conflicts
            "merge_tools": [],  # External merge tools for complex conflicts
            "conflict_markers": True,  # Add conflict markers to merged files
        },
        
        # Checkpoint and versioning configuration
        "checkpoints": {
            "enabled": True,  # Enable checkpoint functionality
            "auto_save": False,  # Automatically save checkpoints
            "auto_save_frequency": 25,  # Save checkpoint every N epochs
            "max_checkpoints": 10,  # Maximum number of checkpoints to keep
            "checkpoint_format": "icechunk",  # icechunk, zarr, hdf5
            "include_optimizer": True,  # Include optimizer state in checkpoints
            "include_metadata": True,  # Include metadata in checkpoints
            "compression": True,  # Compress checkpoint data
        },
        
        # Analysis and reporting configuration
        "analysis": {
            "enabled": True,  # Enable analysis features
            "auto_generate_reports": False,  # Auto-generate training reports
            "report_frequency": 100,  # Generate report every N epochs
            "report_format": "html",  # html, json, pdf
            "include_visualizations": True,  # Include plots in reports
            "metrics_to_track": ["loss", "accuracy", "gradient_norm"],  # Metrics to track
            "layer_analysis": True,  # Enable per-layer analysis
        },
        
        # Collaboration features
        "collaboration": {
            "enabled": False,  # Enable collaboration features
            "sync_frequency": 60,  # Sync with remote every N seconds
            "conflict_notification": True,  # Notify on conflicts
            "auto_pull": False,  # Automatically pull updates
            "auto_push": False,  # Automatically push commits
            "team_settings": {
                "require_review": False,  # Require review for merges
                "max_branch_age": 30,  # Max days before branch is considered stale
            },
        },
        
        # Experimental features
        "experimental": {
            "lazy_loading": True,  # Enable lazy loading for large datasets
            "async_analysis": False,  # Enable asynchronous analysis
            "distributed_storage": False,  # Enable distributed storage
            "model_diffing": True,  # Enable advanced model diffing
            "parameter_tracking": True,  # Track individual parameter changes
        },
        
        # Notification and monitoring
        "notifications": {
            "enabled": False,  # Enable notifications
            "webhook_url": None,  # Webhook URL for notifications
            "email_settings": {
                "enabled": False,
                "smtp_server": None,
                "smtp_port": 587,
                "recipients": [],
            },
            "events": {
                "training_complete": True,
                "merge_conflicts": True,
                "checkpoint_saved": False,
                "branch_created": False,
            },
        },
        
        # Debug options
        "verbose": False,  # Enable verbose logging
        "debug_git": False,  # Enable detailed Git operation debugging
        "debug_storage": False,  # Enable storage operation debugging
        "debug_analysis": False,  # Enable analysis debugging
        "log_level": "INFO",  # Logging level: DEBUG, INFO, WARNING, ERROR
        "log_file": None,  # Path to log file (None for console only)
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
        
        # Validate git features configuration
        git_config = self.config.get("git_features", {})
        if git_config.get("enabled", False):
            # Ensure valid branch names
            default_branch = git_config.get("default_branch", "main")
            if not default_branch or not isinstance(default_branch, str):
                git_config["default_branch"] = "main"
            
            # Validate conflict resolution strategy
            valid_strategies = ["auto", "detect", "ours", "theirs", "manual"]
            if git_config.get("conflict_resolution") not in valid_strategies:
                git_config["conflict_resolution"] = "auto"
            
            # Validate auto-commit/tag frequencies
            if git_config.get("auto_commit_frequency", 0) <= 0:
                git_config["auto_commit_frequency"] = 10
            if git_config.get("auto_tag_frequency", 0) <= 0:
                git_config["auto_tag_frequency"] = 50
        
        # Validate checkpoint configuration
        checkpoint_config = self.config.get("checkpoints", {})
        if checkpoint_config.get("enabled", True):
            # Ensure valid checkpoint format
            valid_formats = ["icechunk", "zarr", "hdf5"]
            if checkpoint_config.get("checkpoint_format") not in valid_formats:
                checkpoint_config["checkpoint_format"] = "icechunk"
            
            # Ensure positive frequencies and limits
            if checkpoint_config.get("auto_save_frequency", 0) <= 0:
                checkpoint_config["auto_save_frequency"] = 25
            if checkpoint_config.get("max_checkpoints", 0) <= 0:
                checkpoint_config["max_checkpoints"] = 10
        
        # Validate analysis configuration
        analysis_config = self.config.get("analysis", {})
        if analysis_config.get("enabled", True):
            # Ensure valid report format
            valid_formats = ["html", "json", "pdf"]
            if analysis_config.get("report_format") not in valid_formats:
                analysis_config["report_format"] = "html"
            
            # Ensure positive report frequency
            if analysis_config.get("report_frequency", 0) <= 0:
                analysis_config["report_frequency"] = 100
        
        # Validate remote configurations
        remotes_config = self.config.get("remotes", {})
        for remote_name, remote_config in remotes_config.items():
            if not isinstance(remote_config, dict):
                continue
            
            # Ensure valid remote type
            valid_types = ["s3", "gcs", "azure", "local"]
            if remote_config.get("type") not in valid_types:
                remote_config["type"] = "s3"
            
            # Type-specific validation
            if remote_config["type"] in ["s3", "gcs"] and not remote_config.get("bucket"):
                raise ValueError(f"Remote '{remote_name}': bucket required for {remote_config['type']} storage")
            elif remote_config["type"] == "azure" and not remote_config.get("container"):
                raise ValueError(f"Remote '{remote_name}': container required for Azure storage")
            elif remote_config["type"] == "local" and not remote_config.get("path"):
                raise ValueError(f"Remote '{remote_name}': path required for local storage")

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value."""
        return self.config[key]
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        return self.config.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return self.config.copy()
    
    def get_git_config(self) -> Dict[str, Any]:
        """Get git-specific configuration."""
        return self.config.get("git_features", {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """Get checkpoint-specific configuration."""
        return self.config.get("checkpoints", {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis-specific configuration."""
        return self.config.get("analysis", {})
    
    def get_remote_config(self, remote_name: str = "origin") -> Optional[Dict[str, Any]]:
        """Get configuration for a specific remote."""
        return self.config.get("remotes", {}).get(remote_name)
    
    def is_git_enabled(self) -> bool:
        """Check if git features are enabled."""
        return self.config.get("git_features", {}).get("enabled", False)
    
    def is_icechunk_backend(self) -> bool:
        """Check if using Icechunk backend."""
        return self.config.get("storage_backend") == "icechunk"
    
    def get_conflict_resolution_strategy(self) -> str:
        """Get the default conflict resolution strategy."""
        return self.config.get("conflict_resolution", {}).get("default_strategy", "detect")
    
    def should_auto_commit(self) -> bool:
        """Check if auto-commit is enabled."""
        return self.config.get("git_features", {}).get("auto_commit", False)
    
    def should_auto_tag(self) -> bool:
        """Check if auto-tagging is enabled."""
        return self.config.get("git_features", {}).get("auto_tag", False)
    
    def get_auto_commit_frequency(self) -> int:
        """Get auto-commit frequency."""
        return self.config.get("git_features", {}).get("auto_commit_frequency", 10)
    
    def get_auto_tag_frequency(self) -> int:
        """Get auto-tag frequency."""
        return self.config.get("git_features", {}).get("auto_tag_frequency", 50)
    
    def export_config(self, file_path: str) -> None:
        """Export current configuration to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    # Repository connection helper methods
    def get_repository_config(self) -> Dict[str, Any]:
        """Get repository-specific configuration."""
        return self.config.get("repository", {})
    
    def get_discovery_config(self) -> Dict[str, Any]:
        """Get repository discovery configuration."""
        return self.config.get("discovery", {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get repository validation configuration."""
        return self.config.get("validation", {})
    
    def get_repair_config(self) -> Dict[str, Any]:
        """Get repository repair configuration."""
        return self.config.get("repair", {})
    
    def get_metadata_config(self) -> Dict[str, Any]:
        """Get metadata tracking configuration."""
        return self.config.get("metadata", {})
    
    def get_workspace_config(self) -> Dict[str, Any]:
        """Get workspace management configuration."""
        return self.config.get("workspace", {})
    
    def get_aliases_config(self) -> Dict[str, Any]:
        """Get repository aliases configuration."""
        return self.config.get("aliases", {})
    
    def get_multi_repo_config(self) -> Dict[str, Any]:
        """Get multi-repository management configuration."""
        return self.config.get("multi_repo", {})
    
    def is_auto_discovery_enabled(self) -> bool:
        """Check if automatic repository discovery is enabled."""
        return self.config.get("repository", {}).get("auto_discovery", True)
    
    def should_validate_on_connect(self) -> bool:
        """Check if repository validation should be performed on connection."""
        return self.config.get("repository", {}).get("validate_on_connect", True)
    
    def should_connect_existing(self) -> bool:
        """Check if this is a connection to an existing repository."""
        return self.config.get("repository", {}).get("connect_existing", False) or self.config.get("connect_existing", False)
    
    def should_create_repo(self) -> bool:
        """Check if a new repository should be created."""
        return self.config.get("repository", {}).get("create_repo", False) or self.config.get("create_repo", False)
    
    def should_allow_missing_metadata(self) -> bool:
        """Check if connection should be allowed even with missing metadata."""
        return self.config.get("repository", {}).get("allow_missing_metadata", False)
    
    def should_repair_on_connect(self) -> bool:
        """Check if repository repair should be attempted on connection."""
        return self.config.get("repository", {}).get("repair_on_connect", False)
    
    def get_discovery_depth(self) -> int:
        """Get the maximum depth to search for repositories."""
        return self.config.get("repository", {}).get("discovery_depth", 10)
    
    def get_connection_timeout(self) -> int:
        """Get the connection timeout in seconds."""
        return self.config.get("repository", {}).get("connection_timeout", 30)
    
    def get_search_patterns(self) -> List[str]:
        """Get patterns to search for when discovering repositories."""
        return self.config.get("discovery", {}).get("search_patterns", [
            "*.zarr", "*.paramlake", ".icechunk", "icechunk.json",
            ".zarray", ".zgroup", "zarr.json", ".paramlake", "paramlake.json"
        ])
    
    def get_exclude_patterns(self) -> List[str]:
        """Get patterns to exclude when searching for repositories."""
        return self.config.get("discovery", {}).get("exclude_patterns", [
            "node_modules", ".git", "__pycache__", "*.tmp", "*.temp"
        ])
    
    def get_max_search_time(self) -> int:
        """Get maximum time to spend searching for repositories."""
        return self.config.get("discovery", {}).get("max_search_time", 5)
    
    def should_cache_search_results(self) -> bool:
        """Check if search results should be cached."""
        return self.config.get("discovery", {}).get("cache_search_results", True)
    
    def get_search_cache_ttl(self) -> int:
        """Get search cache time-to-live in seconds."""
        return self.config.get("discovery", {}).get("search_cache_ttl", 3600)
    
    def should_check_storage_integrity(self) -> bool:
        """Check if storage integrity should be validated."""
        return self.config.get("validation", {}).get("check_storage_integrity", True)
    
    def should_check_metadata_consistency(self) -> bool:
        """Check if metadata consistency should be validated."""
        return self.config.get("validation", {}).get("check_metadata_consistency", True)
    
    def should_check_branch_validity(self) -> bool:
        """Check if branch validity should be validated."""
        return self.config.get("validation", {}).get("check_branch_validity", True)
    
    def should_check_tag_validity(self) -> bool:
        """Check if tag validity should be validated."""
        return self.config.get("validation", {}).get("check_tag_validity", True)
    
    def should_check_snapshot_integrity(self) -> bool:
        """Check if snapshot integrity should be deeply validated."""
        return self.config.get("validation", {}).get("check_snapshot_integrity", False)
    
    def get_max_validation_time(self) -> int:
        """Get maximum time for validation in seconds."""
        return self.config.get("validation", {}).get("max_validation_time", 10)
    
    def get_validation_log_level(self) -> str:
        """Get log level for validation messages."""
        return self.config.get("validation", {}).get("validation_log_level", "WARNING")
    
    def should_auto_repair_minor_issues(self) -> bool:
        """Check if minor issues should be automatically repaired."""
        return self.config.get("repair", {}).get("auto_repair_minor_issues", False)
    
    def should_backup_before_repair(self) -> bool:
        """Check if backup should be created before repairs."""
        return self.config.get("repair", {}).get("backup_before_repair", True)
    
    def get_max_repair_attempts(self) -> int:
        """Get maximum number of repair attempts."""
        return self.config.get("repair", {}).get("max_repair_attempts", 3)
    
    def get_repair_strategies(self) -> List[str]:
        """Get list of repair strategies to try."""
        return self.config.get("repair", {}).get("repair_strategies", ["metadata", "references", "indexes"])
    
    def should_track_connection_history(self) -> bool:
        """Check if connection history should be tracked."""
        return self.config.get("metadata", {}).get("track_connection_history", True)
    
    def should_store_user_info(self) -> bool:
        """Check if user information should be stored."""
        return self.config.get("metadata", {}).get("store_user_info", True)
    
    def should_store_environment_info(self) -> bool:
        """Check if environment information should be stored."""
        return self.config.get("metadata", {}).get("store_environment_info", True)
    
    def get_connection_log_file(self) -> Optional[str]:
        """Get path to connection log file."""
        return self.config.get("metadata", {}).get("connection_log_file")
    
    def get_last_accessed_file(self) -> str:
        """Get filename for last accessed tracking."""
        return self.config.get("metadata", {}).get("last_accessed_file", ".last_accessed")
    
    def get_user_preferences_file(self) -> str:
        """Get filename for user preferences."""
        return self.config.get("metadata", {}).get("user_preferences_file", ".user_preferences")
    
    def is_workspace_enabled(self) -> bool:
        """Check if workspace features are enabled."""
        return self.config.get("workspace", {}).get("enabled", False)
    
    def get_workspace_file(self) -> str:
        """Get workspace configuration file path."""
        return self.config.get("workspace", {}).get("workspace_file", ".paramlake/workspace.json")
    
    def get_project_roots(self) -> List[str]:
        """Get list of project root directories."""
        return self.config.get("workspace", {}).get("project_roots", [])
    
    def should_auto_detect_projects(self) -> bool:
        """Check if project boundaries should be automatically detected."""
        return self.config.get("workspace", {}).get("auto_detect_projects", True)
    
    def should_share_repos_across_projects(self) -> bool:
        """Check if repositories should be shared across projects."""
        return self.config.get("workspace", {}).get("share_repos_across_projects", True)
    
    def is_aliases_enabled(self) -> bool:
        """Check if repository aliases are enabled."""
        return self.config.get("aliases", {}).get("enabled", True)
    
    def get_alias_file(self) -> str:
        """Get alias configuration file path."""
        return self.config.get("aliases", {}).get("alias_file", ".paramlake/aliases.json")
    
    def should_auto_create_aliases(self) -> bool:
        """Check if aliases should be automatically created."""
        return self.config.get("aliases", {}).get("auto_create_aliases", True)
    
    def get_global_aliases(self) -> Dict[str, str]:
        """Get global repository aliases."""
        return self.config.get("aliases", {}).get("global_aliases", {})
    
    def is_multi_repo_enabled(self) -> bool:
        """Check if multi-repository features are enabled."""
        return self.config.get("multi_repo", {}).get("enabled", False)
    
    def get_default_repo(self) -> Optional[str]:
        """Get default repository name or path."""
        return self.config.get("multi_repo", {}).get("default_repo")
    
    def get_repo_registry(self) -> Dict[str, Any]:
        """Get registry of known repositories."""
        return self.config.get("multi_repo", {}).get("repo_registry", {})
    
    def should_auto_switch_repos(self) -> bool:
        """Check if repository switching should be automatic."""
        return self.config.get("multi_repo", {}).get("auto_switch", False)
    
    def should_sync_preferences(self) -> bool:
        """Check if preferences should be synced across repositories."""
        return self.config.get("multi_repo", {}).get("sync_preferences", True) 