"""
Factory functions for creating storage managers with enhanced git feature support.
"""

from typing import Optional, Dict, Any

from paramlake.collectors.metrics_collector import MetricsCollector
from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.storage_interface import StorageInterface
from paramlake.storage.zarr_manager import ZarrStorageManager

try:
    from paramlake.storage.icechunk_manager import IcechunkStorageManager, HAS_ICECHUNK
except ImportError:
    HAS_ICECHUNK = False


def create_storage_manager(config: ParamLakeConfig) -> StorageInterface:
    """
    Create a storage manager based on the configuration.
    
    Args:
        config: ParamLake configuration
        
    Returns:
        StorageInterface: A storage manager that implements StorageInterface
    """
    storage_type = config.get("storage_type", "zarr")
    
    # Enable verbose mode for debugging if not explicitly set
    config_dict = config.to_dict()
    if "verbose" not in config_dict:
        config_dict["verbose"] = True
        config = ParamLakeConfig(config_dict)
        
    if config.get("verbose", False):
        print(f"Creating storage manager of type: {storage_type}")
    
    storage_manager = None
    
    if storage_type == "icechunk":
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk storage requested but icechunk is not installed. "
                             "Install it with 'pip install icechunk'")
        
        if config.get("verbose", False):
            print("Creating IcechunkStorageManager")
        
        try:
            storage_manager = IcechunkStorageManager(config)
            if config.get("verbose", False):
                print(f"Created IcechunkStorageManager with run_id: {storage_manager.run_id}")
            
            # Enhanced Git-like feature configuration
            _configure_git_features(storage_manager, config)
            
        except Exception as e:
            print(f"Error creating IcechunkStorageManager: {e}")
            # Try to provide helpful error messages
            if "repositories can only be created in clean prefixes" in str(e):
                print("Hint: The repository location already exists. Try setting create_repo=False to open the existing repository, or use a different prefix/bucket.")
            elif "No credentials" in str(e) or "access" in str(e).lower():
                print("Hint: Check your AWS credentials are properly configured.")
            raise
                    
    else:
        # Default to Zarr
        if config.get("verbose", False):
            print("Creating ZarrStorageManager")
        storage_manager = ZarrStorageManager(config)
        if config.get("verbose", False):
            print(f"Created ZarrStorageManager with run_id: {storage_manager.run_id}")
    
    # Create and register metrics collector if enabled
    _configure_metrics_collector(storage_manager, config)
    
    # Configure analysis features if available
    _configure_analysis_features(storage_manager, config)
    
    # Configure checkpoint features
    _configure_checkpoint_features(storage_manager, config)
    
    return storage_manager


def _configure_git_features(storage_manager: StorageInterface, config: ParamLakeConfig) -> None:
    """Configure git-like features for the storage manager."""
    git_config = config.get_git_config()
    
    if not git_config.get("enabled", False):
        return
    
    if config.get("verbose", False):
        print("Configuring Git-like features...")
    
    try:
        # Set up default branch
        default_branch = git_config.get("default_branch", "main")
        if hasattr(storage_manager, 'switch_branch'):
            try:
                result = storage_manager.switch_branch(default_branch, create_if_missing=True)
                if config.get("verbose", False):
                    print(f"Switched to default branch: {default_branch}")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not switch to branch {default_branch}: {e}")
        
        # Configure remote repositories
        remotes_config = git_config.get("remotes", {})
        if remotes_config and hasattr(storage_manager, 'configure_remotes'):
            try:
                storage_manager.configure_remotes(remotes_config)
                if config.get("verbose", False):
                    print(f"Configured {len(remotes_config)} remote(s)")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not configure remotes: {e}")
        
        # Set up conflict resolution strategy
        conflict_strategy = git_config.get("conflict_resolution", "auto")
        if hasattr(storage_manager, 'set_default_conflict_strategy'):
            try:
                storage_manager.set_default_conflict_strategy(conflict_strategy)
                if config.get("verbose", False):
                    print(f"Set default conflict resolution strategy: {conflict_strategy}")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not set conflict strategy: {e}")
        
        # Configure auto-commit settings
        if git_config.get("auto_commit", False):
            auto_commit_freq = git_config.get("auto_commit_frequency", 10)
            if hasattr(storage_manager, 'enable_auto_commit'):
                try:
                    storage_manager.enable_auto_commit(frequency=auto_commit_freq)
                    if config.get("verbose", False):
                        print(f"Enabled auto-commit every {auto_commit_freq} epochs")
                except Exception as e:
                    if config.get("verbose", False):
                        print(f"Warning: Could not enable auto-commit: {e}")
        
        # Configure auto-tagging settings
        if git_config.get("auto_tag", False):
            auto_tag_freq = git_config.get("auto_tag_frequency", 50)
            tag_prefix = git_config.get("tag_prefix", "v")
            if hasattr(storage_manager, 'enable_auto_tag'):
                try:
                    storage_manager.enable_auto_tag(frequency=auto_tag_freq, prefix=tag_prefix)
                    if config.get("verbose", False):
                        print(f"Enabled auto-tagging every {auto_tag_freq} epochs with prefix '{tag_prefix}'")
                except Exception as e:
                    if config.get("verbose", False):
                        print(f"Warning: Could not enable auto-tagging: {e}")
        
        # Configure branch protection
        branch_protection = git_config.get("branch_protection", {})
        if branch_protection and hasattr(storage_manager, 'configure_branch_protection'):
            try:
                storage_manager.configure_branch_protection(branch_protection)
                if config.get("verbose", False):
                    print("Configured branch protection rules")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not configure branch protection: {e}")
        
        if config.get("verbose", False):
            print("✓ Git-like features configured successfully")
            
    except Exception as e:
        print(f"Error configuring git features: {e}")


def _configure_metrics_collector(storage_manager: StorageInterface, config: ParamLakeConfig) -> None:
    """Configure metrics collection for the storage manager."""
    if not config.get("metrics", {}).get("enabled", True):
        return
    
    if config.get("verbose", False):
        print("Creating MetricsCollector")
    
    try:
        metrics_config = config.get("metrics", {})
        enabled_metrics = metrics_config.get("compute", ["l2", "mean", "var", "max", "min", "sparsity"])
        enabled_metrics += metrics_config.get("advanced_compute", [])
        
        metrics_collector = MetricsCollector(
            storage_manager,
            enabled_metrics=enabled_metrics,
            include_layers=config.get("include_layers"),
            exclude_layers=config.get("exclude_layers"),
            capture_frequency=metrics_config.get("capture_frequency", config.get("capture_frequency", 1))
        )
        
        storage_manager.register_metrics_collector(metrics_collector)
        
        if config.get("verbose", False):
            print(f"✓ Registered MetricsCollector with {len(enabled_metrics)} metrics")
            
    except Exception as e:
        print(f"Warning: Could not create MetricsCollector: {e}")


def _configure_analysis_features(storage_manager: StorageInterface, config: ParamLakeConfig) -> None:
    """Configure analysis features for the storage manager."""
    analysis_config = config.get_analysis_config()
    
    if not analysis_config.get("enabled", True):
        return
    
    if config.get("verbose", False):
        print("Configuring analysis features...")
    
    try:
        # Configure auto-report generation
        if analysis_config.get("auto_generate_reports", False):
            report_freq = analysis_config.get("report_frequency", 100)
            report_format = analysis_config.get("report_format", "html")
            
            if hasattr(storage_manager, 'enable_auto_reports'):
                try:
                    storage_manager.enable_auto_reports(
                        frequency=report_freq,
                        format=report_format,
                        include_visualizations=analysis_config.get("include_visualizations", True)
                    )
                    if config.get("verbose", False):
                        print(f"Enabled auto-report generation every {report_freq} epochs")
                except Exception as e:
                    if config.get("verbose", False):
                        print(f"Warning: Could not enable auto-reports: {e}")
        
        # Configure metrics tracking
        metrics_to_track = analysis_config.get("metrics_to_track", ["loss", "accuracy", "gradient_norm"])
        if hasattr(storage_manager, 'set_tracked_metrics'):
            try:
                storage_manager.set_tracked_metrics(metrics_to_track)
                if config.get("verbose", False):
                    print(f"Tracking metrics: {metrics_to_track}")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not set tracked metrics: {e}")
        
        # Enable layer-wise analysis if requested
        if analysis_config.get("layer_analysis", True):
            if hasattr(storage_manager, 'enable_layer_analysis'):
                try:
                    storage_manager.enable_layer_analysis(True)
                    if config.get("verbose", False):
                        print("Enabled layer-wise analysis")
                except Exception as e:
                    if config.get("verbose", False):
                        print(f"Warning: Could not enable layer analysis: {e}")
        
        if config.get("verbose", False):
            print("✓ Analysis features configured")
            
    except Exception as e:
        print(f"Error configuring analysis features: {e}")


def _configure_checkpoint_features(storage_manager: StorageInterface, config: ParamLakeConfig) -> None:
    """Configure checkpoint features for the storage manager."""
    checkpoint_config = config.get_checkpoint_config()
    
    if not checkpoint_config.get("enabled", True):
        return
    
    if config.get("verbose", False):
        print("Configuring checkpoint features...")
    
    try:
        # Configure auto-save checkpoints
        if checkpoint_config.get("auto_save", False):
            save_freq = checkpoint_config.get("auto_save_frequency", 25)
            max_checkpoints = checkpoint_config.get("max_checkpoints", 10)
            
            if hasattr(storage_manager, 'enable_auto_checkpoints'):
                try:
                    storage_manager.enable_auto_checkpoints(
                        frequency=save_freq,
                        max_checkpoints=max_checkpoints,
                        include_optimizer=checkpoint_config.get("include_optimizer", True),
                        compression=checkpoint_config.get("compression", True)
                    )
                    if config.get("verbose", False):
                        print(f"Enabled auto-checkpoints every {save_freq} epochs (max: {max_checkpoints})")
                except Exception as e:
                    if config.get("verbose", False):
                        print(f"Warning: Could not enable auto-checkpoints: {e}")
        
        # Set checkpoint format preference
        checkpoint_format = checkpoint_config.get("checkpoint_format", "icechunk")
        if hasattr(storage_manager, 'set_checkpoint_format'):
            try:
                storage_manager.set_checkpoint_format(checkpoint_format)
                if config.get("verbose", False):
                    print(f"Set checkpoint format: {checkpoint_format}")
            except Exception as e:
                if config.get("verbose", False):
                    print(f"Warning: Could not set checkpoint format: {e}")
        
        if config.get("verbose", False):
            print("✓ Checkpoint features configured")
            
    except Exception as e:
        print(f"Error configuring checkpoint features: {e}")


def get_storage_manager_class(storage_type: str) -> type:
    """
    Get the storage manager class for a given storage type.
    
    Args:
        storage_type: Type of storage ('zarr', 'icechunk')
        
    Returns:
        Storage manager class
    """
    if storage_type == "icechunk":
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk storage requested but icechunk is not installed. "
                             "Install it with 'pip install icechunk'")
        return IcechunkStorageManager
    else:
        return ZarrStorageManager


def validate_storage_config(config: ParamLakeConfig) -> Dict[str, Any]:
    """
    Validate storage configuration and return validation results.
    
    Args:
        config: ParamLake configuration
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "recommendations": []
    }
    
    storage_type = config.get("storage_type", "zarr")
    
    # Validate storage type
    if storage_type == "icechunk" and not HAS_ICECHUNK:
        validation["errors"].append("Icechunk requested but not installed")
        validation["valid"] = False
    
    # Validate git features
    if config.is_git_enabled():
        if storage_type != "icechunk":
            validation["warnings"].append("Git features are only fully supported with Icechunk storage")
        
        git_config = config.get_git_config()
        
        # Check for valid branch name
        default_branch = git_config.get("default_branch", "main")
        if not default_branch or "/" in default_branch or " " in default_branch:
            validation["errors"].append("Invalid default branch name")
            validation["valid"] = False
        
        # Check conflict resolution strategy
        conflict_strategy = git_config.get("conflict_resolution", "auto")
        valid_strategies = ["auto", "detect", "ours", "theirs", "manual"]
        if conflict_strategy not in valid_strategies:
            validation["errors"].append(f"Invalid conflict resolution strategy: {conflict_strategy}")
            validation["valid"] = False
    
    # Validate remote configurations
    remotes = config.get("remotes", {})
    for remote_name, remote_config in remotes.items():
        if not isinstance(remote_config, dict):
            validation["errors"].append(f"Invalid remote configuration for '{remote_name}'")
            validation["valid"] = False
            continue
        
        remote_type = remote_config.get("type", "s3")
        if remote_type not in ["s3", "gcs", "azure", "local"]:
            validation["errors"].append(f"Invalid remote type '{remote_type}' for remote '{remote_name}'")
            validation["valid"] = False
    
    # Validate cloud storage settings
    if config.get("storage_type") in ["s3", "gcs", "azure"]:
        if config.get("storage_type") == "s3" and not config.get("bucket"):
            validation["errors"].append("S3 bucket name is required")
            validation["valid"] = False
        elif config.get("storage_type") == "gcs" and not config.get("bucket"):
            validation["errors"].append("GCS bucket name is required")
            validation["valid"] = False
        elif config.get("storage_type") == "azure" and not config.get("container"):
            validation["errors"].append("Azure container name is required")
            validation["valid"] = False
    
    # Performance recommendations
    if config.is_icechunk_backend() and config.is_git_enabled():
        commit_freq = config.get("icechunk", {}).get("commit_frequency", 10)
        if commit_freq < 5:
            validation["recommendations"].append("Consider increasing commit_frequency for better performance")
        
        if config.get("capture_gradients", False) and not config.get("gradient_compression"):
            validation["recommendations"].append("Consider enabling gradient compression for better storage efficiency")
    
    return validation


def create_analyzer_from_config(config: ParamLakeConfig, snapshot_id: Optional[str] = None) -> Optional[Any]:
    """
    Create an analyzer instance from configuration.
    
    Args:
        config: ParamLake configuration
        snapshot_id: Specific snapshot to analyze
        
    Returns:
        Analyzer instance or None if not available
    """
    if not config.is_icechunk_backend():
        return None
    
    try:
        from paramlake.storage.icechunk_analyzer import IcechunkModelAnalyzer
        
        # Build storage configuration for analyzer
        storage_config = {
            "type": config.get("storage_type", "s3"),
            "bucket": config.get("bucket"),
            "prefix": config.get("prefix", "paramlake_data"),
            "region": config.get("region", "us-east-1"),
            "endpoint_url": config.get("endpoint_url"),
        }
        
        if config.get("storage_type") == "local":
            storage_config = config.get("output_path")
        
        return IcechunkModelAnalyzer(
            storage_config,
            snapshot_id=snapshot_id,
            lazy_loading=config.get("experimental", {}).get("lazy_loading", True)
        )
        
    except ImportError:
        return None
    except Exception as e:
        if config.get("verbose", False):
            print(f"Could not create analyzer: {e}")
        return None 