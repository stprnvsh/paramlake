"""
Factory functions for creating storage managers.
"""

from typing import Optional

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
    
    # Enable verbose mode for debugging
    # Create a copy of the config with verbose enabled
    config_dict = config.to_dict()
    if not config_dict.get("verbose", False):
        config_dict["verbose"] = True
        # Update the config object with the modified dictionary
        config = ParamLakeConfig(config_dict)
        
    print(f"Creating storage manager of type: {storage_type}")
    
    storage_manager = None
    
    if storage_type == "icechunk":
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk storage requested but icechunk is not installed. "
                             "Install it with 'pip install icechunk'")
        print("Creating IcechunkStorageManager")
        storage_manager = IcechunkStorageManager(config)
        print(f"Created IcechunkStorageManager with run_id: {storage_manager.run_id}")
    else:
        # Default to Zarr
        print("Creating ZarrStorageManager")
        storage_manager = ZarrStorageManager(config)
        print(f"Created ZarrStorageManager with run_id: {storage_manager.run_id}")
    
    # Create and register metrics collector if enabled
    if config.get("metrics", {}).get("enabled", True):
        print("Creating MetricsCollector")
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
        print("Registered MetricsCollector with storage manager")
    
    return storage_manager


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