"""
Factory functions for creating storage managers.
"""

from typing import Optional

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
    
    if storage_type == "icechunk":
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk storage requested but icechunk is not installed. "
                             "Install it with 'pip install icechunk'")
        return IcechunkStorageManager(config)
    else:
        # Default to Zarr
        return ZarrStorageManager(config)


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