"""
Interface for ParamLake storage managers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr


class StorageInterface(ABC):
    """Abstract interface for ParamLake storage managers."""
    
    @abstractmethod
    def create_or_get_layer_group(self, layer_name: str, layer_type: str) -> zarr.Group:
        """
        Create or get a group for a layer.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of the layer
            
        Returns:
            Group for the layer
        """
        pass
    
    @abstractmethod
    def store_tensor(
        self,
        layer_group: zarr.Group,
        tensor_name: str,
        tensor_type: str,
        tensor_data: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Store a tensor in the store.
        
        Args:
            layer_group: Layer group
            tensor_name: Name of the tensor
            tensor_type: Type of tensor (weights, gradients, non_trainable, activations)
            tensor_data: Tensor data
            step: Current step (if None, uses internal counter)
        """
        pass
    
    @abstractmethod
    def store_layer_metadata(
        self,
        layer_group: zarr.Group,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store metadata for a layer.
        
        Args:
            layer_group: Layer group
            metadata: Metadata dictionary
        """
        pass
    
    @abstractmethod
    def store_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Store a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Current step (if None, uses internal counter)
        """
        pass
    
    @abstractmethod
    def increment_step(self) -> None:
        """Increment the current step."""
        pass
    
    @abstractmethod
    def set_step(self, step: int) -> None:
        """Set the current step."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage manager and finalize the dataset."""
        pass 