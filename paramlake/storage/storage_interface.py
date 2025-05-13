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

    @abstractmethod
    def store_optimizer_state(
        self,
        optimizer_weights: List[np.ndarray],
        step: Optional[int] = None,
    ) -> None:
        """
        Store the optimizer's state (weights).

        Args:
            optimizer_weights: List of NumPy arrays representing optimizer state.
            step: Current step.
        """
        pass

    @abstractmethod
    def store_optimizer_config(
        self,
        optimizer_name: str,
        optimizer_config: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Store the optimizer's configuration.

        Args:
            optimizer_name: Name of the optimizer.
            optimizer_config: Dictionary containing optimizer configuration.
            step: Current step (optional, as config is often static for a run).
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        weights_data: List[np.ndarray],
        weights_names: List[str],
        weights_shapes: List[Tuple[int, ...]],
        optimizer_data: List[np.ndarray],
        optimizer_config: Optional[Dict[str, Any]],
        compile_config: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        step: Optional[int] = None,
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            weights_data: List of weight arrays
            weights_names: List of weight names
            weights_shapes: List of weight shapes
            optimizer_data: List of optimizer state arrays
            optimizer_config: Optimizer configuration
            compile_config: Model compilation configuration
            metadata: Checkpoint metadata
            step: Current step (if None, uses internal counter)
            
        Returns:
            Checkpoint ID or snapshot ID
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of the checkpoint
            
        Returns:
            Dictionary with checkpoint data
        """
        pass
    
    @abstractmethod
    def load_checkpoint_by_step(self, step: int) -> Dict[str, Any]:
        """
        Load a checkpoint by step number.
        
        Args:
            step: Step number
            
        Returns:
            Dictionary with checkpoint data
        """
        pass
    
    @abstractmethod
    def load_latest_checkpoint(self) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Returns:
            Dictionary with checkpoint data
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of dictionaries with checkpoint metadata
        """
        pass 