"""
Interface for ParamLake storage managers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr


class StorageInterface(ABC):
    """Abstract interface for ParamLake storage managers."""
    
    def __init__(self):
        """Initialize the storage interface."""
        self.metrics_collector = None
    
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

    def register_metrics_collector(self, metrics_collector: Any) -> None:
        """
        Register a metrics collector to process tensors.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector 

    # --- Git-like functionality methods (non-abstract) ---
    # These are primarily for IcechunkStorageManager and have default no-op implementations

    def commit_model_state(
        self,
        model_parameters: Dict[str, np.ndarray],
        message: str,
        branch_name: str,
        author: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Default: Commits the model's parameters as a new snapshot on the given branch."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def create_branch(self, branch_name: str, from_snapshot_id: Optional[str] = None) -> None:
        """Default: Creates a new branch, optionally from a specific snapshot."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def list_branches(self) -> List[str]:
        """Default: Lists all branches in the repository."""
        return []

    def delete_branch(self, branch_name: str) -> None:
        """Default: Deletes a branch."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def create_tag(self, tag_name: str, snapshot_id: str, message: Optional[str] = None) -> None:
        """Default: Creates a tag pointing to a specific snapshot."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def list_tags(self) -> Dict[str, str]: 
        """Default: Lists all tags and their associated snapshot IDs."""
        return {}

    def delete_tag(self, tag_name: str) -> None:
        """Default: Deletes a tag."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def get_history(self, reference: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Default: Gets the commit history for a given reference (branch, tag, or snapshot_id)."""
        return []

    def get_snapshot_id_for_reference(self, reference: str) -> Optional[str]:
        """Default: Resolves a branch name or tag name to a snapshot ID."""
        return None

    def load_parameters_from_snapshot(self, snapshot_id: str) -> Dict[str, np.ndarray]:
        """Default: Loads all model parameters from a given snapshot ID."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")
    
    def diff_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> List[Dict[str, Any]]:
        """Default: Computes the diff between two snapshots."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def merge_branches(
        self, 
        source_branch: str, 
        target_branch: str, 
        strategy: str = 'manual',
        commit_message: Optional[str] = None
    ) -> Optional[str]:
        """Default: Merges the source branch into the target branch."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.")

    def import_model_from_path(
        self, 
        source_path: str, 
        source_format: str, 
        branch_name: str, 
        commit_message: Optional[str] = None
    ) -> str:
        """Default: Imports a model from an external file into the repository."""
        raise NotImplementedError("Git-like versioning is not available in this storage manager.") 