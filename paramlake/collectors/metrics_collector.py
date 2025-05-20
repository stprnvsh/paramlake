"""
Metrics collector for computing tensor statistics in TensorFlow models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from paramlake.storage.storage_interface import StorageInterface


class MetricsCollector:
    """Computes and collects statistics on tensor data during training."""

    def __init__(
        self,
        storage_manager: StorageInterface,
        enabled_metrics: Optional[List[str]] = None,
        include_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        capture_frequency: int = 1,
    ):
        """
        Initialize metrics collector.
        
        Args:
            storage_manager: Storage manager that implements StorageInterface
            enabled_metrics: List of metrics to compute (defaults to basic metrics)
            include_layers: List of layer name patterns to include
            exclude_layers: List of layer name patterns to exclude
            capture_frequency: Compute metrics every N steps
        """
        self.storage = storage_manager
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.capture_frequency = capture_frequency
        
        # Default metrics if none specified
        if enabled_metrics is None:
            self.enabled_metrics = ["l2", "mean", "var", "max", "min", "sparsity"]
        else:
            self.enabled_metrics = enabled_metrics
        
        # Cache for layers we've already checked
        self._layers_to_capture = {}
        
        # Track number of metrics computed for reporting
        self._metrics_computed = 0
    
    def should_capture_layer(self, layer_name: str) -> bool:
        """
        Determine if metrics should be captured for this layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            True if metrics should be captured, False otherwise
        """
        # Check cache first
        if layer_name in self._layers_to_capture:
            return self._layers_to_capture[layer_name]
            
        # Check if layer is in exclude list
        if self.exclude_layers:
            import fnmatch
            if any(fnmatch.fnmatch(layer_name, pattern) for pattern in self.exclude_layers):
                self._layers_to_capture[layer_name] = False
                return False
        
        # Check if layer is in include list (if provided)
        if self.include_layers:
            import fnmatch
            if not any(fnmatch.fnmatch(layer_name, pattern) for pattern in self.include_layers):
                self._layers_to_capture[layer_name] = False
                return False
        
        # Default to capturing
        self._layers_to_capture[layer_name] = True
        return True
    
    def compute_metrics(
        self,
        tensor_data: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute metrics for a tensor.
        
        Args:
            tensor_data: Tensor data as numpy array
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Compute requested metrics
        if "l2" in self.enabled_metrics:
            try:
                metrics["l2"] = float(np.linalg.norm(tensor_data))
            except Exception as e:
                print(f"Error computing L2 norm: {e}")
                
        if "mean" in self.enabled_metrics:
            try:
                metrics["mean"] = float(np.mean(tensor_data))
            except Exception as e:
                print(f"Error computing mean: {e}")
                
        if "var" in self.enabled_metrics:
            try:
                metrics["var"] = float(np.var(tensor_data))
            except Exception as e:
                print(f"Error computing variance: {e}")
                
        if "max" in self.enabled_metrics:
            try:
                metrics["max"] = float(np.max(tensor_data))
            except Exception as e:
                print(f"Error computing max: {e}")
                
        if "min" in self.enabled_metrics:
            try:
                metrics["min"] = float(np.min(tensor_data))
            except Exception as e:
                print(f"Error computing min: {e}")
                
        if "sparsity" in self.enabled_metrics:
            try:
                metrics["sparsity"] = float(np.mean(tensor_data == 0))
            except Exception as e:
                print(f"Error computing sparsity: {e}")
                
        # Advanced metrics (conditionally computed to avoid expensive operations)
        if "spectral_norm" in self.enabled_metrics and len(tensor_data.shape) == 2:
            try:
                # For 2D tensors (matrices) only
                # Use power iteration for large matrices to estimate largest singular value
                if max(tensor_data.shape) > 1000:
                    # Simple power iteration for spectral norm (largest singular value)
                    v = np.random.rand(tensor_data.shape[1])
                    v = v / np.linalg.norm(v)
                    for _ in range(10):  # 10 iterations usually sufficient
                        v_new = tensor_data.T @ (tensor_data @ v)
                        v = v_new / np.linalg.norm(v_new)
                    metrics["spectral_norm"] = float(np.sqrt(v.T @ tensor_data.T @ tensor_data @ v))
                else:
                    # Direct SVD for smaller matrices
                    metrics["spectral_norm"] = float(np.linalg.svd(tensor_data, compute_uv=False)[0])
            except Exception as e:
                print(f"Error computing spectral norm: {e}")
        
        # Count computed metrics
        self._metrics_computed += len(metrics)
        
        return metrics
        
    def process_tensor(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_name: str,
        tensor_data: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Process a tensor to compute and store metrics.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (weights, gradients, etc.)
            tensor_name: Name of the tensor
            tensor_data: Tensor data as numpy array
            step: Current step
        """
        # Skip if not capturing this layer
        if not self.should_capture_layer(layer_name):
            return
            
        # Skip if not at the right frequency
        if step is not None and step % self.capture_frequency != 0:
            return
        
        # Compute metrics
        metrics = self.compute_metrics(tensor_data)
        
        # Store metrics
        for metric_name, value in metrics.items():
            # Use hierarchical path for metric name
            metric_path = f"{layer_name}/{tensor_type}/{tensor_name}/{metric_name}"
            try:
                self.storage.store_metric(metric_path, value, step)
            except Exception as e:
                print(f"Error storing metric {metric_path}: {e}")
    
    def process_tensor_batch(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_data_pairs: List[Tuple[str, np.ndarray]],
        step: Optional[int] = None,
    ) -> None:
        """
        Process a batch of tensors to compute metrics.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (weights, gradients, etc.)
            tensor_data_pairs: List of (tensor_name, tensor_data) pairs
            step: Current step
        """
        # Skip if not capturing this layer
        if not self.should_capture_layer(layer_name):
            return
            
        # Skip if not at the right frequency
        if step is not None and step % self.capture_frequency != 0:
            return
            
        # Process each tensor
        for tensor_name, tensor_data in tensor_data_pairs:
            self.process_tensor(layer_name, tensor_type, tensor_name, tensor_data, step)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about metrics collection."""
        return {
            "metrics_computed": self._metrics_computed,
        }

    # Method for streaming computation from storage (used in post-processing)
    def compute_metrics_from_array(
        self,
        array: Any,  # zarr.Array
        step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute metrics by streaming data from an array.
        
        Args:
            array: Zarr array
            step: Optional step to compute metrics for
            
        Returns:
            Dictionary of metrics
        """
        # Initialize accumulators for statistics
        count = 0
        sum_values = 0.0
        sum_squares = 0.0
        min_value = float('inf')
        max_value = float('-inf')
        zero_count = 0
        
        if step is not None:
            # Get just the specified step
            data = array[step]
            count = data.size
            sum_values = data.sum()
            sum_squares = (data**2).sum()
            min_value = data.min()
            max_value = data.max()
            zero_count = (data == 0).sum()
        elif hasattr(array, 'iter_chunks'):
            # Stream chunks (for large arrays)
            for chunk_coords in array.iter_chunks():
                chunk = array[chunk_coords]
                count += chunk.size
                sum_values += chunk.sum()
                sum_squares += (chunk**2).sum()
                min_value = min(min_value, chunk.min())
                max_value = max(max_value, chunk.max())
                zero_count += (chunk == 0).sum()
        else:
            # Direct array access (for smaller arrays)
            data = array[:]
            count = data.size
            sum_values = data.sum()
            sum_squares = (data**2).sum()
            min_value = data.min()
            max_value = data.max()
            zero_count = (data == 0).sum()
        
        # Compute metrics
        metrics = {}
        if count > 0:
            metrics["mean"] = sum_values / count
            metrics["var"] = (sum_squares / count) - (metrics["mean"] ** 2)
            metrics["min"] = min_value
            metrics["max"] = max_value
            metrics["l2"] = np.sqrt(sum_squares)
            metrics["sparsity"] = zero_count / count
        
        return metrics 