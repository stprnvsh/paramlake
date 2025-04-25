"""
Zarr model analyzer for ParamLake data.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import functools

import numpy as np
import zarr

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ZarrModelAnalyzer:
    """Analyzer for ParamLake Zarr data."""

    def __init__(self, store_path: str, run_id: Optional[str] = None, lazy_loading: bool = True):
        """
        Initialize the model analyzer.
        
        Args:
            store_path: Path to the Zarr store
            run_id: ID of the run to analyze. If None, uses the most recent run.
            lazy_loading: Whether to use lazy loading for tensor data
        """
        self.store_path = store_path
        self.store = zarr.open(store_path, mode="r")
        
        # Find run ID if not provided
        if run_id is None:
            # Find the latest run
            runs = list(self.store.keys())
            if not runs:
                raise ValueError(f"No runs found in {store_path}")
            
            # Find the most recent run based on timestamp
            latest_run = None
            latest_timestamp = None
            
            for run in runs:
                run_group = self.store[run]
                if "timestamp" in run_group.attrs:
                    timestamp = run_group.attrs["timestamp"]
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_run = run
            
            if latest_run is None:
                # Fall back to the first run
                latest_run = runs[0]
            
            self.run_id = latest_run
        else:
            self.run_id = run_id
            if self.run_id not in self.store:
                raise ValueError(f"Run ID {run_id} not found in {store_path}")
        
        self.run_group = self.store[self.run_id]
        self.layers_group = self.run_group["layers"]
        self.metrics_group = self.run_group["metrics"] if "metrics" in self.run_group else None
        
        # Load config if available
        self.config = None
        if "config" in self.run_group.attrs:
            try:
                self.config = json.loads(self.run_group.attrs["config"])
            except (json.JSONDecodeError, TypeError):
                pass
        
        self.lazy_loading = lazy_loading
        
        # Cache for layer info - will be lazily populated
        self._layer_info_cache = {}
        
        # Cache for tensor data - will be lazily populated if lazy_loading is True
        self._tensor_data_cache = {}
    
    def get_run_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the current run.
        
        Returns:
            Dictionary of run metadata
        """
        metadata = {}
        
        for key, value in self.run_group.attrs.items():
            metadata[key] = value
            
        return metadata
    
    def get_layer_names(self) -> List[str]:
        """Get all layer names in the model."""
        return list(self.layers_group.keys())
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """
        Get information about a layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary of layer information
        """
        # Check cache first
        if layer_name in self._layer_info_cache:
            return self._layer_info_cache[layer_name]
            
        layer_group = self.layers_group[layer_name]
        info = dict(layer_group.attrs)
        
        # Add tensor types and names
        info["tensor_types"] = list(layer_group.keys())
        info["tensors"] = {}
        info["available_tensors"] = {}  # Add for backward compatibility
        
        for tensor_type in layer_group.keys():
            tensor_group = layer_group[tensor_type]
            tensor_names = list(tensor_group.keys())
            info["tensors"][tensor_type] = tensor_names
            info["available_tensors"][tensor_type] = tensor_names  # Add for backward compatibility
            
            # Also add legacy keys for compatibility
            if tensor_type == "weights":
                info["weight_tensors"] = tensor_names
            elif tensor_type == "gradients":
                info["gradient_tensors"] = tensor_names
            elif tensor_type == "non_trainable":
                info["non_trainable_tensors"] = tensor_names
            elif tensor_type == "activations":
                info["activation_tensors"] = tensor_names
        
        # Cache the result
        self._layer_info_cache[layer_name] = info
        
        return info
    
    def _get_tensor_key(self, layer_name: str, tensor_name: str, tensor_type: str) -> str:
        """Generate a unique key for tensor caching."""
        return f"{layer_name}/{tensor_type}/{tensor_name}"
    
    def _lazy_load_tensor(self, layer_name: str, tensor_name: str, tensor_type: str) -> zarr.Array:
        """
        Lazily load a tensor from the Zarr store.
        
        Args:
            layer_name: Name of the layer
            tensor_name: Name of the tensor
            tensor_type: Type of tensor (weights, gradients, non_trainable, activations)
            
        Returns:
            Zarr array containing the tensor data
        """
        key = self._get_tensor_key(layer_name, tensor_name, tensor_type)
        
        # Check if already loaded
        if key in self._tensor_data_cache:
            return self._tensor_data_cache[key]
            
        if layer_name not in self.layers_group:
            raise ValueError(f"Layer {layer_name} not found")
            
        layer_group = self.layers_group[layer_name]
        
        if tensor_type not in layer_group:
            raise ValueError(f"Tensor type {tensor_type} not found in layer {layer_name}")
            
        tensor_group = layer_group[tensor_type]
        
        if tensor_name not in tensor_group:
            raise ValueError(f"Tensor {tensor_name} not found in {layer_name}/{tensor_type}")
            
        # Get tensor array
        tensor_array = tensor_group[tensor_name]
        
        # Cache if not using lazy loading
        if not self.lazy_loading:
            self._tensor_data_cache[key] = tensor_array[:]
            return self._tensor_data_cache[key]
        
        # Return the zarr array directly for lazy loading
        return tensor_array
    
    def get_tensor_data(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_name: str,
        step: Optional[Union[int, slice]] = None,
    ) -> np.ndarray:
        """
        Get tensor data for a specific layer, type, and name.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (e.g., 'weights', 'gradients')
            tensor_name: Name of the tensor (e.g., 'kernel', 'bias')
            step: Step or slice to retrieve (None for all steps)
            
        Returns:
            Numpy array of tensor data
        """
        tensor_array = self._lazy_load_tensor(layer_name, tensor_name, tensor_type)
        
        if step is None:
            return tensor_array[:]
        else:
            return tensor_array[step]
    
    def get_layer_stats(
        self,
        layer_name: str,
        tensor_type: str = "weights",
        tensor_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for a layer's tensor over time.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (e.g., 'weights', 'gradients')
            tensor_name: Name of the tensor (e.g., 'kernel', 'bias')
                         If None, computes stats for all tensors in the layer
            
        Returns:
            Dictionary of statistics
        """
        layer_group = self.layers_group[layer_name]
        tensor_group = layer_group[tensor_type]
        
        if tensor_name is None:
            # Compute stats for all tensors in the layer/type
            all_stats = {}
            for name in tensor_group.keys():
                all_stats[name] = self._compute_tensor_stats(
                    tensor_group[name]
                )
            return all_stats
        else:
            tensor_array = tensor_group[tensor_name]
            return self._compute_tensor_stats(tensor_array)
    
    def _compute_tensor_stats(self, tensor_array: zarr.Array) -> Dict[str, np.ndarray]:
        """
        Compute statistics for a tensor array over time.
        
        Args:
            tensor_array: Zarr array with time as first dimension
            
        Returns:
            Dictionary of statistics (mean, std, min, max, norm)
        """
        # Get number of steps
        n_steps = tensor_array.shape[0]
        
        # Initialize stats arrays
        mean = np.zeros(n_steps)
        std = np.zeros(n_steps)
        min_val = np.zeros(n_steps)
        max_val = np.zeros(n_steps)
        l2_norm = np.zeros(n_steps)
        
        # Compute stats for each step
        for i in range(n_steps):
            tensor = tensor_array[i]
            mean[i] = np.mean(tensor)
            std[i] = np.std(tensor)
            min_val[i] = np.min(tensor)
            max_val[i] = np.max(tensor)
            l2_norm[i] = np.sqrt(np.sum(tensor**2))
        
        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "l2_norm": l2_norm,
        }
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get metrics data.
        
        Args:
            metric_name: Name of the metric to retrieve.
                        If None, returns all metrics.
            
        Returns:
            Dictionary of metrics or a single metric array
        """
        if self.metrics_group is None:
            return {} if metric_name is None else np.array([])
        
        if metric_name is None:
            metrics = {}
            for name in self.metrics_group.keys():
                metrics[name] = self.metrics_group[name][:]
            return metrics
        else:
            if metric_name not in self.metrics_group:
                raise KeyError(f"Metric {metric_name} not found")
            return self.metrics_group[metric_name][:]
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model summary information
        """
        summary = {
            "run_id": self.run_id,
            "framework": self.run_group.attrs.get("framework", "unknown"),
            "framework_version": self.run_group.attrs.get("framework_version", "unknown"),
            "paramlake_version": self.run_group.attrs.get("paramlake_version", "unknown"),
            "timestamp": self.run_group.attrs.get("timestamp", "unknown"),
            "steps": self.run_group.attrs.get("final_step", 0) + 1,
            "layers": {}
        }
        
        # Add layer information
        layer_names = self.get_layer_names()
        summary["total_layers"] = len(layer_names)
        
        for layer_name in layer_names:
            layer_info = self.get_layer_info(layer_name)
            summary["layers"][layer_name] = {
                "type": layer_info.get("type", "unknown"),
                "tensor_types": layer_info.get("tensor_types", []),
            }
        
        # Add metrics if available
        if self.metrics_group is not None:
            summary["metrics"] = list(self.metrics_group.keys())
        
        return summary
    
    def compare_runs(
        self,
        other_path: str,
        other_run_id: Optional[str] = None,
        layers: Optional[List[str]] = None,
        tensor_type: str = "weights",
        stat: str = "norm",
        steps: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compare tensor statistics between two runs.
        
        Args:
            other_path: Path to the other Zarr store
            other_run_id: ID of the other run (if None, uses the latest run)
            layers: List of layers to compare (if None, compares all common layers)
            tensor_type: Type of tensor to compare
            stat: Statistic to compare (mean, std, min, max, norm)
            steps: List of steps to compare (if None, uses all steps)
            
        Returns:
            Dictionary mapping layer names to dictionaries mapping tensor names to
            tuples of (this_run_stat, other_run_stat)
        """
        # Load other run
        other_analyzer = ZarrModelAnalyzer(other_path, other_run_id, self.lazy_loading)
        
        # Get layers to compare
        if layers is None:
            # Get common layers
            this_layers = set(self.get_layer_names())
            other_layers = set(other_analyzer.get_layer_names())
            layers = list(this_layers.intersection(other_layers))
            
        # Compare each layer
        results = {}
        for layer_name in layers:
            try:
                # Get layer info for both runs
                this_info = self.get_layer_info(layer_name)
                other_info = other_analyzer.get_layer_info(layer_name)
                
                # Get tensor list based on tensor type
                if tensor_type == "weights":
                    this_tensors = this_info.get("weight_tensors", [])
                    other_tensors = other_info.get("weight_tensors", [])
                elif tensor_type == "gradients":
                    this_tensors = this_info.get("gradient_tensors", [])
                    other_tensors = other_info.get("gradient_tensors", [])
                elif tensor_type == "non_trainable":
                    this_tensors = this_info.get("non_trainable_tensors", [])
                    other_tensors = other_info.get("non_trainable_tensors", [])
                else:
                    this_tensors = []
                    other_tensors = []
                    
                # Get common tensors
                common_tensors = set(this_tensors).intersection(set(other_tensors))
                
                # Compare each tensor
                layer_results = {}
                for tensor_name in common_tensors:
                    # Get stats for both runs
                    this_stats = self.get_tensor_stats(layer_name, tensor_name, tensor_type, steps)
                    other_stats = other_analyzer.get_tensor_stats(layer_name, tensor_name, tensor_type, steps)
                    
                    if stat in this_stats and stat in other_stats:
                        layer_results[tensor_name] = (this_stats[stat], other_stats[stat])
                        
                if layer_results:
                    results[layer_name] = layer_results
            except Exception as e:
                print(f"Error comparing layer {layer_name}: {e}")
                
        return results
    
    def plot_weight_evolution(
        self,
        layer_name: str,
        tensor_name: Optional[str] = None,
        tensor_type: str = "weights",
    ) -> None:
        """
        Plot the evolution of weights over time.
        
        Args:
            layer_name: Name of the layer
            tensor_name: Name of the tensor (None for all tensors in the layer)
            tensor_type: Type of tensor (default: 'weights')
            
        Returns:
            None (displays plot)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
        
        stats = self.get_layer_stats(layer_name, tensor_type, tensor_name)
        
        if tensor_name is None:
            # Plot all tensors
            fig, axes = plt.subplots(len(stats), 4, figsize=(15, 4 * len(stats)))
            
            for i, (name, tensor_stats) in enumerate(stats.items()):
                if len(stats) == 1:
                    row_axes = axes
                else:
                    row_axes = axes[i]
                
                # Plot mean
                row_axes[0].plot(tensor_stats["mean"])
                row_axes[0].set_title(f"{name} - Mean")
                row_axes[0].grid(True)
                
                # Plot std
                row_axes[1].plot(tensor_stats["std"])
                row_axes[1].set_title(f"{name} - Std")
                row_axes[1].grid(True)
                
                # Plot min/max
                row_axes[2].plot(tensor_stats["min"], label="Min")
                row_axes[2].plot(tensor_stats["max"], label="Max")
                row_axes[2].set_title(f"{name} - Min/Max")
                row_axes[2].legend()
                row_axes[2].grid(True)
                
                # Plot L2 norm
                row_axes[3].plot(tensor_stats["l2_norm"])
                row_axes[3].set_title(f"{name} - L2 Norm")
                row_axes[3].grid(True)
            
            plt.tight_layout()
            plt.suptitle(f"Layer: {layer_name} - Type: {tensor_type}", fontsize=16)
            plt.subplots_adjust(top=0.95)
            plt.show()
        else:
            # Plot a single tensor
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot mean
            axes[0, 0].plot(stats["mean"])
            axes[0, 0].set_title("Mean")
            axes[0, 0].grid(True)
            
            # Plot std
            axes[0, 1].plot(stats["std"])
            axes[0, 1].set_title("Standard Deviation")
            axes[0, 1].grid(True)
            
            # Plot min/max
            axes[1, 0].plot(stats["min"], label="Min")
            axes[1, 0].plot(stats["max"], label="Max")
            axes[1, 0].set_title("Min/Max Values")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot L2 norm
            axes[1, 1].plot(stats["l2_norm"])
            axes[1, 1].set_title("L2 Norm")
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.suptitle(f"Layer: {layer_name} - Tensor: {tensor_name} - Type: {tensor_type}", fontsize=16)
            plt.subplots_adjust(top=0.9)
            plt.show()
    
    def plot_gradient_norm_by_layer(self, layers: Optional[List[str]] = None, steps: Optional[Union[int, slice]] = None) -> None:
        """
        Plot the L2 norm of gradients for each layer over time.
        
        Args:
            layers: List of layers to plot (if None, plots all layers with gradients)
            steps: Specific training steps to include (if None, includes all steps)
            
        Returns:
            None (displays plot)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
            
        # Get all layers or specified layers
        if layers is None:
            layers = self.get_layer_names()
        
        # Filter to layers that have gradients
        gradient_layers = []
        for layer_name in layers:
            layer_info = self.get_layer_info(layer_name)
            if "gradients" in layer_info.get("tensor_types", []):
                gradient_layers.append(layer_name)
        
        if not gradient_layers:
            print("No gradient data found in the selected layers.")
            return
            
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot gradient norm for each layer
        for layer_name in gradient_layers:
            # Get all gradient tensors for this layer
            layer_info = self.get_layer_info(layer_name)
            tensors = layer_info.get("tensors", {}).get("gradients", [])
            
            # Calculate combined L2 norm for all tensors in the layer
            combined_norm = None
            
            for tensor_name in tensors:
                # Get gradient data
                try:
                    grad_data = self.get_tensor_data(layer_name, "gradients", tensor_name, steps)
                    
                    # Convert to numpy if it's a zarr array
                    if not isinstance(grad_data, np.ndarray):
                        grad_data = np.array(grad_data)
                    
                    # Calculate L2 norm across all dimensions except the first (time/step)
                    flattened = grad_data.reshape(grad_data.shape[0], -1)
                    norm = np.linalg.norm(flattened, axis=1)
                    
                    # Combine with other tensors in the layer
                    if combined_norm is None:
                        combined_norm = norm**2
                    else:
                        # Pad if necessary for different lengths
                        if len(norm) > len(combined_norm):
                            combined_norm = np.pad(combined_norm, (0, len(norm) - len(combined_norm)))
                        elif len(norm) < len(combined_norm):
                            norm = np.pad(norm, (0, len(combined_norm) - len(norm)))
                        combined_norm += norm**2
                except Exception as e:
                    print(f"Error processing gradients for {layer_name}/{tensor_name}: {e}")
            
            if combined_norm is not None:
                # Take square root for final combined norm
                combined_norm = np.sqrt(combined_norm)
                plt.plot(combined_norm, label=f"{layer_name}")
        
        plt.xlabel("Step")
        plt.ylabel("Gradient L2 Norm")
        plt.title("Gradient Norm by Layer")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def analyze_gradient_statistics(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze gradient statistics across the model or for a specific layer.
        
        Args:
            layer_name: Name of layer to analyze (if None, analyzes all layers)
            
        Returns:
            Dictionary with gradient statistics
        """
        # Get layer names to analyze
        if layer_name is not None:
            if layer_name not in self.layers_group:
                raise ValueError(f"Layer {layer_name} not found")
            layers = [layer_name]
        else:
            layers = self.get_layer_names()
            
        stats = {}
        total_layers_with_gradients = 0
        
        # Analyze each layer
        for name in layers:
            layer_info = self.get_layer_info(name)
            
            # Check if layer has gradients
            if "gradients" not in layer_info.get("tensor_types", []):
                continue
                
            total_layers_with_gradients += 1
            layer_stats = {}
            
            # Get all gradient tensors for this layer
            tensors = layer_info.get("tensors", {}).get("gradients", [])
            
            for tensor_name in tensors:
                try:
                    # Get gradient data
                    grad_data = self.get_tensor_data(name, "gradients", tensor_name)
                    
                    # Calculate statistics
                    tensor_stats = {
                        "mean_abs": np.mean(np.abs(grad_data), axis=0).mean(),
                        "mean": np.mean(grad_data),
                        "std": np.std(grad_data),
                        "min": np.min(grad_data),
                        "max": np.max(grad_data),
                        "zero_fraction": np.mean(grad_data == 0.0),
                        "shape": grad_data.shape,
                    }
                    
                    # Add norm across time
                    norms = np.linalg.norm(grad_data.reshape(grad_data.shape[0], -1), axis=1)
                    tensor_stats["norm_mean"] = np.mean(norms)
                    tensor_stats["norm_std"] = np.std(norms)
                    tensor_stats["norm_min"] = np.min(norms)
                    tensor_stats["norm_max"] = np.max(norms)
                    
                    layer_stats[tensor_name] = tensor_stats
                except Exception as e:
                    print(f"Error analyzing gradients for {name}/{tensor_name}: {e}")
            
            if layer_stats:
                stats[name] = layer_stats
        
        # Add summary information
        summary = {
            "total_layers": len(layers),
            "layers_with_gradients": total_layers_with_gradients,
            "gradient_coverage": total_layers_with_gradients / len(layers) if layers else 0
        }
        
        return {
            "summary": summary,
            "layer_stats": stats
        }
    
    def close(self) -> None:
        """Close the Zarr store and release resources."""
        # Zarr doesn't require explicit closing, but we'll clear references
        self.store = None
        self.run_group = None
        self.layers_group = None
        self.metrics_group = None
        self.clear_cache()
    
    def clear_cache(self) -> None:
        """Clear the tensor data cache to free memory."""
        self._tensor_data_cache.clear()
    
    # Lazy loading decorator for methods that access tensor data
    @staticmethod
    def _with_lazy_loading(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            
            # Clear cache after operation if using lazy loading
            if self.lazy_loading:
                self.clear_cache()
                
            return result
        return wrapper 