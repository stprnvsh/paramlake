"""
Analyzer for ParamLake data stored in Icechunk.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union
import os

import numpy as np
import zarr

try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class IcechunkModelAnalyzer:
    """Analyzer for ParamLake data stored in Icechunk."""

    def __init__(
        self, 
        repo_path: Union[str, Dict[str, Any]], 
        snapshot_id: Optional[str] = None, 
        branch: str = "main",
        lazy_loading: bool = True
    ):
        """
        Initialize the model analyzer.
        
        Args:
            repo_path: Path or storage config for the Icechunk repo
            snapshot_id: Specific snapshot to analyze. If None, uses the latest on the branch.
            branch: Branch name to use if snapshot_id is None
            lazy_loading: Whether to use lazy loading for tensor data
        """
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk is required but not installed. Install it with 'pip install icechunk'.")
            
        # Configure storage
        if isinstance(repo_path, dict):
            # Configure storage from dict
            storage_config = repo_path
            storage_type = storage_config.get("type", "s3")
            
            if storage_type == "s3":
                # Setup environment variables if needed
                if "endpoint_url" in storage_config and not os.environ.get('AWS_S3_ENDPOINT'):
                    os.environ['AWS_S3_ENDPOINT'] = storage_config.get("endpoint_url", "https://s3.amazonaws.com")
                
                storage = icechunk.s3_storage(
                    bucket=storage_config.get("bucket"),
                    prefix=storage_config.get("prefix", ""),
                    region=storage_config.get("region", "us-east-1"),
                    endpoint_url=storage_config.get("endpoint_url", "https://s3.amazonaws.com"),
                    from_env=True
                )
            elif storage_type == "gcs":
                storage = icechunk.gcs_storage(
                    bucket=storage_config.get("bucket"),
                    prefix=storage_config.get("prefix", ""),
                    from_env=True
                )
            elif storage_type == "azure":
                storage = icechunk.azure_storage(
                    account=storage_config.get("account"),
                    container=storage_config.get("container"),
                    prefix=storage_config.get("prefix", ""),
                    from_env=True
                )
            elif storage_type == "local":
                storage = icechunk.local_filesystem_storage(storage_config.get("path"))
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")
        else:
            # Simple local path
            storage = icechunk.local_filesystem_storage(repo_path)
            
        # Create repository configuration
        config = icechunk.RepositoryConfig.default()
        
        # Configure storage settings
        config.storage = icechunk.StorageSettings(
            concurrency=icechunk.StorageConcurrencySettings(
                max_concurrent_requests_for_object=10,
                ideal_concurrent_request_size=1000000,
            ),
            storage_class="STANDARD",
            metadata_storage_class="STANDARD",
            chunks_storage_class="STANDARD",
        )
        
        # Configure compression
        config.compression = icechunk.CompressionConfig(
            level=3,
            algorithm=icechunk.CompressionAlgorithm.Zstd,
        )
        
        # Configure caching for better performance
        config.caching = icechunk.CachingConfig(
            num_snapshot_nodes=100,
            num_chunk_refs=100,
            num_transaction_changes=100,
            num_bytes_attributes=10000,
            num_bytes_chunks=1000000,
        )
        
        # Open repository with configuration
        try:
            self.repo = icechunk.Repository.open(storage, config=config)
        except Exception as e:
            raise ValueError(f"Error opening repository: {e}")
        
        # Get latest snapshot ID if not provided
        if snapshot_id is None:
            # Try to get the latest snapshot on the specified branch
            try:
                snapshot_id = self.repo.lookup_branch(branch)
            except Exception as e:
                print(f"Error getting branch {branch}: {e}")
                # Try to get the most recent snapshot
                try:
                    # Get initial snapshot as fallback
                    snapshot_id = self.repo.initial_snapshot().id
                    print(f"Using initial snapshot: {snapshot_id}")
                except Exception as e2:
                    raise ValueError(f"Error getting initial snapshot: {e2}")
        
        # Set up read session
        try:
            self.session = self.repo.readonly_session(snapshot_id=snapshot_id)
            self.store = self.session.store
            self.snapshot_id = snapshot_id
        except Exception as e:
            raise ValueError(f"Error creating readonly session: {e}")
        
        # Open zarr groups
        try:
            self.root_group = zarr.open_group(self.store, mode="r")
        except Exception as e:
            raise ValueError(f"Error opening root group: {e}")
        
        # Open layers group if it exists
        if "layers" in self.root_group:
            self.layers_group = self.root_group["layers"]
        else:
            print("Warning: No layers found in the repository")
            self.layers_group = None
            
        # Open metrics group if it exists
        if "metrics" in self.root_group:
            self.metrics_group = self.root_group["metrics"]
        else:
            self.metrics_group = None
            
        # Load metadata
        self.metadata = dict(self.root_group.attrs)
        
        # Try to parse config
        self.config = None
        if "config" in self.metadata:
            try:
                self.config = json.loads(self.metadata["config"])
            except Exception:
                pass
                
        # Initialize caching
        self.lazy_loading = lazy_loading
        self._layer_info_cache = {}
        self._tensor_data_cache = {}
        
    def get_run_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for the current run.
        
        Returns:
            Dictionary of run metadata
        """
        return self.metadata.copy()
    
    def get_layer_names(self) -> List[str]:
        """Get all layer names in the model."""
        if self.layers_group is None:
            return []
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
            
        if self.layers_group is None:
            return {}
            
        if layer_name not in self.layers_group:
            raise ValueError(f"Layer {layer_name} not found")
            
        layer_group = self.layers_group[layer_name]
        info = dict(layer_group.attrs)
        
        # Add tensor types and names
        tensor_types = list(layer_group.keys())
        
        # CRITICAL FIX: Explicitly check for has_gradients attribute
        if "has_gradients" in layer_group.attrs and layer_group.attrs["has_gradients"]:
            if "gradients" not in tensor_types:
                # Try to see if there's a hidden gradients group
                try:
                    if hasattr(layer_group, 'gradients') or (hasattr(layer_group, '__contains__') and 'gradients' in layer_group):
                        tensor_types.append("gradients")
                except Exception:
                    pass
        
        info["tensor_types"] = tensor_types
        info["tensors"] = {}
        
        for tensor_type in tensor_types:
            try:
                tensor_group = layer_group[tensor_type]
                tensor_names = list(tensor_group.keys())
                info["tensors"][tensor_type] = tensor_names
                
                # Check for actual tensor data to verify existence
                if tensor_type == "gradients" and tensor_names:
                    # Verify at least one gradient tensor actually has data
                    has_valid_gradients = False
                    for tensor_name in tensor_names:
                        try:
                            tensor_array = tensor_group[tensor_name]
                            if tensor_array.shape[0] > 0:  # Has at least one time step
                                has_valid_gradients = True
                                break
                        except Exception:
                            continue
                    
                    if not has_valid_gradients:
                        info["tensors"][tensor_type] = []  # Mark as empty if no valid data
                
            except Exception as e:
                print(f"Error getting tensor names for {layer_name}/{tensor_type}: {e}")
                info["tensors"][tensor_type] = []
        
        # Cache the result
        self._layer_info_cache[layer_name] = info
        
        return info
    
    def _get_tensor_key(self, layer_name: str, tensor_name: str, tensor_type: str) -> str:
        """Generate a unique key for tensor caching."""
        return f"{layer_name}/{tensor_type}/{tensor_name}"
    
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
        if self.layers_group is None:
            raise ValueError("No layers group found")
            
        if layer_name not in self.layers_group:
            raise ValueError(f"Layer {layer_name} not found")
            
        layer_group = self.layers_group[layer_name]
        
        if tensor_type not in layer_group:
            raise ValueError(f"Tensor type {tensor_type} not found in layer {layer_name}")
            
        tensor_group = layer_group[tensor_type]
        
        if tensor_name not in tensor_group:
            raise ValueError(f"Tensor {tensor_name} not found in {layer_name}/{tensor_type}")
            
        # Get tensor array
        try:
            tensor_array = tensor_group[tensor_name]
            
            # Return data for the specified step(s)
            if step is None:
                return tensor_array[:]
            else:
                return tensor_array[step]
        except Exception as e:
            print(f"Warning: Error accessing tensor data for {layer_name}/{tensor_type}/{tensor_name}: {e}")
            print("This might be due to data fragmentation across multiple snapshots")
            
            # Try to read tensor data from the latest available snapshot
            # This can happen if arrays were recreated after commits
            try:
                # Get other snapshots in chronological order (oldest to newest)
                training_history = self.get_training_history()
                
                # For each snapshot (starting from the current one)
                for snapshot in reversed(training_history):
                    snapshot_id = snapshot["id"]
                    # Skip current snapshot since we already tried it
                    if snapshot_id == self.snapshot_id:
                        continue
                    
                    # Try to open a temporary session with this snapshot
                    try:
                        temp_session = self.repo.readonly_session(snapshot_id=snapshot_id)
                        temp_store = temp_session.store
                        
                        # Try to open the tensor data from this snapshot
                        temp_root = zarr.open_group(temp_store, mode="r")
                        if "layers" in temp_root and layer_name in temp_root["layers"]:
                            temp_layer = temp_root["layers"][layer_name]
                            if tensor_type in temp_layer and tensor_name in temp_layer[tensor_type]:
                                temp_tensor = temp_layer[tensor_type][tensor_name]
                                
                                # Return data for the specified step(s)
                                try:
                                    if step is None:
                                        return temp_tensor[:]
                                    else:
                                        return temp_tensor[step]
                                except Exception:
                                    # Continue to next snapshot if this one doesn't have the data
                                    continue
                    except Exception:
                        # Continue to next snapshot if we can't open this one
                        continue
                
                # If we get here, we couldn't find the data in any snapshot
                raise ValueError(f"Could not find tensor data for {layer_name}/{tensor_type}/{tensor_name} in any snapshot")
                
            except Exception as e2:
                # If all else fails, re-raise the original error
                raise ValueError(f"Failed to access tensor data: {e}") from e
    
    def get_layer_stats(
        self,
        layer_name: str,
        tensor_type: str = "weights",
        tensor_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute statistics for a layer's tensor(s) over time.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (e.g., 'weights', 'gradients')
            tensor_name: Name of the tensor (or None for all tensors)
            
        Returns:
            Dictionary of statistics
        """
        if self.layers_group is None:
            raise ValueError("No layers group found")
            
        if layer_name not in self.layers_group:
            raise ValueError(f"Layer {layer_name} not found")
            
        layer_group = self.layers_group[layer_name]
        
        if tensor_type not in layer_group:
            raise ValueError(f"Tensor type {tensor_type} not found in layer {layer_name}")
            
        tensor_group = layer_group[tensor_type]
        
        stats = {}
        
        if tensor_name is not None:
            # Compute stats for a specific tensor
            if tensor_name not in tensor_group:
                raise ValueError(f"Tensor {tensor_name} not found in {layer_name}/{tensor_type}")
                
            tensor_array = tensor_group[tensor_name]
            stats[tensor_name] = self._compute_tensor_stats(tensor_array)
        else:
            # Compute stats for all tensors
            for name in tensor_group.keys():
                tensor_array = tensor_group[name]
                stats[name] = self._compute_tensor_stats(tensor_array)
                
        return stats
    
    def _compute_tensor_stats(self, tensor_array: zarr.Array) -> Dict[str, np.ndarray]:
        """
        Compute statistics for a tensor over time.
        
        Args:
            tensor_array: Zarr array containing tensor data
            
        Returns:
            Dictionary of statistics (min, max, mean, etc.)
        """
        # Get the tensor data
        data = tensor_array[:]
        
        # Calculate basic statistics
        stats = {}
        
        # For 1D tensors (like timesteps only), just return the raw data
        if len(data.shape) == 1:
            stats["values"] = data
            return stats
            
        # For tensors with time dimension, compute stats over time
        stats["min"] = np.min(data, axis=tuple(range(1, len(data.shape))))
        stats["max"] = np.max(data, axis=tuple(range(1, len(data.shape))))
        stats["mean"] = np.mean(data, axis=tuple(range(1, len(data.shape))))
        stats["std"] = np.std(data, axis=tuple(range(1, len(data.shape))))
        stats["norm"] = np.linalg.norm(data.reshape(data.shape[0], -1), axis=1)
        
        return stats
    
    def get_metrics(
        self,
        metric_path: Optional[str] = None,
        steps: Optional[Union[int, slice]] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get metrics data.
        
        Args:
            metric_path: Path to the metric(s) to retrieve.
                        Can be a full path like 'layer_name/tensor_type/tensor_name/metric_name'
                        or None for all metrics.
            steps: Step or slice to retrieve (None for all steps)
            
        Returns:
            Metric data as array or dictionary of arrays
        """
        if self.metrics_group is None:
            if metric_path is None:
                return {}
            else:
                return np.array([])
            
        if metric_path is None:
            # Return all metrics (flattened dictionary with full paths as keys)
            metrics = {}
            # Collect all metrics using a recursive helper function
            self._collect_metrics_recursively(self.metrics_group, "", metrics)
            return metrics
        else:
            # Handle the case of a specific metric path
            try:
                # Split the path into components
                path_components = metric_path.split("/")
                current_group = self.metrics_group
                
                # Navigate through the path components
                for component in path_components:
                    if component in current_group:
                        current_group = current_group[component]
                    else:
                        # Path component not found
                        return np.array([])
                
                # Check if we reached a leaf node (actual metric array)
                if isinstance(current_group, zarr.Array):
                    if steps is None:
                        return current_group[:]
                    else:
                        return current_group[steps]
                else:
                    # We're still at a group, not a leaf metric
                    print(f"Warning: Path '{metric_path}' points to a group, not a metric array")
                    return np.array([])
                    
            except Exception as e:
                print(f"Warning: Could not read metric {metric_path}: {e}")
                return np.array([])
    
    def _collect_metrics_recursively(
        self,
        group: zarr.Group,
        current_path: str,
        result_dict: Dict[str, np.ndarray]
    ) -> None:
        """
        Recursively collect metrics from nested groups.
        
        Args:
            group: Current Zarr group
            current_path: Current path in the metrics hierarchy
            result_dict: Dictionary to store results in
        """
        for name in group.keys():
            full_path = f"{current_path}/{name}" if current_path else name
            try:
                item = group[name]
                if isinstance(item, zarr.Array):
                    # This is a metric array
                    result_dict[full_path] = item[:]
                elif hasattr(item, 'keys'):  # Check if it's a group-like object
                    # This is a subgroup, recurse into it
                    self._collect_metrics_recursively(item, full_path, result_dict)
            except Exception as e:
                print(f"Warning: Could not read metric {full_path}: {e}")
                result_dict[full_path] = np.array([])
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get the training history from the Icechunk repository.
        
        Returns:
            List of snapshots with metadata
        """
        # Get the history of snapshots
        try:
            ancestry = self.repo.ancestry(snapshot_id=self.snapshot_id)
            
            snapshots = []
            for ancestor in ancestry:
                snapshots.append({
                    "id": ancestor.id,
                    "message": ancestor.message,
                    "written_at": ancestor.written_at,
                })
                
            return snapshots
        except Exception as e:
            print(f"Error getting ancestry: {e}")
            return []
    
    def plot_weight_evolution(
        self,
        layer_name: str,
        tensor_name: Optional[str] = None,
        tensor_type: str = "weights",
        stat: str = "norm",
        title: Optional[str] = None,
    ) -> None:
        """
        Plot the evolution of a tensor statistic over time.
        
        Args:
            layer_name: Name of the layer
            tensor_name: Name of the tensor (or None to plot all tensors)
            tensor_type: Type of tensor (e.g., 'weights', 'gradients')
            stat: Statistic to plot (norm, mean, min, max, std)
            title: Plot title (or None for auto-generated)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting but not installed.")
            
        stats = self.get_layer_stats(layer_name, tensor_type, tensor_name)
        
        plt.figure(figsize=(10, 6))
        
        if tensor_name is not None:
            if stat not in stats[tensor_name]:
                raise ValueError(f"Statistic {stat} not available. Available stats: {list(stats[tensor_name].keys())}")
                
            values = stats[tensor_name][stat]
            plt.plot(values, label=f"{layer_name}/{tensor_name} {stat}")
        else:
            for name, tensor_stats in stats.items():
                if stat not in tensor_stats:
                    continue
                    
                values = tensor_stats[stat]
                plt.plot(values, label=f"{layer_name}/{name} {stat}")
                
        plt.xlabel("Step")
        plt.ylabel(stat.capitalize())
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{stat.capitalize()} Evolution for {layer_name} {tensor_type}")
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_metric(
        self,
        metric_name: str,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot a training metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            title: Plot title (or None for auto-generated)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting but not installed.")
            
        metric_data = self.get_metrics(metric_name)
        
        plt.figure(figsize=(10, 6))
        plt.plot(metric_data)
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        
        if title:
            plt.title(title)
        else:
            plt.title(f"{metric_name} over time")
            
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_snapshots(
        self,
        other_snapshot_id: str,
        layer_name: str,
        tensor_type: str = "weights",
        tensor_name: Optional[str] = None,
        stat: str = "norm",
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compare tensor statistics between two snapshots.
        
        Args:
            other_snapshot_id: ID of the other snapshot to compare with
            layer_name: Name of the layer to compare
            tensor_type: Type of tensor to compare
            tensor_name: Name of the tensor to compare (or None for all)
            stat: Statistic to compare (norm, mean, min, max, std)
            
        Returns:
            Dictionary of comparison results
        """
        # Current snapshot stats
        current_stats = self.get_layer_stats(layer_name, tensor_type, tensor_name)
        
        # Create analyzer for the other snapshot
        other_analyzer = IcechunkModelAnalyzer(
            self.repo, 
            snapshot_id=other_snapshot_id,
            lazy_loading=self.lazy_loading
        )
        
        # Get stats from the other snapshot
        other_stats = other_analyzer.get_layer_stats(layer_name, tensor_type, tensor_name)
        
        # Compare statistics
        results = {}
        
        if tensor_name is not None:
            # Compare specific tensor
            current_tensor_stats = current_stats[tensor_name]
            other_tensor_stats = other_stats[tensor_name]
            
            if stat in current_tensor_stats and stat in other_tensor_stats:
                results[tensor_name] = {
                    stat: (current_tensor_stats[stat], other_tensor_stats[stat])
                }
        else:
            # Compare all tensors
            for name in current_stats.keys():
                if name in other_stats:
                    current_tensor_stats = current_stats[name]
                    other_tensor_stats = other_stats[name]
                    
                    if stat in current_tensor_stats and stat in other_tensor_stats:
                        results[name] = {
                            stat: (current_tensor_stats[stat], other_tensor_stats[stat])
                        }
                        
        return results
    
    def plot_snapshot_comparison(
        self,
        other_snapshot_id: str,
        layer_name: str,
        tensor_type: str = "weights",
        tensor_name: Optional[str] = None,
        stat: str = "norm",
        title: Optional[str] = None,
    ) -> None:
        """
        Plot a comparison of tensor statistics between two snapshots.
        
        Args:
            other_snapshot_id: ID of the other snapshot to compare with
            layer_name: Name of the layer to compare
            tensor_type: Type of tensor to compare
            tensor_name: Name of the tensor to compare (or None for all)
            stat: Statistic to compare (norm, mean, min, max, std)
            title: Plot title (or None for auto-generated)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting but not installed.")
            
        # Get comparison data
        comparison = self.compare_snapshots(
            other_snapshot_id,
            layer_name,
            tensor_type,
            tensor_name,
            stat
        )
        
        plt.figure(figsize=(12, 6))
        
        if tensor_name is not None:
            # Plot specific tensor
            current_values, other_values = comparison[tensor_name][stat]
            
            # Plot current snapshot values
            plt.subplot(1, 2, 1)
            plt.plot(current_values)
            plt.title(f"Current ({self.snapshot_id[:8]}...)")
            plt.xlabel("Step")
            plt.ylabel(stat.capitalize())
            plt.grid(True, alpha=0.3)
            
            # Plot other snapshot values
            plt.subplot(1, 2, 2)
            plt.plot(other_values)
            plt.title(f"Other ({other_snapshot_id[:8]}...)")
            plt.xlabel("Step")
            plt.ylabel(stat.capitalize())
            plt.grid(True, alpha=0.3)
        else:
            # Plot all tensors
            plt.subplot(1, 2, 1)
            for name, stats_dict in comparison.items():
                current_values, _ = stats_dict[stat]
                plt.plot(current_values, label=name)
            plt.title(f"Current ({self.snapshot_id[:8]}...)")
            plt.xlabel("Step")
            plt.ylabel(stat.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            for name, stats_dict in comparison.items():
                _, other_values = stats_dict[stat]
                plt.plot(other_values, label=name)
            plt.title(f"Other ({other_snapshot_id[:8]}...)")
            plt.xlabel("Step")
            plt.ylabel(stat.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        if title:
            plt.suptitle(title, fontsize=14)
        else:
            plt.suptitle(f"Comparison of {layer_name} {tensor_type} {stat}", fontsize=14)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    
    def close(self) -> None:
        """Close the analyzer and release resources."""
        self._tensor_data_cache.clear()
        self._layer_info_cache.clear()
    
    def plot_gradient_norm_by_layer(self, layers: Optional[List[str]] = None, steps: Optional[Union[int, slice]] = None) -> None:
        """
        Plot the L2 norm of gradients for each layer over time.
        
        Args:
            layers: List of layers to plot (if None, plots all layers with gradients)
            steps: Specific training steps to include (if None, includes all steps)
            
        Returns:
            None (displays plot)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting but not installed. Install it with 'pip install matplotlib'.")
            
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
        plt.grid(True, alpha=0.3)
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
            if self.layers_group is None:
                raise ValueError("No layers group found")
            if layer_name not in self.layers_group:
                raise ValueError(f"Layer {layer_name} not found")
            layers = [layer_name]
        else:
            layers = self.get_layer_names()
            
        stats = {}
        total_layers = len(layers)
        total_layers_with_gradients = 0
        total_gradient_tensors = 0
        errors_encountered = []
        
        print(f"Analyzing gradients for {total_layers} layers...")
        
        # Analyze each layer
        for name in layers:
            try:
                layer_info = self.get_layer_info(name)
                
                # Check if layer has gradients
                if "gradients" not in layer_info.get("tensor_types", []):
                    continue
                    
                total_layers_with_gradients += 1
                layer_stats = {}
                
                # Get all gradient tensors for this layer
                tensors = layer_info.get("tensors", {}).get("gradients", [])
                
                if not tensors:
                    continue
                
                layer_gradient_count = 0
                
                for tensor_name in tensors:
                    try:
                        # Get gradient data
                        grad_data = self.get_tensor_data(name, "gradients", tensor_name)
                        
                        # Check if we actually got data
                        if grad_data is None or grad_data.size == 0:
                            print(f"Warning: Empty gradient data for {name}/{tensor_name}")
                            continue
                        
                        layer_gradient_count += 1
                        total_gradient_tensors += 1
                        
                        # Calculate statistics with error handling
                        try:
                            tensor_stats = {
                                "shape": grad_data.shape,
                                "dtype": str(grad_data.dtype),
                            }
                            
                            # Basic statistics
                            if grad_data.size > 0:
                                tensor_stats.update({
                                    "mean_abs": float(np.mean(np.abs(grad_data))),
                                    "mean": float(np.mean(grad_data)),
                                    "std": float(np.std(grad_data)),
                                    "min": float(np.min(grad_data)),
                                    "max": float(np.max(grad_data)),
                                    "zero_fraction": float(np.mean(grad_data == 0.0)),
                                })
                                
                                # Add norm across time if we have time dimension
                                if len(grad_data.shape) > 1:
                                    try:
                                        norms = np.linalg.norm(grad_data.reshape(grad_data.shape[0], -1), axis=1)
                                        tensor_stats.update({
                                            "norm_mean": float(np.mean(norms)),
                                            "norm_std": float(np.std(norms)),
                                            "norm_min": float(np.min(norms)),
                                            "norm_max": float(np.max(norms)),
                                        })
                                    except Exception as norm_error:
                                        tensor_stats["norm_error"] = str(norm_error)
                                else:
                                    # For 1D data, norm is just absolute value
                                    tensor_stats.update({
                                        "norm_mean": tensor_stats["mean_abs"],
                                        "norm_std": 0.0,
                                        "norm_min": tensor_stats["mean_abs"],
                                        "norm_max": tensor_stats["mean_abs"],
                                    })
                            
                            layer_stats[tensor_name] = tensor_stats
                            
                        except Exception as stat_error:
                            error_msg = f"Error computing statistics for {name}/{tensor_name}: {stat_error}"
                            errors_encountered.append(error_msg)
                            print(f"Warning: {error_msg}")
                            
                    except Exception as data_error:
                        error_msg = f"Error analyzing gradients for {name}/{tensor_name}: {data_error}"
                        errors_encountered.append(error_msg)
                        print(f"Warning: {error_msg}")
                
                if layer_gradient_count > 0:
                    stats[name] = layer_stats
                    print(f"Found {layer_gradient_count} gradient tensors for layer {name}")
                    
            except Exception as layer_error:
                error_msg = f"Error analyzing layer {name}: {layer_error}"
                errors_encountered.append(error_msg)
                print(f"Warning: {error_msg}")
        
        # Add summary information
        summary = {
            "total_layers": total_layers,
            "layers_with_gradients": total_layers_with_gradients,
            "total_gradient_tensors": total_gradient_tensors,
            "gradient_coverage": total_layers_with_gradients / total_layers if total_layers else 0,
            "errors_encountered": len(errors_encountered),
        }
        
        if total_gradient_tensors == 0:
            print("WARNING: No gradient tensors were found in the dataset!")
            if errors_encountered:
                print("Errors encountered during analysis:")
                for error in errors_encountered[:5]:  # Show first 5 errors
                    print(f"  - {error}")
        else:
            print(f"Found a total of {total_gradient_tensors} gradient tensors across {total_layers_with_gradients} layers")
        
        return {
            "summary": summary,
            "layer_stats": stats,
            "errors": errors_encountered if errors_encountered else None
        }

    def get_optimizer_state(self, step: int) -> List[np.ndarray]:
        """
        Get the optimizer state for a specific step from the IceChunk store.

        Args:
            step: The training step (epoch) for which to retrieve the optimizer state.

        Returns:
            List of NumPy arrays representing the optimizer's state.
            Returns an empty list if the state is not found for the given step.
        """
        optimizer_states_group_name = "optimizer_states"
        if optimizer_states_group_name not in self.root_group: # self.root_group for IceChunk
            print(f"Optimizer states not found in snapshot {self.snapshot_id}.")
            return []

        optimizer_states_group = self.root_group[optimizer_states_group_name]
        step_group_name = f"step_{step}"

        if step_group_name not in optimizer_states_group:
            print(f"Optimizer state for step {step} not found in snapshot {self.snapshot_id}.")
            return []

        step_group = optimizer_states_group[step_group_name]
        optimizer_weights = []
        idx = 0
        while True:
            weight_array_name = f"weight_{idx}"
            if weight_array_name in step_group:
                # Safely get array data regardless of shape
                weight_array = step_group[weight_array_name]
                try:
                    # For arrays with dimensions
                    if weight_array.shape:
                        optimizer_weights.append(weight_array[:])
                    else:
                        # For scalar arrays
                        optimizer_weights.append(np.array(weight_array[()]))
                except Exception as e:
                    print(f"Error reading optimizer weight {weight_array_name} from snapshot {self.snapshot_id}: {e}")
                    # Fallback: try getting as scalar
                    try:
                        optimizer_weights.append(np.array(weight_array.astype(float)))
                    except:
                        print(f"Could not read optimizer weight {weight_array_name} at step {step} from snapshot {self.snapshot_id}")
                idx += 1
            else:
                break
        
        return optimizer_weights

    def get_optimizer_config(self, optimizer_name: str = "optimizer") -> Dict[str, Any]:
        """
        Get the optimizer's configuration from the IceChunk store.

        Args:
            optimizer_name: The name of the optimizer config to retrieve (defaults to "optimizer").

        Returns:
            Dictionary containing the optimizer configuration.
            Returns an empty dict if not found.
        """
        optimizer_info_group_name = "optimizer_info"
        if optimizer_info_group_name not in self.root_group:
            print(f"Optimizer info (configurations) not found in snapshot {self.snapshot_id}.")
            return {}

        optimizer_info_group = self.root_group[optimizer_info_group_name]

        if optimizer_name not in optimizer_info_group:
            print(f"Configuration for optimizer '{optimizer_name}' not found in snapshot {self.snapshot_id}.")
            # Check if default 'optimizer' exists if a specific one was requested and not found
            if optimizer_name != "optimizer" and "optimizer" in optimizer_info_group:
                print(f"Returning config for default 'optimizer' instead.")
                optimizer_name = "optimizer"
            else:
                return {}
            
        opt_config_group = optimizer_info_group[optimizer_name]
        config = {}
        # In IceChunk/Zarr, attributes are directly accessible and should be of correct type or JSON string
        for key, value in opt_config_group.attrs.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse if it looks like a JSON string
                    # (e.g., complex objects might be stored as JSON strings in attributes)
                    parsed_value = json.loads(value)
                    config[key] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    # If not a valid JSON string, use the string value as is
                    config[key] = value
            else:
                # If not a string, assume it's already the correct type
                config[key] = value
        return config 
        
    def get_tensor_metrics(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_name: str,
        metric_path: str,
        steps: Optional[Union[int, slice]] = None,
    ) -> np.ndarray:
        """
        Get metrics data for a specific tensor.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (weights, gradients, activations)
            tensor_name: Name of the tensor
            metric_name: Name of the metric (l2, mean, var, etc.)
            steps: Step or slice to retrieve (None for all steps)
            
        Returns:
            Numpy array of metric values
        """
        metric_path = f"{layer_name}/{tensor_type}/{tensor_name}/{metric_name}"
        return self.get_metrics(metric_path, steps)
    
    def plot_tensor_metrics(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_name: str,
        metric_names: Union[str, List[str]],
        steps: Optional[Union[int, slice]] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Plot metrics for a specific tensor.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (weights, gradients, activations)
            tensor_name: Name of the tensor
            metric_names: Name or list of names of metrics to plot
            steps: Step or slice to retrieve (None for all steps)
            title: Plot title (or None for auto-generated)
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Install it with 'pip install matplotlib'.")
            
        if isinstance(metric_names, str):
            metric_names = [metric_names]
            
        plt.figure(figsize=(10, 6))
        
        for metric_name in metric_names:
            try:
                data = self.get_tensor_metrics(layer_name, tensor_type, tensor_name, metric_name, steps)
                plt.plot(data, label=metric_name)
            except Exception as e:
                print(f"Error plotting metric {metric_name}: {e}")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Metrics for {layer_name}/{tensor_type}/{tensor_name}")
            
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compute_metrics_for_tensor(
        self,
        layer_name: str,
        tensor_type: str,
        tensor_name: str,
        metrics: Optional[List[str]] = None,
        force_compute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Compute or retrieve metrics for a tensor.
        
        Args:
            layer_name: Name of the layer
            tensor_type: Type of tensor (weights, gradients, activations)
            tensor_name: Name of the tensor
            metrics: List of metrics to compute (or None for defaults)
            force_compute: Whether to force recomputation even if metrics exist
            
        Returns:
            Dictionary of metric names to arrays of values
        """
        # Default metrics if none specified
        if metrics is None:
            metrics = ["l2", "mean", "var", "max", "min", "sparsity"]
            
        # Try to get existing metrics
        if not force_compute:
            try:
                results = {}
                for metric_name in metrics:
                    results[metric_name] = self.get_tensor_metrics(
                        layer_name, tensor_type, tensor_name, metric_name)
                return results
            except Exception as e:
                # Fall back to computing if retrieval fails
                print(f"Metrics retrieval failed: {e}. Computing metrics...")
                
        # Get tensor data for computing metrics
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
        
        # Initialize metrics storage
        results = {}
        steps = tensor_array.shape[0]
        
        # Stream through each step to minimize memory use
        for step in range(steps):
            try:
                data = tensor_array[step]
                
                # Compute each metric
                if "l2" in metrics:
                    if "l2" not in results:
                        results["l2"] = np.zeros(steps)
                    results["l2"][step] = float(np.linalg.norm(data))
                    
                if "mean" in metrics:
                    if "mean" not in results:
                        results["mean"] = np.zeros(steps)
                    results["mean"][step] = float(np.mean(data))
                    
                if "var" in metrics:
                    if "var" not in results:
                        results["var"] = np.zeros(steps)
                    results["var"][step] = float(np.var(data))
                    
                if "max" in metrics:
                    if "max" not in results:
                        results["max"] = np.zeros(steps)
                    results["max"][step] = float(np.max(data))
                    
                if "min" in metrics:
                    if "min" not in results:
                        results["min"] = np.zeros(steps)
                    results["min"][step] = float(np.min(data))
                    
                if "sparsity" in metrics:
                    if "sparsity" not in results:
                        results["sparsity"] = np.zeros(steps)
                    results["sparsity"][step] = float(np.mean(data == 0))
            except Exception as e:
                print(f"Error computing metrics for step {step}: {e}")
        
        return results 

    # Enhanced Git-like functionality
    
    def diff_snapshots_visual(
        self,
        reference1: str,
        reference2: str,
        output_format: str = "console",
        include_values: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Create a visual diff between two snapshots.
        
        Args:
            reference1: First reference (snapshot ID, branch, or tag)
            reference2: Second reference (snapshot ID, branch, or tag)
            output_format: Output format - 'console', 'html', or 'dict'
            include_values: Whether to include actual tensor values in diff
            
        Returns:
            Formatted diff output
        """
        # Resolve references
        snapshot1_id = reference1 if len(reference1) > 10 else None
        snapshot2_id = reference2 if len(reference2) > 10 else None
        
        if not snapshot1_id or not snapshot2_id:
            # Try to resolve through repo
            try:
                if hasattr(self, 'repo'):
                    if not snapshot1_id:
                        snapshot1_id = self._resolve_reference(reference1)
                    if not snapshot2_id:
                        snapshot2_id = self._resolve_reference(reference2)
            except:
                pass
                
        if not snapshot1_id or not snapshot2_id:
            raise ValueError("Could not resolve snapshot references")
        
        # Create analyzers for both snapshots
        analyzer1 = IcechunkModelAnalyzer(self.repo, snapshot_id=snapshot1_id)
        analyzer2 = IcechunkModelAnalyzer(self.repo, snapshot_id=snapshot2_id)
        
        diff_data = {
            "snapshot1": snapshot1_id[:8],
            "snapshot2": snapshot2_id[:8],
            "metadata": self._diff_metadata(analyzer1, analyzer2),
            "layers": self._diff_layers_detailed(analyzer1, analyzer2, include_values),
            "metrics": self._diff_metrics(analyzer1, analyzer2),
            "summary": {
                "total_changes": 0,
                "layers_added": [],
                "layers_removed": [],
                "layers_modified": []
            }
        }
        
        # Calculate summary
        for layer_name, layer_diff in diff_data["layers"].items():
            if layer_diff.get("status") == "added":
                diff_data["summary"]["layers_added"].append(layer_name)
            elif layer_diff.get("status") == "removed":
                diff_data["summary"]["layers_removed"].append(layer_name)
            elif layer_diff.get("has_changes", False):
                diff_data["summary"]["layers_modified"].append(layer_name)
                
        diff_data["summary"]["total_changes"] = (
            len(diff_data["summary"]["layers_added"]) +
            len(diff_data["summary"]["layers_removed"]) +
            len(diff_data["summary"]["layers_modified"]) +
            len(diff_data["metadata"])
        )
        
        # Format output
        if output_format == "dict":
            return diff_data
        elif output_format == "console":
            return self._format_diff_console(diff_data)
        elif output_format == "html":
            return self._format_diff_html(diff_data)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _resolve_reference(self, reference: str) -> str:
        """Resolve a reference to a snapshot ID."""
        try:
            # Try as branch
            return self.repo.lookup_branch(reference)
        except:
            pass
        try:
            # Try as tag
            return self.repo.lookup_tag(reference)
        except:
            pass
        # Assume it's a snapshot ID
        return reference
    
    def _diff_metadata(self, analyzer1, analyzer2) -> Dict[str, Any]:
        """Compare metadata between two analyzers."""
        meta1 = analyzer1.get_run_metadata()
        meta2 = analyzer2.get_run_metadata()
        
        diff = {}
        all_keys = set(meta1.keys()) | set(meta2.keys())
        
        for key in all_keys:
            val1 = meta1.get(key)
            val2 = meta2.get(key)
            if val1 != val2:
                diff[key] = {"old": val1, "new": val2}
                
        return diff
    
    def _diff_layers_detailed(self, analyzer1, analyzer2, include_values: bool) -> Dict[str, Any]:
        """Compare layers between two analyzers with detailed information."""
        layers1 = set(analyzer1.get_layer_names())
        layers2 = set(analyzer2.get_layer_names())
        
        diff = {}
        
        # Check removed layers
        for layer in layers1 - layers2:
            diff[layer] = {"status": "removed"}
            
        # Check added layers
        for layer in layers2 - layers1:
            diff[layer] = {"status": "added"}
            
        # Check modified layers
        for layer in layers1 & layers2:
            layer_diff = self._compare_layer_detailed(
                analyzer1, analyzer2, layer, include_values
            )
            if layer_diff["has_changes"]:
                diff[layer] = layer_diff
                
        return diff
    
    def _compare_layer_detailed(self, analyzer1, analyzer2, layer_name: str, include_values: bool) -> Dict[str, Any]:
        """Compare a specific layer between two analyzers."""
        info1 = analyzer1.get_layer_info(layer_name)
        info2 = analyzer2.get_layer_info(layer_name)
        
        layer_diff = {
            "has_changes": False,
            "attribute_changes": {},
            "tensor_changes": {}
        }
        
        # Compare attributes
        attrs1 = {k: v for k, v in info1.items() if k not in ['tensor_types', 'tensors']}
        attrs2 = {k: v for k, v in info2.items() if k not in ['tensor_types', 'tensors']}
        
        for key in set(attrs1.keys()) | set(attrs2.keys()):
            if attrs1.get(key) != attrs2.get(key):
                layer_diff["attribute_changes"][key] = {
                    "old": attrs1.get(key),
                    "new": attrs2.get(key)
                }
                layer_diff["has_changes"] = True
        
        # Compare tensors
        tensors1 = info1.get('tensors', {})
        tensors2 = info2.get('tensors', {})
        
        for tensor_type in set(tensors1.keys()) | set(tensors2.keys()):
            type_diff = {}
            
            t1_names = set(tensors1.get(tensor_type, []))
            t2_names = set(tensors2.get(tensor_type, []))
            
            # Removed tensors
            for name in t1_names - t2_names:
                type_diff[name] = {"status": "removed"}
                
            # Added tensors
            for name in t2_names - t1_names:
                type_diff[name] = {"status": "added"}
                
            # Modified tensors
            for name in t1_names & t2_names:
                if include_values:
                    try:
                        # Compare actual values
                        data1 = analyzer1.get_tensor_data(layer_name, tensor_type, name)
                        data2 = analyzer2.get_tensor_data(layer_name, tensor_type, name)
                        
                        if data1.shape != data2.shape:
                            type_diff[name] = {
                                "shape_changed": True,
                                "old_shape": data1.shape,
                                "new_shape": data2.shape
                            }
                        elif not np.array_equal(data1[-1], data2[-1]):
                            # Just compare last timestep
                            type_diff[name] = {
                                "values_changed": True,
                                "norm_diff": float(np.linalg.norm(data2[-1] - data1[-1]))
                            }
                    except:
                        type_diff[name] = {"error": "Could not compare values"}
                        
            if type_diff:
                layer_diff["tensor_changes"][tensor_type] = type_diff
                layer_diff["has_changes"] = True
                
        return layer_diff
    
    def _diff_metrics(self, analyzer1, analyzer2) -> Dict[str, Any]:
        """Compare metrics between two analyzers."""
        try:
            metrics1 = analyzer1.get_metrics()
            metrics2 = analyzer2.get_metrics()
            
            diff = {}
            all_metrics = set(metrics1.keys()) | set(metrics2.keys())
            
            for metric in all_metrics:
                if metric not in metrics1:
                    diff[metric] = {"status": "added"}
                elif metric not in metrics2:
                    diff[metric] = {"status": "removed"}
                else:
                    # Compare values
                    v1 = metrics1[metric]
                    v2 = metrics2[metric]
                    if len(v1) != len(v2) or not np.array_equal(v1, v2):
                        diff[metric] = {
                            "old_length": len(v1),
                            "new_length": len(v2),
                            "last_value_old": float(v1[-1]) if len(v1) > 0 else None,
                            "last_value_new": float(v2[-1]) if len(v2) > 0 else None
                        }
                        
            return diff
        except:
            return {}
    
    def _format_diff_console(self, diff_data: Dict[str, Any]) -> str:
        """Format diff data for console output."""
        lines = []
        
        # Header
        lines.append(f"\nDiff between {diff_data['snapshot1']} and {diff_data['snapshot2']}")
        lines.append("=" * 60)
        
        # Summary
        summary = diff_data['summary']
        lines.append(f"\nSummary: {summary['total_changes']} total changes")
        if summary['layers_added']:
            lines.append(f"  Added layers: {', '.join(summary['layers_added'])}")
        if summary['layers_removed']:
            lines.append(f"  Removed layers: {', '.join(summary['layers_removed'])}")
        if summary['layers_modified']:
            lines.append(f"  Modified layers: {', '.join(summary['layers_modified'])}")
        
        # Metadata changes
        if diff_data['metadata']:
            lines.append("\nMetadata Changes:")
            for key, change in diff_data['metadata'].items():
                lines.append(f"  {key}: {change['old']} → {change['new']}")
        
        # Layer changes
        if diff_data['layers']:
            lines.append("\nLayer Changes:")
            for layer_name, layer_diff in diff_data['layers'].items():
                if layer_diff.get('status') == 'added':
                    lines.append(f"  + {layer_name} (added)")
                elif layer_diff.get('status') == 'removed':
                    lines.append(f"  - {layer_name} (removed)")
                else:
                    lines.append(f"  ~ {layer_name} (modified)")
                    
                    # Show attribute changes
                    if layer_diff.get('attribute_changes'):
                        for attr, change in layer_diff['attribute_changes'].items():
                            lines.append(f"    {attr}: {change['old']} → {change['new']}")
                    
                    # Show tensor changes
                    if layer_diff.get('tensor_changes'):
                        for tensor_type, tensors in layer_diff['tensor_changes'].items():
                            lines.append(f"    {tensor_type}:")
                            for tensor_name, change in tensors.items():
                                if change.get('status') == 'added':
                                    lines.append(f"      + {tensor_name}")
                                elif change.get('status') == 'removed':
                                    lines.append(f"      - {tensor_name}")
                                elif change.get('shape_changed'):
                                    lines.append(f"      ~ {tensor_name}: shape {change['old_shape']} → {change['new_shape']}")
                                elif change.get('values_changed'):
                                    lines.append(f"      ~ {tensor_name}: values changed (norm diff: {change.get('norm_diff', 'N/A'):.6f})")
        
        return "\n".join(lines)
    
    def _format_diff_html(self, diff_data: Dict[str, Any]) -> str:
        """Format diff data as HTML."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: monospace; }}
                .added {{ color: green; }}
                .removed {{ color: red; }}
                .modified {{ color: orange; }}
                .diff-section {{ margin: 20px 0; }}
                .indent1 {{ margin-left: 20px; }}
                .indent2 {{ margin-left: 40px; }}
            </style>
        </head>
        <body>
            <h2>Diff between {diff_data['snapshot1']} and {diff_data['snapshot2']}</h2>
            
            <div class="diff-section">
                <h3>Summary</h3>
                <p>Total changes: {diff_data['summary']['total_changes']}</p>
                <ul>
                    <li class="added">Added layers: {len(diff_data['summary']['layers_added'])}</li>
                    <li class="removed">Removed layers: {len(diff_data['summary']['layers_removed'])}</li>
                    <li class="modified">Modified layers: {len(diff_data['summary']['layers_modified'])}</li>
                </ul>
            </div>
        """
        
        # Add more HTML formatting as needed
        
        html += "</body></html>"
        return html
    
    def checkout_analyzer(self, reference: str) -> 'IcechunkModelAnalyzer':
        """
        Create a new analyzer instance for a different snapshot.
        
        Args:
            reference: Snapshot ID, branch, or tag
            
        Returns:
            New IcechunkModelAnalyzer instance
        """
        snapshot_id = self._resolve_reference(reference)
        return IcechunkModelAnalyzer(
            self.repo,
            snapshot_id=snapshot_id,
            lazy_loading=self.lazy_loading
        )

    def analyze_branch_evolution(self, branch_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze the evolution of a model across commits on a branch.
        
        Args:
            branch_name: Name of the branch to analyze
            limit: Maximum number of commits to analyze
            
        Returns:
            Analysis of model evolution across the branch
        """
        if not self.repo:
            raise ConnectionError("Repository not open.")
        
        try:
            # Get branch history
            branch_snapshot = self.repo.lookup_branch(branch_name)
            history = self.repo.ancestry(snapshot_id=branch_snapshot)
            
            evolution = {
                "branch": branch_name,
                "total_commits": 0,
                "commits_analyzed": 0,
                "parameter_evolution": {},
                "architecture_changes": [],
                "training_progression": {}
            }
            
            prev_analyzer = None
            commit_count = 0
            
            for snapshot_info in history:
                if limit and commit_count >= limit:
                    break
                
                evolution["total_commits"] += 1
                
                try:
                    # Create analyzer for this snapshot
                    current_analyzer = IcechunkModelAnalyzer(
                        self.repo, snapshot_id=snapshot_info.id, lazy_loading=True
                    )
                    
                    # Get basic metadata
                    metadata = current_analyzer.get_run_metadata()
                    
                    commit_info = {
                        "snapshot_id": snapshot_info.id,
                        "message": snapshot_info.message,
                        "timestamp": snapshot_info.written_at.isoformat() if snapshot_info.written_at else None,
                        "step": metadata.get("current_step", commit_count),
                    }
                    
                    # Compare with previous commit if available
                    if prev_analyzer:
                        try:
                            # Simple parameter count comparison
                            curr_layers = current_analyzer.get_layer_names()
                            prev_layers = prev_analyzer.get_layer_names()
                            
                            if set(curr_layers) != set(prev_layers):
                                evolution["architecture_changes"].append({
                                    "commit": snapshot_info.id,
                                    "layers_added": list(set(curr_layers) - set(prev_layers)),
                                    "layers_removed": list(set(prev_layers) - set(curr_layers))
                                })
                            
                        except Exception as e:
                            print(f"Error comparing snapshots: {e}")
                    
                    evolution["commits_analyzed"] += 1
                    prev_analyzer = current_analyzer
                    commit_count += 1
                    
                except Exception as e:
                    print(f"Error analyzing snapshot {snapshot_info.id}: {e}")
            
            return evolution
            
        except Exception as e:
            print(f"Error analyzing branch evolution: {e}")
            return {"error": str(e)}

    def compare_branches(self, branch1: str, branch2: str) -> Dict[str, Any]:
        """
        Compare the latest state of two branches.
        
        Args:
            branch1: First branch name
            branch2: Second branch name
            
        Returns:
            Comparison results between the branches
        """
        if not self.repo:
            raise ConnectionError("Repository not open.")
        
        try:
            # Get latest snapshots for both branches
            snap1 = self.repo.lookup_branch(branch1)
            snap2 = self.repo.lookup_branch(branch2)
            
            # Create analyzers for both branches
            analyzer1 = IcechunkModelAnalyzer(self.repo, snapshot_id=snap1, lazy_loading=True)
            analyzer2 = IcechunkModelAnalyzer(self.repo, snapshot_id=snap2, lazy_loading=True)
            
            comparison = {
                "branch1": {"name": branch1, "snapshot": snap1},
                "branch2": {"name": branch2, "snapshot": snap2},
                "metadata_diff": self._diff_metadata(analyzer1, analyzer2),
                "layer_diff": self._diff_layers_detailed(analyzer1, analyzer2, include_values=False),
                "training_diff": {},
                "summary": {
                    "branches_diverged": snap1 != snap2,
                    "architecture_identical": True,
                    "training_differences": []
                }
            }
            
            # Check if architectures are identical
            layers1 = set(analyzer1.get_layer_names())
            layers2 = set(analyzer2.get_layer_names())
            comparison["summary"]["architecture_identical"] = layers1 == layers2
            
            # Compare training progression
            try:
                meta1 = analyzer1.get_run_metadata()
                meta2 = analyzer2.get_run_metadata()
                
                step1 = meta1.get("current_step", 0)
                step2 = meta2.get("current_step", 0)
                
                comparison["training_diff"] = {
                    "step_difference": step2 - step1,
                    "branch1_step": step1,
                    "branch2_step": step2
                }
                
                if step1 != step2:
                    comparison["summary"]["training_differences"].append("Different training progress")
                    
            except Exception as e:
                comparison["training_diff"]["error"] = str(e)
            
            return comparison
            
        except Exception as e:
            return {"error": f"Error comparing branches: {e}"}

    def get_commit_info(self, reference: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific commit.
        
        Args:
            reference: Snapshot ID, branch, or tag
            
        Returns:
            Detailed commit information
        """
        snapshot_id = self._resolve_reference(reference)
        if not snapshot_id:
            raise ValueError(f"Reference '{reference}' not found")
        
        try:
            # Get snapshot info
            snapshot = self.repo.get_snapshot(snapshot_id)
            
            # Create analyzer for this snapshot
            analyzer = IcechunkModelAnalyzer(self.repo, snapshot_id=snapshot_id, lazy_loading=True)
            
            commit_info = {
                "snapshot_id": snapshot_id,
                "message": snapshot.message,
                "timestamp": snapshot.written_at.isoformat() if snapshot.written_at else None,
                "metadata": analyzer.get_run_metadata(),
                "layers": analyzer.get_layer_names(),
                "layer_count": len(analyzer.get_layer_names()),
                "has_gradients": False,
                "gradient_layers": []
            }
            
            # Check for gradient data
            for layer_name in analyzer.get_layer_names():
                try:
                    layer_info = analyzer.get_layer_info(layer_name)
                    if "gradients" in layer_info.get("tensor_types", []):
                        commit_info["has_gradients"] = True
                        commit_info["gradient_layers"].append(layer_name)
                except:
                    pass
            
            # Get tags pointing to this snapshot
            try:
                all_tags = self.repo.list_tags()
                commit_tags = [tag for tag, tag_snap in all_tags.items() if tag_snap == snapshot_id]
                commit_info["tags"] = commit_tags
            except:
                commit_info["tags"] = []
            
            return commit_info
            
        except Exception as e:
            return {"error": f"Error getting commit info: {e}"}

    def analyze_training_trends(self, reference: Optional[str] = None, lookback_commits: int = 10) -> Dict[str, Any]:
        """
        Analyze training trends across recent commits.
        
        Args:
            reference: Reference to start analysis from (defaults to current)
            lookback_commits: Number of commits to analyze
            
        Returns:
            Training trend analysis
        """
        if not self.repo:
            raise ConnectionError("Repository not open.")
        
        start_snapshot = reference or self.snapshot_id
        if reference:
            start_snapshot = self._resolve_reference(reference)
        
        try:
            # Get commit history
            history = list(self.repo.ancestry(snapshot_id=start_snapshot))[:lookback_commits]
            
            trends = {
                "commits_analyzed": len(history),
                "step_progression": [],
                "parameter_trends": {},
                "architecture_stability": True,
                "gradient_availability": []
            }
            
            baseline_layers = None
            
            for i, snapshot_info in enumerate(history):
                try:
                    analyzer = IcechunkModelAnalyzer(
                        self.repo, snapshot_id=snapshot_info.id, lazy_loading=True
                    )
                    
                    metadata = analyzer.get_run_metadata()
                    step = metadata.get("current_step", i)
                    layers = analyzer.get_layer_names()
                    
                    trends["step_progression"].append({
                        "snapshot_id": snapshot_info.id,
                        "step": step,
                        "layer_count": len(layers),
                        "message": snapshot_info.message
                    })
                    
                    # Check architecture stability
                    if baseline_layers is None:
                        baseline_layers = set(layers)
                    elif set(layers) != baseline_layers:
                        trends["architecture_stability"] = False
                    
                    # Check gradient availability
                    has_gradients = False
                    for layer_name in layers[:3]:  # Check first few layers
                        try:
                            layer_info = analyzer.get_layer_info(layer_name)
                            if "gradients" in layer_info.get("tensor_types", []):
                                has_gradients = True
                                break
                        except:
                            pass
                    
                    trends["gradient_availability"].append(has_gradients)
                    
                except Exception as e:
                    print(f"Error analyzing snapshot {snapshot_info.id}: {e}")
            
            # Calculate trend statistics
            if len(trends["step_progression"]) > 1:
                steps = [entry["step"] for entry in trends["step_progression"]]
                trends["step_trend"] = {
                    "increasing": all(steps[i] <= steps[i+1] for i in range(len(steps)-1)),
                    "total_progress": steps[-1] - steps[0] if steps else 0
                }
            
            trends["gradient_consistency"] = all(trends["gradient_availability"]) if trends["gradient_availability"] else False
            
            return trends
            
        except Exception as e:
            return {"error": f"Error analyzing training trends: {e}"} 