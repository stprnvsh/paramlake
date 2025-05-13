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
                print(f"Layer {layer_name} has has_gradients=True but no gradients group found")
                # Try to see if there's a hidden gradients group
                try:
                    if hasattr(layer_group, 'gradients') or (hasattr(layer_group, '__contains__') and 'gradients' in layer_group):
                        print(f"Found hidden gradients group for layer {layer_name}")
                        tensor_types.append("gradients")
                except Exception as e:
                    print(f"Error checking for hidden gradients group: {e}")
        
        info["tensor_types"] = tensor_types
        info["tensors"] = {}
        
        for tensor_type in tensor_types:
            try:
                tensor_group = layer_group[tensor_type]
                tensor_names = list(tensor_group.keys())
                info["tensors"][tensor_type] = tensor_names
                
                # Print for debugging if this is a gradients group
                if tensor_type == "gradients":
                    print(f"Found gradient tensors for layer {layer_name}: {tensor_names}")
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
        metric_name: Optional[str] = None,
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Get metrics data.
        
        Args:
            metric_name: Name of the metric (or None for all metrics)
            
        Returns:
            Metric data as array or dictionary of arrays
        """
        if self.metrics_group is None:
            if metric_name is None:
                return {}
            else:
                return np.array([])
            
        if metric_name is not None:
            if metric_name not in self.metrics_group:
                return np.array([])
                
            return self.metrics_group[metric_name][:]
        else:
            metrics = {}
            for name in self.metrics_group.keys():
                metrics[name] = self.metrics_group[name][:]
                
            return metrics
    
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
            if layer_name not in self.layers_group:
                raise ValueError(f"Layer {layer_name} not found")
            layers = [layer_name]
        else:
            layers = self.get_layer_names()
            
        stats = {}
        total_layers = len(layers)
        total_layers_with_gradients = 0
        total_gradient_tensors = 0
        
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
            
            if not tensors:
                print(f"Layer {name} has 'gradients' tensor type but no gradient tensors were found.")
                continue
            
            layer_gradient_count = 0
            
            for tensor_name in tensors:
                try:
                    # Get gradient data
                    grad_data = self.get_tensor_data(name, "gradients", tensor_name)
                    
                    # Check if we actually got data
                    if grad_data.size == 0:
                        print(f"Warning: Empty gradient data for {name}/{tensor_name}")
                        continue
                    
                    layer_gradient_count += 1
                    total_gradient_tensors += 1
                    
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
            
            if layer_gradient_count > 0:
                stats[name] = layer_stats
                print(f"Found {layer_gradient_count} gradient tensors for layer {name}")
            else:
                print(f"No valid gradient tensors found for layer {name}")
        
        # Add summary information
        summary = {
            "total_layers": total_layers,
            "layers_with_gradients": total_layers_with_gradients,
            "total_gradient_tensors": total_gradient_tensors,
            "gradient_coverage": total_layers_with_gradients / total_layers if total_layers else 0
        }
        
        if total_gradient_tensors == 0:
            print("WARNING: No gradient tensors were found in the dataset!")
        else:
            print(f"Found a total of {total_gradient_tensors} gradient tensors across {total_layers_with_gradients} layers")
        
        return {
            "summary": summary,
            "layer_stats": stats
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