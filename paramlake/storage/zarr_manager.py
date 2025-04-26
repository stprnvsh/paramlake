"""
Zarr storage manager for ParamLake.
"""

import json
import os
import threading
import queue
import time
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
from numcodecs import Zstd
from zarr.codecs import Blosc

from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.storage_interface import StorageInterface


class ZarrStorageManager(StorageInterface):
    """Manages efficient storage of model parameters in Zarr format."""

    def __init__(self, config: ParamLakeConfig):
        """
        Initialize the Zarr storage manager.
        
        Args:
            config: ParamLake configuration
        """
        self.config = config
        self.store_path = config["output_path"]
        self.run_id = config["run_id"]
        
        # Create root store and groups
        self.store = zarr.open(self.store_path, mode="a")
        
        # Initialize run group if it doesn't exist
        if self.run_id not in self.store:
            self.run_group = self.store.create_group(self.run_id)
            self._initialize_run_metadata()
        else:
            self.run_group = self.store[self.run_id]
            
        # Create or get layers group
        if "layers" not in self.run_group:
            self.layers_group = self.run_group.create_group("layers")
        else:
            self.layers_group = self.run_group["layers"]
            
        # Create metrics group if it doesn't exist
        if "metrics" not in self.run_group:
            self.metrics_group = self.run_group.create_group("metrics")
        else:
            self.metrics_group = self.run_group["metrics"]
        
        # Track the current step
        self.current_step = 0
        
        # Get step index from existing arrays if any
        self._sync_current_step()
        
        # Setup async writing capabilities if enabled
        self.async_enabled = config.get("async_writes", False)
        self.buffer_size = config.get("buffer_size", 100)
        self.write_queue = queue.Queue(maxsize=self.buffer_size) if self.async_enabled else None
        self.stop_event = threading.Event() if self.async_enabled else None
        self.write_thread = None
        
        # Track memory usage
        self.last_memory_check = time.time()
        self.memory_check_interval = config.get("memory_check_interval", 10)  # seconds
        self.memory_threshold = config.get("memory_threshold", 80)  # percent
        self.adaptive_collection = config.get("adaptive_collection", False)
        self.current_frequency_multiplier = 1.0  # Multiplier for collection frequency
        
        # Tracked layers for debugging
        self.tracked_layers = set()
        
        # Start background thread if async is enabled
        if self.async_enabled:
            self.write_thread = threading.Thread(
                target=self._async_writer_thread,
                daemon=True,
                name="zarr_writer"
            )
            self.write_thread.start()
    
    def _sync_current_step(self) -> None:
        """Synchronize the current step with existing arrays."""
        if "current_step" in self.run_group.attrs:
            self.current_step = self.run_group.attrs["current_step"]
    
    def _initialize_run_metadata(self) -> None:
        """Initialize metadata for the current run."""
        import tensorflow as tf
        from datetime import datetime
        
        # Store basic metadata
        self.run_group.attrs["paramlake_version"] = "0.1.0"
        self.run_group.attrs["framework"] = "tensorflow"
        self.run_group.attrs["framework_version"] = tf.__version__
        self.run_group.attrs["timestamp"] = datetime.now().isoformat()
        self.run_group.attrs["current_step"] = 0
        
        # Store configuration
        self.run_group.attrs["config"] = json.dumps(self.config.to_dict())
    
    def get_compressor(self, tensor_type: str = "weights") -> Any:
        """
        Get the configured compressor.
        
        Args:
            tensor_type: Type of tensor (weights, gradients, activations, etc.)
            
        Returns:
            Compressor object or None
        """
        compression = self.config.get("compression", {})
        algorithm = compression.get("algorithm", "blosc_lz4")
        level = compression.get("level", 3)
        shuffle = compression.get("shuffle", True)
        
        # Use specialized gradient compression settings if available
        if tensor_type == "gradients":
            grad_compression = self.config.get("gradient_compression", {})
            algorithm = grad_compression.get("algorithm", algorithm)
            level = grad_compression.get("level", level)
            shuffle = grad_compression.get("shuffle", shuffle)
        
        if algorithm.startswith("blosc"):
            # Extract blosc variant if specified, use lz4 by default
            cname = algorithm.split("_")[1] if "_" in algorithm else "lz4"
            shuffle_mode = 1 if shuffle else 0
            return Blosc(cname=cname, clevel=level, shuffle=shuffle_mode)
        elif algorithm == "zstd":
            return Zstd(level=level)
        elif algorithm == "none" or algorithm is None:
            return None
        else:
            # Default to blosc with lz4
            return Blosc(cname="lz4", clevel=level, shuffle=1 if shuffle else 0)
    
    def _check_memory_usage(self) -> float:
        """
        Check current memory usage and adjust collection frequency if needed.
        
        Returns:
            Current memory usage percentage
        """
        if not self.adaptive_collection:
            return 0.0
            
        # Only check memory at intervals to reduce overhead
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return 0.0
            
        self.last_memory_check = current_time
        
        # Get current memory usage
        memory_usage = psutil.virtual_memory().percent
        
        # Adjust collection frequency based on memory pressure
        if memory_usage > self.memory_threshold:
            # Reduce collection frequency when memory is high
            self.current_frequency_multiplier = min(4.0, self.current_frequency_multiplier * 1.5)
        elif memory_usage < self.memory_threshold * 0.7:  # Hysteresis to prevent oscillation
            # Return to normal collection frequency when memory is low
            self.current_frequency_multiplier = max(1.0, self.current_frequency_multiplier / 1.2)
            
        return memory_usage
    
    def get_adjusted_frequency(self, base_frequency: int) -> int:
        """
        Get collection frequency adjusted based on memory pressure.
        
        Args:
            base_frequency: Base collection frequency from config
            
        Returns:
            Adjusted collection frequency
        """
        memory_usage = self._check_memory_usage()
        adjusted_frequency = int(base_frequency * self.current_frequency_multiplier)
        return max(1, adjusted_frequency)  # Ensure minimum frequency of 1
    
    def determine_chunks(self, shape: Tuple[int, ...], tensor_type: str = "weights") -> Tuple[int, ...]:
        """
        Determine optimal chunk size for a given tensor shape.
        
        Args:
            shape: Shape of the tensor with time dimension added
            tensor_type: Type of tensor (weights, gradients, activations, etc.)
            
        Returns:
            Tuple of chunk sizes for each dimension
        """
        # Get chunking configuration
        chunking = self.config["chunking"]
        time_chunks = chunking["time_dimension"]
        target_size = chunking["target_chunk_size"]
        
        # Gradients may need different chunking strategy for more efficient storage
        # since they can change more rapidly between steps
        if tensor_type == "gradients":
            time_chunks = min(1, time_chunks)  # Store each gradient step separately by default
            
            # Adjust target size for gradients (potentially smaller chunks)
            if "gradient_chunk_size" in chunking:
                target_size = chunking.get("gradient_chunk_size", target_size)
        
        # Get element size (assuming float32)
        element_size = 4  # bytes per element
        
        # Always chunk along time dimension first
        chunks = [time_chunks]
        
        if len(shape) == 1:
            # Just time dimension, return as is
            return (time_chunks,)
        
        remaining_dimensions = shape[1:]
        
        if chunking["spatial_dimensions"] == "auto":
            # Calculate target elements per chunk based on target size
            target_elements = target_size / element_size
            
            # Divide target elements by time chunks
            target_elements_per_time = target_elements / time_chunks
            
            # Calculate spatial dimension chunks based on target elements
            total_spatial_elements = np.prod(remaining_dimensions)
            
            if total_spatial_elements <= target_elements_per_time:
                # If the spatial elements are small enough, chunk whole arrays
                return tuple([time_chunks] + list(remaining_dimensions))
            
            # Otherwise, calculate chunks for each spatial dimension
            # Start with largest dimensions first for efficiency
            spatial_chunks = []
            elements_per_chunk = 1
            
            # Sort dimensions by size for more optimal chunking
            sorted_dims = sorted(enumerate(remaining_dimensions), key=lambda x: -x[1])
            dim_indices = [i for i, _ in sorted_dims]
            dim_sizes = [d for _, d in sorted_dims]
            
            for dim_size in dim_sizes:
                # If including this whole dimension stays under target, use it
                if elements_per_chunk * dim_size <= target_elements_per_time:
                    spatial_chunks.append(dim_size)
                    elements_per_chunk *= dim_size
                else:
                    # Otherwise subdivide this dimension
                    # Try to make chunks roughly cubic (same size in each dimension)
                    remaining_factor = target_elements_per_time / elements_per_chunk
                    chunk_size = min(dim_size, max(1, int(np.sqrt(remaining_factor))))
                    spatial_chunks.append(chunk_size)
                    elements_per_chunk *= chunk_size
            
            # Reorder chunks back to original dimension order
            reordered_chunks = [None] * len(spatial_chunks)
            for i, dim_idx in enumerate(dim_indices):
                if i < len(spatial_chunks):
                    reordered_chunks[dim_idx] = spatial_chunks[i]
                else:
                    reordered_chunks[dim_idx] = 1
            
            return tuple([time_chunks] + reordered_chunks)
        else:
            # User provided specific chunk sizes
            spatial_chunks = chunking["spatial_dimensions"]
            if isinstance(spatial_chunks, int):
                return tuple([time_chunks] + [spatial_chunks] * len(remaining_dimensions))
            elif isinstance(spatial_chunks, list):
                # Pad or truncate to match shape length
                if len(spatial_chunks) < len(remaining_dimensions):
                    spatial_chunks = spatial_chunks + [1] * (len(remaining_dimensions) - len(spatial_chunks))
                elif len(spatial_chunks) > len(remaining_dimensions):
                    spatial_chunks = spatial_chunks[:len(remaining_dimensions)]
                return tuple([time_chunks] + spatial_chunks)
        
        # Default fallback
        return tuple([time_chunks] + [1] * len(remaining_dimensions))
    
    def create_or_get_layer_group(self, layer_name: str, layer_type: str) -> zarr.Group:
        """
        Create or get a group for a specific layer.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of the layer (e.g., 'Dense', 'Conv2D')
            
        Returns:
            Zarr group for the layer
        """
        # Sanitize layer name for use as a file path
        sanitized_name = layer_name.replace("/", "_").replace(":", "_")
        
        if sanitized_name not in self.layers_group:
            layer_group = self.layers_group.create_group(sanitized_name)
            layer_group.attrs["name"] = layer_name
            layer_group.attrs["type"] = layer_type
            # Add to tracked layers for debugging
            self.tracked_layers.add(layer_name)
            return layer_group
        else:
            layer_group = self.layers_group[sanitized_name]
            # Add to tracked layers for debugging
            self.tracked_layers.add(layer_name)
            return layer_group
    
    def _async_writer_thread(self) -> None:
        """Background thread function that processes write operations from the queue."""
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                # Get next write task with a timeout to allow checking stop_event
                task = self.write_queue.get(timeout=0.5)
                
                try:
                    # Execute the write task
                    layer_group, tensor_name, tensor_type, tensor_data, step = task
                    self._store_tensor_sync(layer_group, tensor_name, tensor_type, tensor_data, step)
                except Exception as e:
                    print(f"Error in async writer thread: {e}")
                finally:
                    # Mark this task as done
                    self.write_queue.task_done()
            except queue.Empty:
                # No tasks available, just continue and check stop_event
                pass
    
    def store_tensor(
        self,
        layer_group: zarr.Group,
        tensor_name: str,
        tensor_type: str,
        tensor_data: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Store a tensor in the Zarr store, using async queue if enabled.
        
        Args:
            layer_group: Zarr group for the layer
            tensor_name: Name of the tensor (e.g., 'kernel', 'bias')
            tensor_type: Type of tensor ('weight', 'gradient', 'activation', etc.)
            tensor_data: Numpy array containing the tensor data
            step: Current step or epoch (if None, use internal counter)
        """
        # Use provided step or current step
        current_step = step if step is not None else self.current_step
        
        if self.async_enabled:
            # Make a copy of the data to prevent it from being modified before writing
            tensor_data_copy = tensor_data.copy()
            
            try:
                # Try to add to queue, but don't block if queue is full
                # This prevents training from slowing down when queue backs up
                self.write_queue.put_nowait((layer_group, tensor_name, tensor_type, tensor_data_copy, current_step))
            except queue.Full:
                # Queue is full, log a warning and drop this tensor
                # Can't use the logger here as it might not be thread-safe
                print(f"Write queue full, dropping tensor: {layer_group.name}/{tensor_type}/{tensor_name} at step {current_step}")
        else:
            # Synchronous storage
            self._store_tensor_sync(layer_group, tensor_name, tensor_type, tensor_data, current_step)
    
    def _store_tensor_sync(
        self,
        layer_group: zarr.Group,
        tensor_name: str,
        tensor_type: str,
        tensor_data: np.ndarray,
        step: int,
    ) -> None:
        """
        Synchronously store a tensor in the Zarr store.
        
        Args:
            layer_group: Zarr group for the layer
            tensor_name: Name of the tensor
            tensor_type: Type of tensor
            tensor_data: Numpy array containing the tensor data
            step: Current step
        """
        # Create a subgroup for the tensor type if it doesn't exist
        if tensor_type not in layer_group:
            tensor_group = layer_group.create_group(tensor_type)
        else:
            tensor_group = layer_group[tensor_type]
        
        # Create or get array for this tensor
        if tensor_name not in tensor_group:
            # Initialize with time dimension
            full_shape = (0,) + tensor_data.shape
            chunks = self.determine_chunks(full_shape, tensor_type)
            
            # Create array with room for growth
            tensor_array = tensor_group.create_dataset(
                tensor_name,
                shape=full_shape,
                chunks=chunks,
                dtype=tensor_data.dtype,
                compressor=self.get_compressor(tensor_type),
            )
            
            # Store metadata
            tensor_array.attrs["shape"] = tensor_data.shape
            tensor_array.attrs["dtype"] = str(tensor_data.dtype)
            
            # Track metadata about when this tensor type was created
            if tensor_type == "gradients":
                tensor_array.attrs["first_gradient_step"] = step
                layer_group.attrs["has_gradients"] = True
        else:
            tensor_array = tensor_group[tensor_name]
        
        # Resize array if needed to accommodate the current step
        if step >= tensor_array.shape[0]:
            new_shape = (step + 1,) + tensor_data.shape
            tensor_array.resize(new_shape)
        
        # Store the data at the current step
        tensor_array[step] = tensor_data
        
        # Log storage of gradient data if it's the first time and verbose is enabled
        if tensor_type == "gradients" and self.config.get("verbose", False):
            if not hasattr(self, "_logged_first_gradient") or tensor_name not in self._logged_first_gradient:
                if not hasattr(self, "_logged_first_gradient"):
                    self._logged_first_gradient = set()
                self._logged_first_gradient.add(tensor_name)
                print(f"Stored first gradient for {layer_group.name}/{tensor_name} at step {step}")
                
                # Check and report statistics about gradient
                try:
                    abs_mean = np.abs(tensor_data).mean()
                    if abs_mean < 1e-10:
                        print(f"Warning: Very small gradient magnitude ({abs_mean:.2e}) for {layer_group.name}/{tensor_name}")
                    elif abs_mean > 100:
                        print(f"Warning: Very large gradient magnitude ({abs_mean:.2e}) for {layer_group.name}/{tensor_name}")
                except:
                    pass
    
    def store_layer_metadata(
        self,
        layer_group: zarr.Group,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store metadata for a layer.
        
        Args:
            layer_group: Zarr group for the layer
            metadata: Dictionary of metadata to store
        """
        for key, value in metadata.items():
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # Store as attribute
            layer_group.attrs[key] = value
    
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
            value: Value of the metric
            step: Current step or epoch (if None, use internal counter)
        """
        # Use provided step or current step
        current_step = step if step is not None else self.current_step
        
        # Create or get array for this metric
        if metric_name not in self.metrics_group:
            # Initialize empty array
            metric_array = self.metrics_group.create_dataset(
                metric_name,
                shape=(0,),
                chunks=(min(100, max(1, self.config["chunking"]["time_dimension"])),),
                dtype=np.float32,
                compressor=self.get_compressor(),
            )
        else:
            metric_array = self.metrics_group[metric_name]
        
        # Resize array if needed
        if current_step >= metric_array.shape[0]:
            metric_array.resize((current_step + 1,))
        
        # Store the value
        metric_array[current_step] = value
    
    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1
        self.run_group.attrs["current_step"] = self.current_step
    
    def set_step(self, step: int) -> None:
        """Set the current step counter."""
        self.current_step = step
        self.run_group.attrs["current_step"] = self.current_step
    
    def get_tracked_layers(self) -> List[str]:
        """Get list of tracked layers."""
        return list(self.tracked_layers)
    
    def close(self) -> None:
        """Close the storage manager and finalize the dataset."""
        # If async enabled, wait for all tasks to complete
        if self.async_enabled and self.write_thread and self.write_thread.is_alive():
            # Set stop event and wait for queue to empty
            self.stop_event.set()
            
            # Wait for all tasks to complete with a timeout
            try:
                self.write_queue.join(timeout=60)  # Wait up to 60 seconds
            except:
                print("Warning: Not all write tasks completed before timeout")
                
            # Wait for thread to terminate
            self.write_thread.join(timeout=5)
            
            if self.write_thread.is_alive():
                print("Warning: Async writer thread did not terminate cleanly")
                
        # Store final step
        self.run_group.attrs["final_step"] = self.current_step 