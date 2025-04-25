"""
Weight collector for TensorFlow models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import sys
import time
import psutil

import numpy as np
import tensorflow as tf

from paramlake.storage.storage_interface import StorageInterface
from paramlake.utils.model_utils import get_all_layers, process_tensors_batch


class WeightCollector:
    """Collects weights from TensorFlow models."""

    def __init__(
        self,
        storage_manager: StorageInterface,
        capture_trainable: bool = True,
        capture_non_trainable: bool = True,
        include_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        include_types: Optional[List[str]] = None,
    ):
        """
        Initialize weight collector.
        
        Args:
            storage_manager: Storage manager that implements StorageInterface
            capture_trainable: Whether to capture trainable weights
            capture_non_trainable: Whether to capture non-trainable weights
            include_layers: List of layer name patterns to include
            exclude_layers: List of layer name patterns to exclude
            include_types: List of layer type patterns to include
        """
        self.storage = storage_manager
        self.capture_trainable = capture_trainable
        self.capture_non_trainable = capture_non_trainable
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.include_types = include_types
        
        # Memory monitoring
        self.memory_threshold = 80  # percent
        self.memory_check_interval = 10  # seconds
        self.last_memory_check = time.time()
        self.memory_warning_issued = False
        
        # Layer size tracking for adaptive collection
        self.layer_sizes = {}  # Map of layer name to approximate size in bytes
        self.total_model_size = 0  # Total size of model parameters in bytes
        self.large_layer_threshold = 10 * 1024 * 1024  # 10MB
    
    def should_capture_layer(self, layer: tf.keras.layers.Layer) -> bool:
        """
        Determine if a layer should be captured based on configuration.
        
        Args:
            layer: TensorFlow layer
            
        Returns:
            True if the layer should be captured, False otherwise
        """
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        
        # Check if layer is in exclude list
        if self.exclude_layers:
            import fnmatch
            if any(fnmatch.fnmatch(layer_name, pattern) for pattern in self.exclude_layers):
                return False
        
        # Check if layer is in include list (if provided)
        if self.include_layers:
            import fnmatch
            if not any(fnmatch.fnmatch(layer_name, pattern) for pattern in self.include_layers):
                return False
        
        # Check if layer type is in include types list (if provided)
        if self.include_types:
            import fnmatch
            if not any(fnmatch.fnmatch(layer_type, pattern) for pattern in self.include_types):
                return False
        
        # Default to capturing all layers
        return True
    
    def check_memory_pressure(self) -> float:
        """
        Check current memory usage and issue warnings if approaching limits.
        
        Returns:
            Current memory usage percentage
        """
        # Only check periodically to reduce overhead
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return 0.0
            
        self.last_memory_check = current_time
        
        # Get current memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Issue warning if memory usage is high
        if memory_usage > self.memory_threshold and not self.memory_warning_issued:
            print(f"Warning: High memory usage detected ({memory_usage:.1f}%). "
                  f"Consider reducing capture frequency or excluding large layers.")
            self.memory_warning_issued = True
        elif memory_usage < self.memory_threshold * 0.8:
            # Reset warning flag when memory usage decreases
            self.memory_warning_issued = False
            
        return memory_usage
    
    def should_skip_large_layer(self, layer_name: str, step: int) -> bool:
        """
        Determine if a large layer should be skipped based on memory pressure and size.
        
        Args:
            layer_name: Name of the layer
            step: Current training step
            
        Returns:
            True if the layer should be skipped, False otherwise
        """
        # If layer is not tracked yet, don't skip
        if layer_name not in self.layer_sizes:
            return False
            
        # Check memory pressure
        memory_usage = self.check_memory_pressure()
        
        # If memory usage is high and layer is large, consider skipping
        if memory_usage > self.memory_threshold and self.layer_sizes[layer_name] > self.large_layer_threshold:
            # Skip large layers more frequently when memory is constrained
            # but still capture periodically (e.g., every 5 steps)
            if step % max(5, int(memory_usage / 10)) != 0:
                return True
                
        return False
    
    def capture_layer_weights(
        self,
        layer: tf.keras.layers.Layer,
        step: Optional[int] = None,
    ) -> None:
        """
        Capture weights for a specific layer.
        
        Args:
            layer: TensorFlow layer
            step: Current training step (if None, uses internal counter)
        """
        if not self.should_capture_layer(layer):
            return
            
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        
        # Skip large layers under memory pressure
        if step is not None and self.should_skip_large_layer(layer_name, step):
            return
            
        # Create a layer group
        layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
        
        # Store metadata if it's the first time
        if "input_shape" not in layer_group.attrs:
            self._store_layer_metadata(layer, layer_group)
        
        # Capture trainable weights
        if self.capture_trainable and layer.trainable_weights:
            self._store_weight_tensors(layer.trainable_weights, layer_group, "weights", step)
        
        # Capture non-trainable weights
        if self.capture_non_trainable and layer.non_trainable_weights:
            self._store_weight_tensors(layer.non_trainable_weights, layer_group, "non_trainable", step)
    
    def _tensor_to_numpy(self, tensor: tf.Tensor) -> np.ndarray:
        """
        Convert a TensorFlow tensor to a NumPy array, handling both eager and graph execution.
        
        Args:
            tensor: TensorFlow tensor
            
        Returns:
            NumPy array
        """
        if tf.executing_eagerly():
            return tensor.numpy()
        else:
            # For graph mode, use tf.keras.backend to get the value
            return tf.keras.backend.get_value(tensor)
    
    def _estimate_tensor_size(self, tensor_data: np.ndarray) -> int:
        """
        Estimate the memory size of a tensor in bytes.
        
        Args:
            tensor_data: NumPy array containing the tensor data
            
        Returns:
            Approximate size in bytes
        """
        # Get element size based on dtype
        element_size = tensor_data.dtype.itemsize
        
        # Calculate total size
        return tensor_data.size * element_size
    
    def _store_weight_tensors(
        self,
        weights: List[tf.Variable],
        layer_group: Any,
        tensor_type: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Store weight tensors in the Zarr store.
        
        Args:
            weights: List of TensorFlow weight variables
            layer_group: Zarr group for the layer
            tensor_type: Type of tensor (weights or non_trainable)
            step: Current training step (if None, uses internal counter)
        """
        layer_size = 0
        
        for weight in weights:
            # Get tensor name (remove variable prefix if present)
            name = weight.name
            if ":" in name:
                name = name.split(":")[0]
            if "/" in name:
                # Extract just the final component of the variable name
                name = name.split("/")[-1]
            
            try:
                # Convert to numpy array
                tensor_data = self._tensor_to_numpy(weight)
                
                # Debug print to verify shape
                print(f"Storing weight for {layer_group.name}/{name} with shape: {tensor_data.shape}")
                
                # Update layer size tracking
                tensor_size = self._estimate_tensor_size(tensor_data)
                layer_size += tensor_size
                
                # Store in Zarr
                self.storage.store_tensor(
                    layer_group,
                    name,
                    tensor_type,
                    tensor_data,
                    step
                )
            except Exception as e:
                # Print the full exception for better debugging
                import traceback
                print(f"Error storing tensor {name} for layer {layer_group.name}:")
                traceback.print_exc()
                
        # Update layer size tracking
        layer_name = layer_group.attrs.get("name", layer_group.name)
        if layer_name not in self.layer_sizes:
            # Update total model size tracking
            self.total_model_size += layer_size
            
        self.layer_sizes[layer_name] = layer_size
    
    def _store_layer_metadata(self, layer: tf.keras.layers.Layer, layer_group: Any) -> None:
        """
        Store metadata for a layer.
        
        Args:
            layer: TensorFlow layer
            layer_group: Zarr group for the layer
        """
        metadata = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "trainable": layer.trainable,
        }
        
        # Add input/output shape if available
        if hasattr(layer, "input_shape") and layer.input_shape is not None:
            input_shape = layer.input_shape
            if isinstance(input_shape, tuple):
                metadata["input_shape"] = list(input_shape)
            else:
                # Handle multiple inputs
                metadata["input_shape"] = [list(shape) for shape in input_shape]
        
        if hasattr(layer, "output_shape") and layer.output_shape is not None:
            output_shape = layer.output_shape
            if isinstance(output_shape, tuple):
                metadata["output_shape"] = list(output_shape)
            else:
                # Handle multiple outputs
                metadata["output_shape"] = [list(shape) for shape in output_shape]
        
        # Add layer configuration if available
        try:
            config = layer.get_config()
            # Convert non-serializable values
            for key, value in config.items():
                if isinstance(value, (np.ndarray, np.number)):
                    config[key] = value.tolist()
            metadata["config"] = config
        except:
            # Some layers may not support get_config
            pass
        
        # Store metadata
        self.storage.store_layer_metadata(layer_group, metadata)
    
    def capture_model_weights(
        self,
        model: tf.keras.Model,
        step: Optional[int] = None,
        recursive: bool = True,
    ) -> None:
        """
        Capture weights for an entire model.
        
        Args:
            model: TensorFlow model
            step: Current training step (if None, uses internal counter)
            recursive: Whether to recursively capture weights for nested layers
        """
        # First ensure the model is built
        if not model.built:
            # Try to build the model with a dummy input
            try:
                if hasattr(model, "input_shape") and model.input_shape is not None:
                    dummy_input_shape = model.input_shape
                    if isinstance(dummy_input_shape, tuple):
                        # Single input
                        dummy_input = np.zeros((1,) + dummy_input_shape[1:], dtype=np.float32)
                        model(dummy_input)
                    else:
                        # Multiple inputs
                        dummy_inputs = [np.zeros((1,) + shape[1:], dtype=np.float32) for shape in dummy_input_shape]
                        model(dummy_inputs)
            except:
                pass
        
        if not model.built:
            # If still not built, we can't capture weights
            print(f"Warning: Model {model.name} is not built, skipping weight capture")
            return
        
        # Check memory pressure before capturing
        memory_usage = self.check_memory_pressure()
        
        # Use the model's layers method to get all layers
        layers = model.layers
        
        # If recursive is True, we'll collect from all nested layers
        if recursive:
            # Recursively get all layers
            all_layers = self._get_all_layers(model)
            
            # Sort layers by size if we've seen them before (capture small layers first)
            if self.layer_sizes:
                # For layers we've never seen, assign a default size of 0
                layer_sizes = {layer.name: self.layer_sizes.get(layer.name, 0) for layer in all_layers}
                # Sort layers by size (smallest first)
                all_layers.sort(key=lambda layer: layer_sizes.get(layer.name, 0))
            
            # Capture weights for each layer
            for layer in all_layers:
                self.capture_layer_weights(layer, step)
        else:
            # Just capture weights for the top-level layers
            for layer in layers:
                self.capture_layer_weights(layer, step)
    
    def _get_all_layers(self, model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        """
        Recursively get all layers in a model, including nested layers.
        
        Args:
            model: TensorFlow model
            
        Returns:
            List of all layers
        """
        return get_all_layers(model)

    def capture_weights_batch(
        self,
        layer: tf.keras.layers.Layer,
        step: Optional[int] = None,
        tensor_type: str = "weights"
    ) -> None:
        """
        Capture weights for a layer in batch mode for better performance.
        
        Args:
            layer: TensorFlow layer
            step: Current training step (if None, uses internal counter)
            tensor_type: Type of tensor ('weights' or 'non_trainable')
        """
        if not self.should_capture_layer(layer):
            return
        
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        
        # Skip large layers under memory pressure
        if step is not None and self.should_skip_large_layer(layer_name, step):
            return
        
        # Create a layer group
        layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
        
        # Store metadata if it's the first time
        if "input_shape" not in layer_group.attrs:
            self._store_layer_metadata(layer, layer_group)
        
        # Select weights based on tensor_type
        weights = layer.trainable_weights if tensor_type == "weights" else layer.non_trainable_weights
        if not weights:
            return
        
        # Prepare batch of tensor name/data pairs
        tensor_data_pairs = []
        layer_size = 0
        
        for weight in weights:
            # Get tensor name (remove variable prefix if present)
            name = weight.name
            if ":" in name:
                name = name.split(":")[0]
            if "/" in name:
                # Extract just the final component of the variable name
                name = name.split("/")[-1]
            
            try:
                # Convert to numpy array
                tensor_data = self._tensor_to_numpy(weight)
                
                # Debug print to verify shape
                print(f"Preparing weight for {layer_group.name}/{name} with shape: {tensor_data.shape}")
                
                # Update layer size tracking
                tensor_size = self._estimate_tensor_size(tensor_data)
                layer_size += tensor_size
                
                # Add to batch
                tensor_data_pairs.append((name, tensor_data))
            except Exception as e:
                import traceback
                print(f"Error processing tensor {name} for layer {layer_group.name}:")
                traceback.print_exc()
        
        # Store the batch of tensors
        successful_writes = process_tensors_batch(
            self.storage, 
            layer_group, 
            tensor_data_pairs, 
            tensor_type, 
            step
        )
        
        # Update layer size tracking
        layer_name = layer_group.attrs.get("name", layer_group.name)
        if successful_writes > 0:
            if layer_name not in self.layer_sizes:
                # Update total model size tracking
                self.total_model_size += layer_size
            
            self.layer_sizes[layer_name] = layer_size

    def capture_model_weights_batch(
        self,
        model: tf.keras.Model,
        step: Optional[int] = None,
        recursive: bool = True,
    ) -> None:
        """
        Capture weights for an entire model using batch processing.
        
        Args:
            model: TensorFlow model
            step: Current training step (if None, uses internal counter)
            recursive: Whether to recursively capture weights for nested layers
        """
        # First ensure the model is built
        if not model.built:
            # Try to build the model with a dummy input
            try:
                if hasattr(model, "input_shape") and model.input_shape is not None:
                    dummy_input_shape = model.input_shape
                    if isinstance(dummy_input_shape, tuple):
                        # Single input
                        dummy_input = np.zeros((1,) + dummy_input_shape[1:], dtype=np.float32)
                        model(dummy_input)
                    else:
                        # Multiple inputs
                        dummy_inputs = [np.zeros((1,) + shape[1:], dtype=np.float32) for shape in dummy_input_shape]
                        model(dummy_inputs)
            except:
                pass
        
        if not model.built:
            # If still not built, we can't capture weights
            print(f"Warning: Model {model.name} is not built, skipping weight capture")
            return
        
        # Check memory pressure before capturing
        memory_usage = self.check_memory_pressure()
        
        # Use the model's layers method to get all layers
        layers = model.layers
        
        # If recursive is True, we'll collect from all nested layers
        if recursive:
            # Recursively get all layers using shared utility
            all_layers = get_all_layers(model)
            
            # Sort layers by size if we've seen them before (capture small layers first)
            if self.layer_sizes:
                # For layers we've never seen, assign a default size of 0
                layer_sizes = {layer.name: self.layer_sizes.get(layer.name, 0) for layer in all_layers}
                # Sort layers by size (smallest first)
                all_layers.sort(key=lambda layer: layer_sizes.get(layer.name, 0))
            
            # Capture weights for each layer using batch processing
            for layer in all_layers:
                if self.capture_trainable and layer.trainable_weights:
                    self.capture_weights_batch(layer, step, "weights")
                if self.capture_non_trainable and layer.non_trainable_weights:
                    self.capture_weights_batch(layer, step, "non_trainable")
        else:
            # Just capture weights for the top-level layers
            for layer in layers:
                if self.capture_trainable and layer.trainable_weights:
                    self.capture_weights_batch(layer, step, "weights")
                if self.capture_non_trainable and layer.non_trainable_weights:
                    self.capture_weights_batch(layer, step, "non_trainable") 