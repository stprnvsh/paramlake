"""
Gradient collector for TensorFlow models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from paramlake.storage.storage_interface import StorageInterface
from paramlake.utils.model_utils import get_all_layers, process_tensors_batch


class GradientCollector:
    """Collects gradients from TensorFlow models."""

    def __init__(
        self,
        storage_manager: StorageInterface,
        include_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        include_types: Optional[List[str]] = None,
    ):
        """
        Initialize gradient collector.
        
        Args:
            storage_manager: Storage manager that implements StorageInterface
            include_layers: List of layer name patterns to include
            exclude_layers: List of layer name patterns to exclude
            include_types: List of layer type patterns to include
        """
        self.storage = storage_manager
        self.include_layers = include_layers
        self.exclude_layers = exclude_layers
        self.include_types = include_types
        
        # Map from variable name to (layer_name, tensor_name) for efficient lookup
        self._var_to_layer_map = {}
        
        # Track the original methods that we patch
        self._original_train_step = None
        self._original_apply_gradients = None
    
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
    
    def build_variable_mapping(self, model: tf.keras.Model) -> None:
        """
        Build a mapping from variable name to layer and tensor name.
        This is needed to map gradients back to their layers efficiently.
        
        Args:
            model: TensorFlow model
        """
        self._var_to_layer_map = {}
        
        # Iterate through all layers
        for layer in self._get_all_layers(model):
            if not self.should_capture_layer(layer):
                continue
                
            # For each trainable weight, map its name to the layer and tensor
            for weight in layer.trainable_weights:
                var_name = weight.name
                
                # Extract tensor name
                tensor_name = var_name
                if ":" in tensor_name:
                    tensor_name = tensor_name.split(":")[0]
                if "/" in tensor_name:
                    tensor_name = tensor_name.split("/")[-1]
                
                # Store mapping
                self._var_to_layer_map[var_name] = (layer.name, tensor_name)
    
    def capture_gradients(
        self,
        gradients: List[tf.Tensor],
        variables: List[tf.Variable],
        step: Optional[int] = None,
    ) -> None:
        """
        Capture gradients for trainable variables.
        
        Args:
            gradients: List of gradient tensors
            variables: List of variables (must match gradients)
            step: Current training step (if None, uses internal counter)
        """
        # Ensure we have a variable mapping
        if not self._var_to_layer_map:
            # Try to rebuild the mapping from the variables
            for var in variables:
                var_name = var.name
                # Try to extract layer name from variable name
                parts = var_name.split("/")
                if len(parts) > 1:
                    layer_name = "/".join(parts[:-1])
                    tensor_name = parts[-1]
                    if ":" in tensor_name:
                        tensor_name = tensor_name.split(":")[0]
                    self._var_to_layer_map[var_name] = (layer_name, tensor_name)
        
        # Process each gradient
        for grad, var in zip(gradients, variables):
            if grad is None:
                # Skip variables with no gradient
                continue
                
            var_name = var.name
            
            if var_name in self._var_to_layer_map:
                layer_name, tensor_name = self._var_to_layer_map[var_name]
                
                # Get layer type (best guess if not available)
                layer_type = "Unknown"
                if "/" in var_name:
                    # Try to guess layer type from variable name pattern
                    name_parts = var_name.split("/")
                    if len(name_parts) > 1:
                        if "conv" in name_parts[-2].lower():
                            layer_type = "Conv2D"
                        elif "dense" in name_parts[-2].lower():
                            layer_type = "Dense"
                        elif "batch" in name_parts[-2].lower():
                            layer_type = "BatchNormalization"
                
                try:
                    # Get or create layer group
                    layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                    
                    # Convert gradient to numpy array
                    grad_numpy = self._tensor_to_numpy(grad)
                    
                    # Debug print to verify shape
                    print(f"Storing gradient for {layer_name}/{tensor_name} with shape: {grad_numpy.shape}")
                    
                    # Store gradient
                    self.storage.store_tensor(
                        layer_group,
                        tensor_name,
                        "gradients",
                        grad_numpy,
                        step
                    )
                except Exception as e:
                    # Print the full exception for better debugging
                    import traceback
                    print(f"Error storing gradient {tensor_name} for layer {layer_name}:")
                    traceback.print_exc()
    
    def compute_and_capture_gradients(
        self,
        model: tf.keras.Model,
        inputs: Any,
        targets: Any,
        loss_fn: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Compute and capture gradients for a model.
        
        Args:
            model: TensorFlow model
            inputs: Model inputs
            targets: Target outputs
            loss_fn: Loss function (if None, uses model's compiled loss)
            step: Current training step (if None, uses internal counter)
        """
        # Ensure we have a variable mapping
        if not self._var_to_layer_map:
            self.build_variable_mapping(model)
        
        try:
            # Use GradientTape to compute gradients
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(inputs, training=True)
                
                # Compute loss
                if loss_fn is None:
                    # Use model's compiled loss
                    if not hasattr(model, "compiled_loss"):
                        raise ValueError("Model must be compiled or loss_fn must be provided")
                    
                    loss = model.compiled_loss(targets, predictions)
                else:
                    # Use provided loss function
                    loss = loss_fn(targets, predictions)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Capture gradients
            self.capture_gradients(gradients, model.trainable_variables, step)
        except Exception as e:
            # Print the full exception for better debugging
            import traceback
            print(f"Error computing gradients:")
            traceback.print_exc()
    
    def setup_automatic_gradient_tracking(self, model: tf.keras.Model) -> bool:
        """
        Set up automatic gradient tracking during model training.
        
        Args:
            model: TensorFlow model to track gradients for
            
        Returns:
            True if tracking was successfully set up, False otherwise
        """
        # Build the variable mapping for this model
        self.build_variable_mapping(model)
        
        # Try different strategies to capture gradients
        
        # Strategy 1: Override train_step for custom training loops
        if hasattr(model, 'train_step') and callable(model.train_step):
            return self._setup_train_step_override(model)
            
        # Strategy 2: Override optimizer's apply_gradients method
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            return self._setup_optimizer_override(model.optimizer)
            
        # Strategy 3: Add a gradient tape callback
        if hasattr(model, 'fit') and callable(model.fit):
            return self._setup_gradient_callback(model)
            
        print("Warning: Could not set up automatic gradient tracking. No suitable method found.")
        return False
    
    def _setup_train_step_override(self, model: tf.keras.Model) -> bool:
        """
        Override the model's train_step method to capture gradients.
        
        Args:
            model: TensorFlow model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the original train_step as a class attribute to avoid recursion
            original_train_step = model.train_step
            self._original_train_step = original_train_step
            
            # Define a new train_step that captures gradients
            def gradient_capturing_train_step(data):
                # Call original train_step directly from the stored reference
                result = original_train_step(data)
                
                # Get current step from storage manager
                step = self.storage.current_step
                
                # Try to capture gradients from different sources
                if hasattr(model, 'optimizer') and model.optimizer is not None:
                    if hasattr(model.optimizer, '_gradients') and model.optimizer._gradients is not None:
                        # Capture gradients directly from optimizer
                        self.capture_gradients(
                            model.optimizer._gradients,
                            model.trainable_variables,
                            step
                        )
                
                return result
            
            # Replace the train_step method
            model.train_step = gradient_capturing_train_step
            print("Successfully set up gradient tracking via train_step override")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error setting up train_step override: {e}")
            traceback.print_exc()
            # Restore original train_step if we modified it
            if self._original_train_step is not None:
                model.train_step = self._original_train_step
                self._original_train_step = None
            return False
    
    def _setup_optimizer_override(self, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        """
        Override the optimizer's apply_gradients method to capture gradients.
        
        Args:
            optimizer: TensorFlow optimizer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store the original apply_gradients method
            self._original_apply_gradients = optimizer.apply_gradients
            
            # Create a wrapper that captures gradients
            def apply_gradients_with_capture(grads_and_vars, name=None, experimental_aggregate_gradients=True):
                # Extract gradients and variables
                gradients = [g for g, _ in grads_and_vars if g is not None]
                variables = [v for g, v in grads_and_vars if g is not None]
                
                # Get current step from storage manager
                step = self.storage.current_step
                
                # Capture gradients before applying them
                self.capture_gradients(gradients, variables, step)
                
                # Call the original method
                return self._original_apply_gradients(
                    grads_and_vars, 
                    name=name,
                    experimental_aggregate_gradients=experimental_aggregate_gradients
                )
            
            # Replace the apply_gradients method
            optimizer.apply_gradients = apply_gradients_with_capture
            print("Successfully set up gradient tracking via optimizer override")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error setting up optimizer override: {e}")
            traceback.print_exc()
            # Restore original apply_gradients if we modified it
            if self._original_apply_gradients is not None:
                optimizer.apply_gradients = self._original_apply_gradients
                self._original_apply_gradients = None
            return False
    
    def _setup_gradient_callback(self, model: tf.keras.Model) -> bool:
        """
        Set up a callback to capture gradients using GradientTape.
        
        Args:
            model: TensorFlow model
            
        Returns:
            True if successful, False otherwise
        """
        # This is a fallback method and not implemented yet
        # We would need to create a custom callback that uses GradientTape
        return False
    
    def capture_optimizer_gradients(self, optimizer: tf.keras.optimizers.Optimizer, step: Optional[int] = None) -> bool:
        """
        Capture gradients from the optimizer's recorded gradients.
        
        Args:
            optimizer: TensorFlow optimizer with gradient information
            step: Current step (if None, uses internal counter)
            
        Returns:
            True if gradients were captured, False otherwise
        """
        if hasattr(optimizer, '_gradients') and optimizer._gradients is not None:
            variables = optimizer._variables if hasattr(optimizer, '_variables') else optimizer.variables()
            self.capture_gradients(optimizer._gradients, variables, step)
            return True
        return False
    
    def restore_original_methods(self, model: tf.keras.Model) -> None:
        """
        Restore original model methods after training.
        
        Args:
            model: TensorFlow model with overridden methods
        """
        # Restore original train_step if we modified it
        if self._original_train_step is not None and hasattr(model, 'train_step'):
            model.train_step = self._original_train_step
            self._original_train_step = None
            
        # Restore original apply_gradients if we modified it
        if self._original_apply_gradients is not None and hasattr(model, 'optimizer') and model.optimizer is not None:
            model.optimizer.apply_gradients = self._original_apply_gradients
            self._original_apply_gradients = None
    
    def _get_all_layers(self, model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        """
        Recursively get all layers in a model, including nested layers.
        
        Args:
            model: TensorFlow model
            
        Returns:
            List of all layers
        """
        return get_all_layers(model)

    def capture_gradients_batch(
        self,
        gradients: List[tf.Tensor],
        variables: List[tf.Variable],
        step: Optional[int] = None,
    ) -> None:
        """
        Capture gradients for trainable variables in batch mode for better performance.
        
        Args:
            gradients: List of gradient tensors
            variables: List of variables (must match gradients)
            step: Current training step (if None, uses internal counter)
        """
        # Ensure we have a variable mapping
        if not self._var_to_layer_map:
            # Try to rebuild the mapping from the variables
            for var in variables:
                var_name = var.name
                # Try to extract layer name from variable name
                parts = var_name.split("/")
                if len(parts) > 1:
                    layer_name = "/".join(parts[:-1])
                    tensor_name = parts[-1]
                    if ":" in tensor_name:
                        tensor_name = tensor_name.split(":")[0]
                    self._var_to_layer_map[var_name] = (layer_name, tensor_name)
        
        # Group gradients by layer for batch processing
        layer_gradient_batches = {}
        
        # Collect all gradients by layer
        for grad, var in zip(gradients, variables):
            if grad is None:
                # Skip variables with no gradient
                continue
            
            var_name = var.name
            
            if var_name in self._var_to_layer_map:
                layer_name, tensor_name = self._var_to_layer_map[var_name]
                
                try:
                    # Convert gradient to numpy array
                    grad_numpy = self._tensor_to_numpy(grad)
                    
                    # Add to batch for this layer
                    if layer_name not in layer_gradient_batches:
                        layer_gradient_batches[layer_name] = []
                    
                    layer_gradient_batches[layer_name].append((tensor_name, grad_numpy))
                except Exception as e:
                    print(f"Error processing gradient for {var_name}: {e}")
        
        # Process each layer's batch
        for layer_name, tensor_data_pairs in layer_gradient_batches.items():
            # Get layer type (best guess if not available)
            layer_type = "Unknown"
            for var_name in self._var_to_layer_map:
                if self._var_to_layer_map[var_name][0] == layer_name:
                    if "/" in var_name:
                        name_parts = var_name.split("/")
                        if len(name_parts) > 1:
                            if "conv" in name_parts[-2].lower():
                                layer_type = "Conv2D"
                            elif "dense" in name_parts[-2].lower():
                                layer_type = "Dense"
                            elif "batch" in name_parts[-2].lower():
                                layer_type = "BatchNormalization"
                    break
            
            try:
                # Get or create layer group
                layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                
                # Process the batch
                process_tensors_batch(
                    self.storage,
                    layer_group,
                    tensor_data_pairs,
                    "gradients",
                    step
                )
            except Exception as e:
                import traceback
                print(f"Error storing gradients for layer {layer_name}:")
                traceback.print_exc()

    def compute_and_capture_gradients_batch(
        self,
        model: tf.keras.Model,
        inputs: Any,
        targets: Any,
        loss_fn: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Compute and capture gradients for a model in batch mode.
        
        Args:
            model: TensorFlow model
            inputs: Model inputs
            targets: Target outputs
            loss_fn: Loss function (if None, uses model's compiled loss)
            step: Current training step (if None, uses internal counter)
        """
        # Ensure we have a variable mapping
        if not self._var_to_layer_map:
            self.build_variable_mapping(model)
        
        try:
            # Use GradientTape to compute gradients
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(inputs, training=True)
                
                # Compute loss
                if loss_fn is None:
                    # Use model's compiled loss
                    if not hasattr(model, "compiled_loss"):
                        raise ValueError("Model must be compiled or loss_fn must be provided")
                    
                    loss = model.compiled_loss(targets, predictions)
                else:
                    # Use provided loss function
                    loss = loss_fn(targets, predictions)
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Capture gradients using batch mode
            self.capture_gradients_batch(gradients, model.trainable_variables, step)
        except Exception as e:
            # Print the full exception for better debugging
            import traceback
            print(f"Error computing gradients in batch mode:")
            traceback.print_exc()

    def capture_optimizer_gradients_batch(self, optimizer: tf.keras.optimizers.Optimizer, step: Optional[int] = None) -> bool:
        """
        Capture gradients from the optimizer's recorded gradients in batch mode.
        
        Args:
            optimizer: TensorFlow optimizer with gradient information
            step: Current step (if None, uses internal counter)
            
        Returns:
            True if gradients were captured, False otherwise
        """
        if hasattr(optimizer, '_gradients') and optimizer._gradients is not None:
            variables = optimizer._variables if hasattr(optimizer, '_variables') else optimizer.variables()
            self.capture_gradients_batch(optimizer._gradients, variables, step)
            return True
        return False

    def compute_gradients_with_tape(
        self,
        model: tf.keras.Model,
        inputs: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Compute and capture gradients for a model using GradientTape without explicit targets.
        Useful when we only have sample inputs and need to compute gradients.
        
        Args:
            model: TensorFlow model
            inputs: Model inputs
            step: Current training step (if None, uses internal counter)
        """
        # Ensure we have a variable mapping
        if not self._var_to_layer_map:
            self.build_variable_mapping(model)
        
        try:
            # For sample inputs, we'll use a simple MSE loss against zero targets
            # This is just to get some gradients flowing for demonstration/testing
            with tf.GradientTape() as tape:
                # Ensure inputs are tensor
                if not isinstance(inputs, tf.Tensor):
                    inputs = tf.convert_to_tensor(inputs)
                    
                # Forward pass
                predictions = model(inputs, training=True)
                
                # Simple loss function: MSE against zeros 
                # (just to have something to differentiate)
                loss = tf.reduce_mean(tf.square(predictions))
            
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Capture gradients
            self.capture_gradients(gradients, model.trainable_variables, step)
            return gradients
        except Exception as e:
            # Print the full exception for better debugging
            import traceback
            print(f"Error computing gradients with tape:")
            traceback.print_exc()
            return None 