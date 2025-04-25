"""
Activation collector for TensorFlow models.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from paramlake.storage.storage_interface import StorageInterface
from paramlake.utils.model_utils import get_all_layers, process_tensors_batch


class ActivationCollector:
    """Collects activations from TensorFlow models."""

    def __init__(
        self,
        storage_manager: StorageInterface,
        include_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        include_types: Optional[List[str]] = None,
    ):
        """
        Initialize activation collector.
        
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
        
        # Map of layer names to activation functions
        self.activation_functions = {}
        
        # Sample input for capturing activations
        self.sample_input = None
    
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
        
        # Skip layers that typically don't have meaningful activations
        if layer_type in ["InputLayer", "Dropout", "Reshape", "Flatten"]:
            return False
        
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

    def setup_activation_capture(self, model: tf.keras.Model) -> None:
        """
        Set up activation capture for a model.
        
        Args:
            model: TensorFlow model
        """
        # Clear previous activation functions
        self.activation_functions = {}
        
        # For Keras Functional API and Sequential models,
        # we can create a model with multiple outputs to get activations
        if isinstance(model, tf.keras.Sequential) or hasattr(model, "inputs") and model.inputs is not None:
            self._setup_functional_activation_capture(model)
        else:
            # For subclassed models, we need to use a different approach
            self._setup_subclassed_activation_capture(model)
    
    def _setup_functional_activation_capture(self, model: tf.keras.Model) -> None:
        """
        Set up activation capture for a Functional API or Sequential model.
        
        Args:
            model: TensorFlow model
        """
        # Get all layers to capture
        layers_to_capture = []
        for layer in model.layers:
            if self.should_capture_layer(layer):
                layers_to_capture.append(layer)
        
        if not layers_to_capture:
            return
        
        # Create a model that outputs all layer activations
        try:
            # For Functional API models
            if hasattr(model, "inputs") and model.inputs is not None:
                # Get the input shape
                inputs = model.inputs
                outputs = [layer.output for layer in layers_to_capture]
                
                # Create a model that outputs all activations
                activation_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                
                # Store the activation model
                self.activation_functions["functional"] = activation_model
                
                # Store layer mapping
                self.activation_functions["layer_mapping"] = {
                    i: layer.name for i, layer in enumerate(layers_to_capture)
                }
                return
            
            # For Sequential models
            if isinstance(model, tf.keras.Sequential):
                # Get input layer or first layer
                if model.layers and isinstance(model.layers[0], tf.keras.layers.InputLayer):
                    input_layer = model.layers[0]
                    input_shape = input_layer.input_shape
                else:
                    # Get input shape from the first layer
                    input_shape = model.layers[0].input_shape
                
                # Create input tensor
                if input_shape is not None:
                    inputs = tf.keras.Input(shape=input_shape[1:])
                    
                    # Build the network with outputs at each layer
                    outputs = []
                    x = inputs
                    
                    for layer in model.layers:
                        x = layer(x)
                        if self.should_capture_layer(layer):
                            outputs.append(x)
                    
                    # Create a model that outputs all activations
                    activation_model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    
                    # Store the activation model
                    self.activation_functions["functional"] = activation_model
                    
                    # Store layer mapping
                    self.activation_functions["layer_mapping"] = {
                        i: layer.name for i, layer in enumerate(layers_to_capture)
                    }
                    return
        except Exception as e:
            print(f"Could not set up functional activation capture: {e}")
            pass
        
        # Fall back to subclassed approach if functional doesn't work
        self._setup_subclassed_activation_capture(model)
    
    def _setup_subclassed_activation_capture(self, model: tf.keras.Model) -> None:
        """
        Set up activation capture for a subclassed model using a custom callback approach.
        
        Args:
            model: TensorFlow model
        """
        # Dictionary to store activations
        activations = {}
        
        # Define activation capture functions for each layer
        for layer in self._get_all_layers(model):
            if not self.should_capture_layer(layer):
                continue
            
            # Create a callback function to capture outputs
            def make_activation_function(layer_name):
                def activation_function(inputs, outputs):
                    activations[layer_name] = outputs
                return activation_function
            
            # Register the callback
            callback_fn = make_activation_function(layer.name)
            
            # Store the callback function
            self.activation_functions[layer.name] = callback_fn
        
        # Store the activations dictionary
        self.activation_functions["activations"] = activations
    
    def capture_activations(
        self,
        model: tf.keras.Model,
        inputs: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Capture activations for a model.
        
        Args:
            model: TensorFlow model
            inputs: Model inputs
            step: Current training step (if None, uses internal counter)
        """
        # Ensure activation capture is set up
        if not self.activation_functions:
            try:
                self.setup_activation_capture(model)
            except Exception as e:
                import traceback
                print(f"Error setting up activation capture: {e}")
                traceback.print_exc()
                return
        
        try:
            # For Functional API or Sequential models
            if "functional" in self.activation_functions:
                activation_model = self.activation_functions["functional"]
                layer_mapping = self.activation_functions["layer_mapping"]
                
                # Get activations
                activations = activation_model(inputs)
                
                # Ensure activations is a list (even if there's only one output)
                if not isinstance(activations, list):
                    activations = [activations]
                
                # Store each activation
                for i, activation in enumerate(activations):
                    if i in layer_mapping:
                        layer_name = layer_mapping[i]
                        
                        # Get layer type (best effort)
                        layer_type = "Unknown"
                        for layer in model.layers:
                            if layer.name == layer_name:
                                layer_type = layer.__class__.__name__
                                break
                        
                        try:
                            # Get or create layer group
                            layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                            
                            # Convert to numpy array
                            activation_numpy = self._tensor_to_numpy(activation)
                            
                            # Debug print to verify shape
                            print(f"Storing activation for {layer_name} with shape: {activation_numpy.shape}")
                            
                            # Store in Zarr
                            self.storage.store_tensor(
                                layer_group,
                                "activation",
                                "activations",
                                activation_numpy,
                                step
                            )
                        except Exception as e:
                            import traceback
                            print(f"Error storing activation for layer {layer_name}:")
                            traceback.print_exc()
                
                return
            
            # For subclassed models with registered callbacks
            if "activations" in self.activation_functions:
                activations = self.activation_functions["activations"]
                
                # Register callbacks for each layer's call method
                callbacks = {}
                
                for layer in self._get_all_layers(model):
                    if layer.name in self.activation_functions and callable(self.activation_functions[layer.name]):
                        # Replace layer's call method to capture activations
                        original_call = layer.call
                        callback_fn = self.activation_functions[layer.name]
                        
                        def wrapped_call(self, *args, **kwargs):
                            outputs = original_call(*args, **kwargs)
                            callback_fn(args[0], outputs)
                            return outputs
                        
                        # Store original function
                        callbacks[layer.name] = (layer, original_call)
                        
                        # Replace with wrapped function
                        layer.call = wrapped_call.__get__(layer, layer.__class__)
                
                try:
                    # Forward pass to capture activations
                    model(inputs, training=False)
                    
                    # Store captured activations
                    for layer_name, activation in activations.items():
                        # Get layer type (best effort)
                        layer_type = "Unknown"
                        for layer in model.layers:
                            if layer.name == layer_name:
                                layer_type = layer.__class__.__name__
                                break
                        
                        try:
                            # Get or create layer group
                            layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                            
                            # Convert to numpy array
                            if isinstance(activation, tf.Tensor):
                                activation_numpy = self._tensor_to_numpy(activation)
                            else:
                                activation_numpy = np.array(activation)
                            
                            # Debug print to verify shape
                            print(f"Storing activation for {layer_name} with shape: {activation_numpy.shape}")
                            
                            # Store in Zarr
                            self.storage.store_tensor(
                                layer_group,
                                "activation",
                                "activations",
                                activation_numpy,
                                step
                            )
                        except Exception as e:
                            import traceback
                            print(f"Error storing activation for layer {layer_name}:")
                            traceback.print_exc()
                finally:
                    # Restore original call methods
                    for layer_name, (layer, original_call) in callbacks.items():
                        layer.call = original_call
        except Exception as e:
            import traceback
            print(f"Error capturing activations: {e}")
            traceback.print_exc()
    
    def set_sample_input(self, sample_input: Any) -> None:
        """
        Set sample input for activation capture.
        
        Args:
            sample_input: Sample input for activation capture
        """
        self.sample_input = sample_input
    
    def create_sample_input(self, model: tf.keras.Model, batch_size: int = 1) -> Any:
        """
        Create a sample input for the model.
        
        Args:
            model: TensorFlow model
            batch_size: Batch size for sample input
            
        Returns:
            Sample input for the model
        """
        # First check if we already have a sample input
        if self.sample_input is not None:
            return self.sample_input
            
        # Try to get the input shape from the model
        input_shape = None
        
        # For Sequential models with an input layer
        if isinstance(model, tf.keras.Sequential) and model.layers and isinstance(model.layers[0], tf.keras.layers.InputLayer):
            input_shape = model.layers[0].input_shape
        # For Functional API models
        elif hasattr(model, "inputs") and model.inputs is not None:
            # Could be multiple inputs
            if isinstance(model.inputs, list):
                input_shapes = [input.shape for input in model.inputs]
                
                # Create a list of sample inputs
                inputs = []
                for shape in input_shapes:
                    # Skip batch dimension (first dimension)
                    # Use len(shape) instead of shape.rank for tuple compatibility
                    if len(shape) > 1:
                        # Also handle None dimensions in the shape tuple
                        try:
                            # Attempt to convert shape elements to int, replacing None with a default (e.g., 1)
                            concrete_shape = tuple(d if d is not None else 1 for d in shape[1:])
                            inputs.append(np.zeros((batch_size,) + concrete_shape, dtype=np.float32))
                        except TypeError:
                            # Handle cases where shape elements are not convertible (should be rare)
                            print(f"Warning: Could not create sample input for shape {shape}, skipping.")
                            return None # Cannot proceed if shape is invalid
                    else:
                        # Handle scalar input case (rank 1, just batch dimension)
                        inputs.append(np.zeros((batch_size,), dtype=np.float32))
                
                self.sample_input = inputs
                return self.sample_input
            else:
                input_shape = model.inputs.shape
        # For any model with an input_shape attribute
        elif hasattr(model, "input_shape"):
            input_shape = model.input_shape
        
        # Create a sample input if we have a shape
        if input_shape is not None:
            # Skip batch dimension (first dimension)
            # Use len(input_shape) instead of rank and handle None dimensions
            if input_shape is not None and len(input_shape) > 1:
                try:
                    # Attempt to convert shape elements to int, replacing None with a default (e.g., 1)
                    concrete_shape = tuple(d if d is not None else 1 for d in input_shape[1:])
                    sample_input = np.zeros((batch_size,) + concrete_shape, dtype=np.float32)
                    self.sample_input = sample_input
                    return self.sample_input
                except TypeError:
                    print(f"Warning: Could not create sample input for shape {input_shape}, skipping.")
        
        # If we can't determine the input shape, return None
        return None
    
    def capture_activations_batch(
        self,
        model: tf.keras.Model,
        inputs: Any,
        layers_to_capture: Optional[List[tf.keras.layers.Layer]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Capture activations for multiple layers in batch mode for better performance.
        
        Args:
            model: TensorFlow model
            inputs: Model inputs
            layers_to_capture: List of layers to capture (if None, uses filtered model layers)
            step: Current training step (if None, uses internal counter)
        """
        # Ensure activation capture is set up
        if not self.activation_functions:
            try:
                self.setup_activation_capture(model)
            except Exception as e:
                import traceback
                print(f"Error setting up activation capture: {e}")
                traceback.print_exc()
                return
        
        try:
            # For Functional API or Sequential models
            if "functional" in self.activation_functions:
                activation_model = self.activation_functions["functional"]
                layer_mapping = self.activation_functions["layer_mapping"]
                
                # Get activations
                activations = activation_model(inputs)
                
                # Ensure activations is a list (even if there's only one output)
                if not isinstance(activations, list):
                    activations = [activations]
                
                # Group tensor data by layer for batch processing
                layer_tensor_batches = {}
                
                # Collect all tensor data by layer
                for i, activation in enumerate(activations):
                    if i in layer_mapping:
                        layer_name = layer_mapping[i]
                        
                        try:
                            # Add to batch for this layer
                            activation_numpy = self._tensor_to_numpy(activation)
                            
                            if layer_name not in layer_tensor_batches:
                                layer_tensor_batches[layer_name] = []
                            
                            layer_tensor_batches[layer_name].append(("activation", activation_numpy))
                        except Exception as e:
                            print(f"Error processing activation for layer {layer_name}: {e}")
                
                # Process each layer's batch
                for layer_name, tensor_data_pairs in layer_tensor_batches.items():
                    # Get layer type (best effort)
                    layer_type = "Unknown"
                    for layer in model.layers:
                        if layer.name == layer_name:
                            layer_type = layer.__class__.__name__
                            break
                    
                    try:
                        # Get or create layer group
                        layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                        
                        # Process the batch
                        process_tensors_batch(
                            self.storage,
                            layer_group,
                            tensor_data_pairs,
                            "activations",
                            step
                        )
                    except Exception as e:
                        import traceback
                        print(f"Error storing activations for layer {layer_name}:")
                        traceback.print_exc()
                
                return
            
            # For subclassed models with registered callbacks
            if "activations" in self.activation_functions:
                activations = self.activation_functions["activations"]
                
                # Register callbacks for each layer's call method
                callbacks = {}
                
                for layer in self._get_all_layers(model):
                    if layer.name in self.activation_functions and callable(self.activation_functions[layer.name]):
                        # Replace layer's call method to capture activations
                        original_call = layer.call
                        callback_fn = self.activation_functions[layer.name]
                        
                        def wrapped_call(self, *args, **kwargs):
                            outputs = original_call(*args, **kwargs)
                            callback_fn(args[0], outputs)
                            return outputs
                        
                        # Store original function
                        callbacks[layer.name] = (layer, original_call)
                        
                        # Replace with wrapped function
                        layer.call = wrapped_call.__get__(layer, layer.__class__)
                
                try:
                    # Forward pass to capture activations
                    model(inputs, training=False)
                    
                    # Group tensor data by layer for batch processing
                    layer_tensor_batches = {}
                    
                    # Collect all tensor data by layer
                    for layer_name, activation in activations.items():
                        try:
                            # Convert to numpy array
                            if isinstance(activation, tf.Tensor):
                                activation_numpy = self._tensor_to_numpy(activation)
                            else:
                                activation_numpy = np.array(activation)
                            
                            if layer_name not in layer_tensor_batches:
                                layer_tensor_batches[layer_name] = []
                            
                            layer_tensor_batches[layer_name].append(("activation", activation_numpy))
                        except Exception as e:
                            print(f"Error processing activation for layer {layer_name}: {e}")
                    
                    # Process each layer's batch
                    for layer_name, tensor_data_pairs in layer_tensor_batches.items():
                        # Get layer type (best effort)
                        layer_type = "Unknown"
                        for layer in model.layers:
                            if layer.name == layer_name:
                                layer_type = layer.__class__.__name__
                                break
                        
                        try:
                            # Get or create layer group
                            layer_group = self.storage.create_or_get_layer_group(layer_name, layer_type)
                            
                            # Process the batch
                            process_tensors_batch(
                                self.storage,
                                layer_group,
                                tensor_data_pairs,
                                "activations",
                                step
                            )
                        except Exception as e:
                            import traceback
                            print(f"Error storing activations for layer {layer_name}:")
                            traceback.print_exc()
                finally:
                    # Restore original call methods
                    for layer_name, (layer, original_call) in callbacks.items():
                        layer.call = original_call
        except Exception as e:
            import traceback
            print(f"Error capturing activations in batch mode: {e}")
            traceback.print_exc()
    
    def _get_all_layers(self, model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
        """
        Recursively get all layers in a model, including nested layers.
        
        Args:
            model: TensorFlow model
            
        Returns:
            List of all layers
        """
        return get_all_layers(model) 