"""
Main decorator for capturing model information during training.
"""

import functools
import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
import yaml
import zarr

from paramlake.collectors.activation_collector import ActivationCollector
from paramlake.collectors.gradient_collector import GradientCollector
from paramlake.collectors.weight_collector import WeightCollector
from paramlake.storage.zarr_manager import ZarrStorageManager
from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.factory import create_storage_manager


class ParamLakeCallback(tf.keras.callbacks.Callback):
    """
    TensorFlow Keras callback that automatically collects model data during training.
    """
    
    def __init__(
        self,
        config: ParamLakeConfig,
        capture_frequency: int = 1,
        include_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None,
        capture_gradients: bool = True,
        capture_activations: bool = True,
    ):
        """
        Initialize the ParamLake callback.
        
        Args:
            config: Configuration object
            capture_frequency: How often to capture data (in epochs)
            include_layers: List of layer name patterns to include
            exclude_layers: List of layer name patterns to exclude
            capture_gradients: Whether to capture gradients
            capture_activations: Whether to capture activations
        """
        super().__init__()
        self.config = config
        self.storage_manager = create_storage_manager(config)
        self.capture_frequency = capture_frequency
        
        # Override capture_gradients with config if provided
        self.capture_gradients = config.get("gradients", {}).get("enabled", capture_gradients)
        self.auto_gradient_tracking = config.get("gradients", {}).get("auto_tracking", True)
        self.gradient_track_method = config.get("gradients", {}).get("track_method", "auto")
        
        # Create collectors
        self.weight_collector = WeightCollector(
            self.storage_manager,
            include_layers=include_layers,
            exclude_layers=exclude_layers
        )
        
        self.activation_collector = None
        if capture_activations:
            self.activation_collector = ActivationCollector(
                self.storage_manager,
                include_layers=include_layers,
                exclude_layers=exclude_layers
            )
        
        self.gradient_collector = None
        if self.capture_gradients:
            self.gradient_collector = GradientCollector(
                self.storage_manager,
                include_layers=include_layers,
                exclude_layers=exclude_layers
            )
        
        self.sample_data = None
        self.current_epoch = 0
        self._model = None  # Store model as a private attribute
        
        # For gradient capture
        self._gradient_tracking_enabled = False
    
    @property
    def model(self):
        """Get the model."""
        return self._model
        
    @model.setter
    def model(self, model):
        """Set the model."""
        self._model = model
    
    def on_train_begin(self, logs=None):
        # Initialize step counter
        self.storage_manager.set_step(0)
        
        # Try to setup activation capture if enabled
        if self.activation_collector and hasattr(self, 'model') and self.model is not None:
            self.activation_collector.setup_activation_capture(self.model)
            
            # Try to create sample input if we don't have one
            if self.sample_data is None:
                self.sample_data = self.activation_collector.create_sample_input(self.model)
        
        # Setup gradient tracking if enabled
        if self.capture_gradients and self.gradient_collector and hasattr(self, 'model') and self.model is not None:
            try:
                # Build variable mapping
                self.gradient_collector.build_variable_mapping(self.model)
                
                # Set up automatic gradient tracking if configured
                if self.auto_gradient_tracking:
                    print(f"Setting up automatic gradient tracking using method: {self.gradient_track_method}")
                    
                    if self.gradient_track_method == "train_step":
                        # Force train_step override method
                        self._gradient_tracking_enabled = self.gradient_collector._setup_train_step_override(self.model)
                    elif self.gradient_track_method == "optimizer" and hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                        # Force optimizer override method
                        self._gradient_tracking_enabled = self.gradient_collector._setup_optimizer_override(self.model.optimizer)
                    elif self.gradient_track_method == "callback":
                        # Use callback method
                        self._gradient_tracking_enabled = self.gradient_collector._setup_gradient_callback(self.model)
                    else:
                        # Use automatic detection (try all methods)
                        self._gradient_tracking_enabled = self.gradient_collector.setup_automatic_gradient_tracking(self.model)
                    
                    if self._gradient_tracking_enabled:
                        print(f"Successfully enabled automatic gradient tracking")
                    else:
                        print("Warning: Automatic gradient tracking could not be enabled. Gradients may not be captured.")
                        print("ParamLake will attempt to capture gradients manually at batch end.")
                    
                    # Try to capture initial gradients using sample data
                    if self.sample_data is not None and hasattr(self.gradient_collector, 'compute_gradients_with_tape'):
                        try:
                            print("Computing initial gradients with sample data")
                            self.gradient_collector.compute_gradients_with_tape(
                                self.model,
                                self.sample_data,
                                step=0
                            )
                        except Exception as e:
                            print(f"Warning: Failed to compute initial gradients: {e}")
            except Exception as e:
                import traceback
                print(f"Error setting up gradient capture: {e}")
                traceback.print_exc()
    
    def on_epoch_end(self, epoch, logs=None):
        # Store the current epoch
        self.current_epoch = epoch
        
        # Set the storage manager step to the current epoch
        self.storage_manager.set_step(epoch)
        
        # Only capture if epoch matches frequency (using 0-based epoch index)
        if epoch % self.capture_frequency == 0:
            # Capture weights
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # Use the batch processing method if available
                    if hasattr(self.weight_collector, 'capture_model_weights_batch'):
                        self.weight_collector.capture_model_weights_batch(self.model, step=epoch)
                    else:
                        self.weight_collector.capture_model_weights(self.model, step=epoch)
                    
                    # If activation collection is enabled and we have sample data
                    if self.activation_collector and self.sample_data is not None:
                        # Use the batch processing method if available
                        if hasattr(self.activation_collector, 'capture_activations_batch'):
                            self.activation_collector.capture_activations_batch(self.model, self.sample_data, step=epoch)
                        else:
                            self.activation_collector.capture_activations(self.model, self.sample_data, step=epoch)
                except Exception as e:
                    import traceback
                    print(f"Error capturing model state at epoch {epoch}: {e}")
                    traceback.print_exc()
            
            # Store metrics
            if logs:
                try:
                    for name, value in logs.items():
                        # Ensure value is scalar before storing
                        if isinstance(value, (int, float, np.number)):
                            self.storage_manager.store_metric(name, float(value), step=epoch)
                        elif isinstance(value, np.ndarray) and value.size == 1:
                            self.storage_manager.store_metric(name, float(value.item()), step=epoch)
                except Exception as e:
                    print(f"Error storing metrics at epoch {epoch}: {e}")
    
    def on_train_batch_end(self, batch, logs=None):
        """
        Capture gradients at the end of each training batch if automatic tracking is not enabled.
        
        Args:
            batch: Current batch index
            logs: Dictionary of logs from training
        """
        # Capture gradients if enabled and at the right frequency
        if (self.capture_gradients and 
            self.gradient_collector and 
            self.current_epoch % self.capture_frequency == 0):
            
            # If we have auto-tracking enabled, the gradients should already be tracked
            if not self._gradient_tracking_enabled and hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
                try:
                    # Try to capture gradients from the optimizer
                    if hasattr(self.model.optimizer, '_gradients') and self.model.optimizer._gradients is not None:
                        gradients = self.model.optimizer._gradients
                        variables = self.model.optimizer._variables
                        if gradients and variables:
                            self.gradient_collector.capture_gradients(gradients, variables, step=self.current_epoch)
                            print(f"Successfully captured gradients at epoch {self.current_epoch}")
                    # Fallback to batch processing methods
                    elif hasattr(self.gradient_collector, 'capture_optimizer_gradients_batch'):
                        self.gradient_collector.capture_optimizer_gradients_batch(
                            self.model.optimizer, 
                            step=self.current_epoch
                        )
                        print(f"Captured optimizer gradients batch at epoch {self.current_epoch}")
                    elif hasattr(self.gradient_collector, "capture_optimizer_gradients"):
                        self.gradient_collector.capture_optimizer_gradients(
                            self.model.optimizer, 
                            step=self.current_epoch
                        )
                        print(f"Captured optimizer gradients at epoch {self.current_epoch}")
                    else:
                        # Try to force gradient capture using the compute_gradients method if we have sample data
                        if hasattr(self, 'sample_data') and self.sample_data is not None:
                            if hasattr(self.gradient_collector, 'compute_gradients_with_tape'):
                                # Use the sample data to compute gradients with tape
                                self.gradient_collector.compute_gradients_with_tape(
                                    self.model,
                                    self.sample_data,
                                    step=self.current_epoch
                                )
                                print(f"Computed gradients with tape at epoch {self.current_epoch}")
                except Exception as e:
                    # Just log the error and continue
                    print(f"Warning: Error capturing gradients from optimizer: {e}")
                    import traceback
                    traceback.print_exc()
    
    def on_train_end(self, logs=None):
        """Clean up at the end of training."""
        # Restore original methods if we overrode them
        if self.capture_gradients and self.gradient_collector and hasattr(self, 'model') and self.model is not None:
            try:
                self.gradient_collector.restore_original_methods(self.model)
            except Exception as e:
                print(f"Warning: Error restoring original methods: {e}")
    
    def set_sample_data(self, sample_data):
        """Set sample data for activation collection."""
        self.sample_data = sample_data
    
    def close(self):
        """Close the storage manager."""
        # Restore original methods if we're being closed early
        if self.capture_gradients and self.gradient_collector and hasattr(self, 'model') and self.model is not None:
            try:
                self.gradient_collector.restore_original_methods(self.model)
            except Exception as e:
                print(f"Warning: Error restoring original methods during close: {e}")
                
        self.storage_manager.close()


class ModelWrapper:
    """
    Wrapper for TensorFlow model with ParamLake integration.
    
    This wrapper adds functionality to capture model state during training.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: ParamLakeConfig,
    ):
        """
        Initialize model wrapper.
        
        Args:
            model: TensorFlow model
            config: ParamLake configuration
        """
        self.model = model
        self.config = config
        
        # Create storage manager
        self.storage = self._initialize_storage(config)
        
        # Create collectors
        self.weight_collector = WeightCollector(
            self.storage,
            capture_trainable=config["capture_weights"],
            capture_non_trainable=config["capture_non_trainable"],
            include_layers=config["include_layers"],
            exclude_layers=config["exclude_layers"],
            include_types=config["include_types"],
        )
        
        self.gradient_collector = GradientCollector(
            self.storage,
            include_layers=config["include_layers"],
            exclude_layers=config["exclude_layers"],
            include_types=config["include_types"],
        )
        
        self.activation_collector = None
        if config["capture_activations"]:
            self.activation_collector = ActivationCollector(
                self.storage,
                include_layers=config["include_layers"],
                exclude_layers=config["exclude_layers"],
                include_types=config["include_types"],
            )
            
            # Load sample batch if provided
            sample_batch = config["activations"]["sample_batch"]
            if sample_batch and os.path.exists(sample_batch):
                try:
                    # Load sample batch from file
                    self.sample_input = np.load(sample_batch)
                    self.activation_collector.set_sample_input(self.sample_input)
                except:
                    self.sample_input = None
            else:
                self.sample_input = None
        
        # Create callback
        self.callback = ParamLakeCallback(
            config=config,
            capture_frequency=config["capture_frequency"],
            include_layers=config["include_layers"],
            exclude_layers=config["exclude_layers"],
            capture_gradients=config["capture_gradients"],
            capture_activations=config["capture_activations"],
        )
        
        # Set up custom train_step if model supports it
        self._original_train_step = None
        if hasattr(model, "train_step") and callable(model.train_step):
            self._original_train_step = model.train_step
            model.train_step = self._wrapped_train_step
            
    def __call__(self, *args, **kwargs):
        """
        Make the wrapped model callable.
        
        This allows the wrapped model to be used directly in custom training loops.
        
        Args:
            *args: Arguments to pass to the model's call method
            **kwargs: Keyword arguments to pass to the model's call method
            
        Returns:
            Output from the model's call method
        """
        return self.model(*args, **kwargs)
    
    def _wrapped_train_step(self, data):
        """
        Wrapped train_step method for capturing model state during custom training.
        
        Args:
            data: Training data
            
        Returns:
            Results from the original train_step
        """
        result = self._original_train_step(data)
        
        # Check if it's time to capture
        step = self.storage.current_step
        capture_frequency = self.callback.capture_frequency
        
        if step % capture_frequency == 0:
            # Capture weights
            if self.config["capture_weights"]:
                # Use batch processing method if available
                if hasattr(self.weight_collector, 'capture_model_weights_batch'):
                    self.weight_collector.capture_model_weights_batch(self.model, step)
                else:
                    self.weight_collector.capture_model_weights(self.model, step)
            
            # Capture gradients if available
            if self.config["capture_gradients"] and hasattr(self.model, "optimizer") and self.model.optimizer is not None:
                # Use batch processing method if available
                if hasattr(self.gradient_collector, 'capture_optimizer_gradients_batch'):
                    self.gradient_collector.capture_optimizer_gradients_batch(self.model.optimizer, step)
                elif hasattr(self.gradient_collector, "capture_optimizer_gradients"):
                    self.gradient_collector.capture_optimizer_gradients(self.model.optimizer, step)
            
            # Capture activations if enabled and we have sample input
            if (self.config["capture_activations"] and self.activation_collector 
                and hasattr(self, "sample_input") and self.sample_input is not None):
                # Use batch processing method if available
                if hasattr(self.activation_collector, 'capture_activations_batch'):
                    self.activation_collector.capture_activations_batch(self.model, self.sample_input, step)
                else:
                    self.activation_collector.capture_activations(self.model, self.sample_input, step)
            
            # Increment step counter
            self.storage.increment_step()
        
        return result
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped model.
        
        Args:
            name: Name of the attribute
            
        Returns:
            Attribute from the wrapped model
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.model, name)
    
    def close(self):
        """Close the storage manager and release resources."""
        if hasattr(self, "storage"):
            self.storage.close()
            
        # Restore original train_step if it was wrapped
        if self._original_train_step is not None and hasattr(self.model, "train_step"):
            self.model.train_step = self._original_train_step

    def _initialize_storage(self, config):
        """Initialize the appropriate storage backend."""
        if config.get("storage_backend") == "icechunk":
            try:
                from paramlake.storage.icechunk_manager import IcechunkStorageManager
                return IcechunkStorageManager(config)
            except ImportError:
                raise ImportError(
                    "Icechunk storage backend requested but Icechunk is not installed. "
                    "Install it with 'pip install icechunk'."
                )
        else:
            from paramlake.storage.zarr_manager import ZarrStorageManager
            return ZarrStorageManager(config)


def load_config(config=None, **kwargs):
    """
    Load configuration from multiple sources.
    
    Args:
        config: Configuration source (dict, file path, or None)
        **kwargs: Additional configuration parameters
        
    Returns:
        ParamLakeConfig object
    """
    config_dict = {}
    
    # Load config from file if provided
    if isinstance(config, str):
        if os.path.exists(config):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file {config} not found")
    elif isinstance(config, dict):
        config_dict = config.copy()
    
    # Update with kwargs
    config_dict.update(kwargs)
    
    # Create config object
    return ParamLakeConfig(config_dict)


def paramlake(func=None, config=None, **kwargs):
    """
    Decorator for automatically tracking model parameters during training.
    
    This can be used in multiple ways:
    
    1. With parameters:
        @paramlake(capture_frequency=5, output_path="model_data.zarr")
        def train_model():
            ...
    
    2. With a config file:
        @paramlake(config="config.yaml")
        def train_model():
            ...
    
    3. Without parameters (using defaults):
        @paramlake
        def train_model():
            ...
    
    Args:
        func: The function to decorate
        config: Configuration source (dict, file path, or None)
        **kwargs: Additional configuration parameters
            capture_frequency (int): How often to capture data
            capture_gradients (bool): Whether to capture gradients
            capture_activations (bool): Whether to capture activations
            output_path (str): Path to store data
            
    Returns:
        Decorated function
    """
    # Initialize params dictionary
    if not hasattr(paramlake, 'params'):
        paramlake.params = {}
    
    # If called with arguments, update params
    if func is None:
        # Make a copy of the existing params
        existing_params = paramlake.params.copy()
        # Update with new kwargs
        paramlake.params = kwargs
        if config:
            paramlake.params['config'] = config
        # Add back any existing params that weren't overridden
        for key, value in existing_params.items():
            if key not in paramlake.params:
                paramlake.params[key] = value
    
    def decorator_paramlake(func):
        @functools.wraps(func)
        def wrapper_paramlake(*args, **kwargs):
            # Load configuration
            cfg = load_config(**paramlake.params)
            
            # Ensure output directory exists - handle case where output path is just a file
            output_path = cfg.get("output_path", "paramlake_output.zarr")
            # Force creation of the zarr directory store
            print(f"Creating output zarr directory at: {output_path}")
            try:
                # Create the directory if it doesn't exist
                # Handle the case where output_path is just a filename with no directory part
                output_dir = os.path.dirname(os.path.abspath(output_path))
                if output_dir:  # Only create directory if there's a non-empty directory part
                    os.makedirs(output_dir, exist_ok=True)
                
                # Create the Zarr root if it doesn't exist
                root = zarr.open(output_path, mode='w')
                # Create the runs group explicitly (rather than waiting for model training)
                run_id = cfg.get("run_id", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
                run_group = root.create_group(run_id, overwrite=True)
                # Create required subgroups
                run_group.create_group("layers", overwrite=True)
                run_group.create_group("metrics", overwrite=True)
                # Add some basic metadata
                run_group.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                run_group.attrs["final_step"] = 0
            except Exception as e:
                print(f"Warning: Error creating Zarr store: {e}")
                import traceback
                traceback.print_exc()
            
            # Extract collection parameters
            capture_frequency = cfg.get("capture_frequency", 1)
            include_layers = cfg.get("include_layers", None)
            exclude_layers = cfg.get("exclude_layers", None)
            
            # Get gradient capture configuration
            # First check for direct parameter capture_gradients
            capture_gradients = kwargs.pop("capture_gradients", None)
            if capture_gradients is None:
                # Then check for gradients config dictionary
                capture_gradients = cfg.get("gradients", {}).get("enabled", 
                                     cfg.get("capture_gradients", True))
            
            # Get activation capture configuration
            capture_activations = kwargs.pop("capture_activations", 
                                 cfg.get("capture_activations", False))
            
            # Create callback
            callback = ParamLakeCallback(
                config=cfg,
                capture_frequency=capture_frequency,
                include_layers=include_layers,
                exclude_layers=exclude_layers,
                capture_gradients=capture_gradients,
                capture_activations=capture_activations
            )
            
            # Store the model reference in the callback BEFORE calling the function
            callback.model = None
            
            # Override Model.fit to include our callback
            original_fit = tf.keras.Model.fit
            
            def patched_fit(self, *fit_args, **fit_kwargs):
                # Add our callback to the callbacks list
                callbacks = fit_kwargs.get('callbacks', [])
                if callbacks is None:
                    callbacks = []
                elif not isinstance(callbacks, list):
                    callbacks = [callbacks]
                    
                # Set the model on our callback
                callback.model = self
                
                # Configure gradient collector before fitting
                if callback.capture_gradients and callback.gradient_collector and not callback._gradient_tracking_enabled:
                    try:
                        # Set up variable mapping
                        callback.gradient_collector.build_variable_mapping(self)
                        
                        # Try to establish gradient tracking
                        if callback.auto_gradient_tracking:
                            print("Ensuring gradient tracking is enabled before model.fit()")
                            callback._gradient_tracking_enabled = callback.gradient_collector.setup_automatic_gradient_tracking(self)
                            
                            if callback._gradient_tracking_enabled:
                                print("Successfully enabled gradient tracking for model.fit()")
                    except Exception as e:
                        print(f"Warning: Error setting up gradient tracking in patched_fit: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Add our callback to the list if not already there
                if callback not in callbacks:
                    callbacks.append(callback)
                    
                fit_kwargs['callbacks'] = callbacks
                return original_fit(self, *fit_args, **fit_kwargs)
            
            # Replace the fit method
            tf.keras.Model.fit = patched_fit
            
            # Declare result variable outside the try block
            result = None
            
            # Call the decorated function (the training function)
            try:
                # Inject the callback into the function call if it accepts it
                func_signature = inspect.signature(func)
                if 'paramlake_callback' in func_signature.parameters:
                    kwargs['paramlake_callback'] = callback
                
                # Execute the user's training function
                result = func(*args, **kwargs)
                
                # If the function returned a model (different from initial one), update callback
                if isinstance(result, tf.keras.Model) and result is not callback.model:
                    callback.model = result
                    # Capture initial state if model wasn't available before
                    if callback.model is not None:
                        callback.weight_collector.capture_model_weights(result, step=0)
                elif isinstance(result, tuple) or isinstance(result, list):
                    # Return tuples or lists directly
                    if len(result) > 0 and isinstance(result[0], tf.keras.Model):
                        callback.model = result[0]
                        # Make sure we capture data for this model 
                        callback.weight_collector.capture_model_weights(result[0], step=0)
                # If the function didn't return a model but we got one earlier, keep it
                elif callback.model is not None:
                    result = callback.model  # Return the model we found earlier
            finally:
                # Always close to ensure data is properly saved and final commit happens
                if callback.model is None and result is not None and isinstance(result, tf.keras.Model):
                    # If model was only available after func call, set it before closing
                    callback.model = result
                    # Make sure we capture data for this model
                    callback.weight_collector.capture_model_weights(result, step=0)
                
                # Ensure the storage manager has the final correct step before closing
                callback.storage_manager.set_step(callback.current_epoch)
                callback.close()
                
                # Restore original fit method
                tf.keras.Model.fit = original_fit
            
            return result
        
        return wrapper_paramlake
    
    # Handle case where decorator is used without arguments
    if func is not None:
        return decorator_paramlake(func)
    
    return decorator_paramlake


# Expose the decorator at the package level
__all__ = ["paramlake", "ParamLakeCallback"] 