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
from paramlake.collectors.optimizer_collector import OptimizerCollector
from paramlake.collectors.metrics_collector import MetricsCollector
from paramlake.storage.zarr_manager import ZarrStorageManager
from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.factory import create_storage_manager
from paramlake.storage.storage_interface import StorageInterface


class ParamLakeCallback(tf.keras.callbacks.Callback):
    """
    TensorFlow/Keras callback for automatic parameter tracking with git-like version control.
    """

    def __init__(
        self,
        storage_manager: StorageInterface,
        config: Optional[ParamLakeConfig] = None,
        collect_weights: bool = True,
        collect_gradients: bool = False,
        collect_activations: bool = False,
        collect_optimizer: bool = False,
        track_layers: Optional[List[str]] = None,
        ignore_layers: Optional[List[str]] = None,
    ):
        """
        Initialize the ParamLake callback.

        Args:
            storage_manager: Storage manager instance
            config: Configuration object
            collect_weights: Whether to collect weights
            collect_gradients: Whether to collect gradients
            collect_activations: Whether to collect activations
            collect_optimizer: Whether to collect optimizer state
            track_layers: Specific layers to track (if None, tracks all)
            ignore_layers: Layers to ignore
        """
        super().__init__()

        self.storage_manager = storage_manager
        self.config = config or ParamLakeConfig()
        
        # Initialize collectors
        self.collectors = []
        
        # Store collection configuration for later access
        self.collect_weights = collect_weights
        self.collect_gradients = collect_gradients
        self.collect_activations = collect_activations
        self.collect_optimizer = collect_optimizer
        
        # Store collector references for direct access
        self.weight_collector = None
        self.gradient_collector = None
        self.activation_collector = None
        self.optimizer_collector = None
        self.metrics_collector = None
        
        if collect_weights:
            # WeightCollector expects: storage_manager, config, track_layers, ignore_layers
            self.weight_collector = WeightCollector(
                storage_manager=storage_manager,
                config=config,
                track_layers=track_layers,
                ignore_layers=ignore_layers,
            )
            self.collectors.append(self.weight_collector)
            
        if collect_gradients:
            # GradientCollector expects: storage_manager, include_layers, exclude_layers, include_types
            self.gradient_collector = GradientCollector(
                storage_manager=storage_manager,
                include_layers=track_layers,  # Map track_layers to include_layers
                exclude_layers=ignore_layers,  # Map ignore_layers to exclude_layers
                include_types=None,  # Not provided by callback interface
            )
            self.collectors.append(self.gradient_collector)
            
        if collect_activations:
            # ActivationCollector expects: storage_manager, include_layers, exclude_layers, include_types
            self.activation_collector = ActivationCollector(
                storage_manager=storage_manager,
                include_layers=track_layers,  # Map track_layers to include_layers
                exclude_layers=ignore_layers,  # Map ignore_layers to exclude_layers
                include_types=None,  # Not provided by callback interface
            )
            self.collectors.append(self.activation_collector)
            
        if collect_optimizer:
            # OptimizerCollector expects: storage_manager only
            self.optimizer_collector = OptimizerCollector(
                storage_manager=storage_manager,
            )
            self.collectors.append(self.optimizer_collector)
            
        # Metrics collector is always included
        # MetricsCollector expects: storage_manager, enabled_metrics, include_layers, exclude_layers, capture_frequency
        metrics_config = config.get("metrics", {}) if config else {}
        self.metrics_collector = MetricsCollector(
            storage_manager=storage_manager,
            enabled_metrics=metrics_config.get("compute", ["l2", "mean", "var", "max", "min", "sparsity"]),
            include_layers=track_layers,  # Map track_layers to include_layers
            exclude_layers=ignore_layers,  # Map ignore_layers to exclude_layers
            capture_frequency=metrics_config.get("capture_frequency", 1),
        )
        self.collectors.append(self.metrics_collector)

        # Git integration settings
        self.auto_commit_enabled = config.get("git", {}).get("auto_commit", False) if config else False
        self.auto_tag_enabled = config.get("git", {}).get("auto_tag", False) if config else False
        self.commit_frequency = config.get("git", {}).get("commit_frequency", 10) if config else 10
        
        # Track the last epoch we committed to avoid duplicate commits
        self.last_commit_epoch = -1
        
        # Add attributes for gradient tracking compatibility
        self._gradient_tracking_enabled = False
        self.auto_gradient_tracking = collect_gradients
        self.current_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        for collector in self.collectors:
            if hasattr(collector, 'on_train_begin'):
                collector.on_train_begin(self.model, logs)
        
        # Set step for storage manager
        self.storage_manager.set_step(0)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        self.current_epoch = epoch  # Track current epoch
        
        for collector in self.collectors:
            if hasattr(collector, 'on_epoch_begin'):
                collector.on_epoch_begin(epoch, self.model, logs)
        
        # Update storage manager step
        self.storage_manager.set_step(epoch)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        self.current_epoch = epoch  # Update current epoch
        
        # Collect data from all collectors
        for collector in self.collectors:
            if hasattr(collector, 'on_epoch_end'):
                collector.on_epoch_end(epoch, self.model, logs)

        # Handle git-aware commits using centralized logic
        if hasattr(self.storage_manager, 'commit_if_needed'):
            # Use the centralized commit mechanism to avoid conflicts
            snapshot_id = self.storage_manager.commit_if_needed(
                step=epoch,
                force=False,  # Let the storage manager decide based on auto-commit settings
                message=f"Training progress at epoch {epoch + 1}"
            )
            
            if snapshot_id:
                self.last_commit_epoch = epoch
        elif self.auto_commit_enabled:
            # Fallback for storage managers without centralized commit logic
            epoch_num = epoch + 1
            if (epoch_num % self.commit_frequency == 0 and 
                epoch != self.last_commit_epoch):
                
                try:
                    snapshot_id = self.storage_manager.commit_changes(
                        f"Training progress at epoch {epoch_num}"
                    )
                    if snapshot_id:
                        self.last_commit_epoch = epoch
                        print(f"✓ Committed training progress at epoch {epoch_num}")
                        
                        # Auto-tag if enabled
                        if (self.auto_tag_enabled and 
                            hasattr(self.storage_manager, 'create_tag')):
                            tag_name = f"epoch_{epoch_num}"
                            try:
                                self.storage_manager.create_tag(
                                    tag_name, 
                                    snapshot_id, 
                                    f"Epoch {epoch_num} checkpoint"
                                )
                                print(f"✓ Tagged: {tag_name}")
                            except Exception as e:
                                print(f"Warning: Could not create tag {tag_name}: {e}")
                                
                except Exception as e:
                    print(f"Warning: Could not commit at epoch {epoch_num}: {e}")

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        for collector in self.collectors:
            if hasattr(collector, 'on_train_end'):
                collector.on_train_end(self.model, logs)

        # Final commit if there are uncommitted changes
        if hasattr(self.storage_manager, 'commit_if_needed'):
            # Force a final commit regardless of frequency
            snapshot_id = self.storage_manager.commit_if_needed(
                step=getattr(self, 'current_epoch', 0),
                force=True,
                message="Final training state"
            )
            if snapshot_id:
                print(f"✓ Final commit: {snapshot_id}")
        else:
            # Fallback commit
            try:
                snapshot_id = self.storage_manager.commit_changes("Final training state")
                if snapshot_id:
                    print(f"✓ Final commit: {snapshot_id}")
            except Exception as e:
                print(f"Warning: Could not perform final commit: {e}")

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        for collector in self.collectors:
            if hasattr(collector, 'on_batch_end'):
                collector.on_batch_end(batch, self.model, logs)

    def close(self) -> None:
        """Clean up resources and ensure data is saved."""
        try:
            # Close storage manager
            if hasattr(self.storage_manager, 'close'):
                self.storage_manager.close()
        except Exception as e:
            print(f"Warning: Error closing storage manager: {e}")


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
        
        self.optimizer_collector = None
        if config.get("capture_optimizer_state", True):
            self.optimizer_collector = OptimizerCollector(self.storage)
        
        # Create callback
        self.callback = ParamLakeCallback(
            storage_manager=self.storage,
            config=config,
            collect_weights=config["capture_weights"],
            collect_gradients=config["capture_gradients"],
            collect_activations=config["capture_activations"],
            collect_optimizer=config.get("capture_optimizer_state", True),
            track_layers=config["include_layers"],
            ignore_layers=config["exclude_layers"],
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
        # Store data for gradient capture
        x, y = data
        self.sample_data = x
        
        # Use gradient tape to capture gradients directly
        with tf.GradientTape() as tape:
            # Forward pass
            logits = self.model(x, training=True)
            # Compute loss
            loss = self.model.compiled_loss(y, logits, regularization_losses=self.model.losses)
        
        # Calculate gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Apply gradients with optimizer
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.model.compiled_metrics.update_state(y, logits)
        
        # Get current step
        step = self.storage.current_step
        
        # Store gradients directly if enabled
        if self.config["capture_gradients"] and step % self.callback.capture_frequency == 0:
            # Explicitly store gradients through the gradient collector
            try:
                print(f"Directly storing gradients at step {step}")
                self.gradient_collector.capture_gradients(gradients, self.model.trainable_variables, step)
            except Exception as e:
                import traceback
                print(f"Error directly storing gradients: {e}")
                traceback.print_exc()
        
        # Return dict with metrics
        return {m.name: m.result() for m in self.model.metrics}
    
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
        storage_type = config.get("storage_type", "zarr")
        
        if storage_type == "icechunk":
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
            
            # Get optimizer state capture configuration (not passed as kwarg directly to decorator usually)
            # It will be read from cfg by the ParamLakeCallback constructor
            capture_weights = cfg.get("capture_weights", True)
            capture_optimizer_state = cfg.get("capture_optimizer_state", True)
            
            # Create storage manager
            storage_type = cfg.get("storage_type", "zarr")
            
            if storage_type == "icechunk":
                try:
                    from paramlake.storage.icechunk_manager import IcechunkStorageManager
                    storage_manager = IcechunkStorageManager(cfg)
                except ImportError:
                    raise ImportError(
                        "Icechunk storage backend requested but Icechunk is not installed. "
                        "Install it with 'pip install icechunk'."
                    )
            else:
                from paramlake.storage.zarr_manager import ZarrStorageManager
                storage_manager = ZarrStorageManager(cfg)
            
            # Create callback
            callback = ParamLakeCallback(
                storage_manager=storage_manager,
                config=cfg,
                collect_weights=capture_weights,
                collect_gradients=capture_gradients,
                collect_activations=capture_activations,
                collect_optimizer=capture_optimizer_state,
                track_layers=include_layers,
                ignore_layers=exclude_layers,
            )
            
            # Store the model reference in the callback BEFORE calling the function
            # Note: callback.model is read-only, so we don't set it directly
            
            # Override Model.fit to include our callback
            original_fit = tf.keras.Model.fit
            
            def patched_fit(self, *fit_args, **fit_kwargs):
                # Add our callback to the callbacks list
                callbacks = fit_kwargs.get('callbacks', [])
                if callbacks is None:
                    callbacks = []
                elif not isinstance(callbacks, list):
                    callbacks = [callbacks]
                    
                # The model will be automatically set by Keras when the callback is used
                # No need to manually set callback.model as it's handled by Keras framework
                
                # Configure gradient collector before fitting
                if callback.collect_gradients and callback.gradient_collector and not callback._gradient_tracking_enabled:
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
                
                # If the function returned a model, update callback
                if isinstance(result, tf.keras.Model):
                    # Capture initial state if we have a weight collector
                    if callback.weight_collector is not None:
                        callback.weight_collector.capture_model_weights(result, step=0)
                elif isinstance(result, tuple) or isinstance(result, list):
                    # Return tuples or lists directly
                    if len(result) > 0 and isinstance(result[0], tf.keras.Model):
                        # Make sure we capture data for this model 
                        if callback.weight_collector is not None:
                            callback.weight_collector.capture_model_weights(result[0], step=0)
            finally:
                # Always close to ensure data is properly saved and final commit happens
                if result is not None and isinstance(result, tf.keras.Model):
                    # If model was only available after func call, capture it
                    if callback.weight_collector is not None:
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