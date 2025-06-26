"""
Optimizer collector for TensorFlow models.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
from paramlake.utils.framework_utils import HAS_TENSORFLOW, require_tensorflow

# Optional TensorFlow import
if HAS_TENSORFLOW:
    import tensorflow as tf
else:
    tf = None

from paramlake.storage.storage_interface import StorageInterface


class OptimizerCollector:
    """Collects optimizer state (weights and configuration) from TensorFlow models."""

    def __init__(
        self,
        storage_manager: StorageInterface,
    ):
        """
        Initialize optimizer collector.

        Args:
            storage_manager: Storage manager that implements StorageInterface.
        """
        self.storage = storage_manager

    def _tensor_to_numpy(self, tensor) -> np.ndarray:
        """
        Convert a TensorFlow tensor to a NumPy array.
        """
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        return np.array(tensor) # Fallback for non-TensorFlow eager tensors if any

    def capture_optimizer_state(
        self,
        optimizer,
        step: Optional[int] = None,
        optimizer_name: str = "optimizer"
    ) -> None:
        """
        Capture the optimizer's state (weights and configuration).

        Args:
            optimizer: TensorFlow optimizer instance.
            step: Current training step.
            optimizer_name: A name for the optimizer (e.g., "optimizer", "adam").
        """
        if not HAS_TENSORFLOW:
            require_tensorflow()
        
        tf = require_tensorflow()
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            print(f"Warning: Provided optimizer is not a tf.keras.optimizers.Optimizer instance. Skipping capture.")
            return

        # Capture optimizer weights (state)
        try:
            # Access optimizer state using the variables
            optimizer_weights = []
            
            # Different TF versions have different ways to access optimizer variables
            if hasattr(optimizer, 'variables'):
                opt_vars = optimizer.variables  # property, not a method
                # Check if it's a list or callable
                if callable(opt_vars):
                    opt_vars = opt_vars()
            elif hasattr(optimizer, 'weights'):
                opt_vars = optimizer.weights
            elif hasattr(optimizer, '_weights'):
                opt_vars = optimizer._weights
            elif hasattr(optimizer, '_var_list'):
                # Legacy approach for some TF versions
                opt_vars = optimizer._var_list
            else:
                # Last attempt - try direct attribute access on optimizer slots
                print(f"Warning: Using fallback method to get optimizer variables for {optimizer_name} at step {step}")
                opt_vars = []
                for attr_name in dir(optimizer):
                    if attr_name.startswith("_") and isinstance(getattr(optimizer, attr_name, None), dict):
                        slot_dict = getattr(optimizer, attr_name)
                        for var_list in slot_dict.values():
                            if isinstance(var_list, (list, tuple)):
                                opt_vars.extend(var_list)
            
            # Convert variables to numpy arrays
            for var in opt_vars:
                if var is None:
                    continue
                if hasattr(var, 'numpy'):
                    optimizer_weights.append(var.numpy())
                else:
                    optimizer_weights.append(np.array(var))
                    
            if optimizer_weights:
                # The existing store_optimizer_state handles the list of numpy arrays
                self.storage.store_optimizer_state(optimizer_weights, step=step)
                print(f"Successfully captured {len(optimizer_weights)} optimizer variables at step {step}")
            else:
                print(f"Optimizer {optimizer_name} reported no weights at step {step}.")
        except Exception as e:
            import traceback
            print(f"Error capturing optimizer weights for {optimizer_name} at step {step}: {e}")
            traceback.print_exc()

        # Capture optimizer configuration
        # We'll store this once, typically at the beginning or if it changes.
        # For simplicity in this collector, we'll call the storage method.
        # The storage manager can decide if it's a per-step or once-per-run storage.
        try:
            optimizer_config = optimizer.get_config()
            # Ensure config is JSON serializable
            serializable_config = self._make_serializable(optimizer_config)
            
            # The new store_optimizer_config method will handle this
            self.storage.store_optimizer_config(
                optimizer_name=optimizer_name, # Pass a name for the optimizer
                optimizer_config=serializable_config, 
                step=step # Pass step, storage can decide if it's needed
            )
        except Exception as e:
            import traceback
            print(f"Error capturing optimizer configuration for {optimizer_name} at step {step}: {e}")
            traceback.print_exc()

    def _make_serializable(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure that the optimizer configuration is JSON serializable.
        Converts non-serializable items like callables or complex objects to strings.
        """
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                if isinstance(value, list):
                    serializable_config[key] = [self._make_serializable_item(item) for item in value]
                elif isinstance(value, dict):
                    serializable_config[key] = self._make_serializable(value)
                else:
                    serializable_config[key] = value
            elif hasattr(value, '__name__'): # For functions or classes
                serializable_config[key] = f"<function_or_class:{value.__name__}>"
            elif callable(value):
                 serializable_config[key] = "<callable>"
            else:
                try:
                    # Attempt to convert to string as a fallback
                    json.dumps({key: value}) # Test serializability
                    serializable_config[key] = value
                except TypeError:
                    serializable_config[key] = str(value) # Fallback to string representation
        return serializable_config

    def _make_serializable_item(self, item: Any) -> Any:
        """Helper to make individual items in a list serializable."""
        if isinstance(item, (int, float, str, bool, list, dict, type(None))):
            if isinstance(item, list):
                return [self._make_serializable_item(sub_item) for sub_item in item]
            elif isinstance(item, dict):
                return self._make_serializable(item)
            return item
        elif hasattr(item, '__name__'):
            return f"<function_or_class:{item.__name__}>"
        elif callable(item):
            return "<callable>"
        else:
            try:
                json.dumps(item) # Test serializability
                return item
            except TypeError:
                return str(item) 