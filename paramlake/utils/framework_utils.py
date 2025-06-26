"""
Framework detection and optional import utilities for ParamLake.

This module provides utilities to detect which ML frameworks are available
and handle optional imports gracefully.
"""
from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING
import warnings
import sys

# Framework availability flags
HAS_TENSORFLOW = False
HAS_TORCH = False
HAS_JAX = False
HAS_TRANSFORMERS = False

# Import tensorflow if available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
    
    # TensorFlow type hints
    if TYPE_CHECKING:
        TensorFlowModel = tf.keras.Model
        TensorFlowTensor = tf.Tensor
    else:
        TensorFlowModel = Any
        TensorFlowTensor = Any
        
except ImportError:
    tf = None
    TensorFlowModel = Any
    TensorFlowTensor = Any

# Import PyTorch if available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    
    # PyTorch type hints
    if TYPE_CHECKING:
        TorchModel = nn.Module
        TorchTensor = torch.Tensor
    else:
        TorchModel = Any
        TorchTensor = Any
        
except ImportError:
    torch = None
    nn = None
    TorchModel = Any
    TorchTensor = Any

# Import JAX if available  
try:
    import jax
    import jax.numpy as jnp
    import flax
    HAS_JAX = True
    
    # JAX type hints
    if TYPE_CHECKING:
        JAXArray = jax.Array
        FlaxModel = flax.linen.Module
    else:
        JAXArray = Any
        FlaxModel = Any
        
except ImportError:
    jax = None
    jnp = None
    flax = None
    JAXArray = Any
    FlaxModel = Any

# Import Transformers if available
try:
    import transformers
    HAS_TRANSFORMERS = True
    
    if TYPE_CHECKING:
        TransformersModel = transformers.PreTrainedModel
    else:
        TransformersModel = Any
        
except ImportError:
    transformers = None
    TransformersModel = Any


class FrameworkError(Exception):
    """Exception raised when required framework is not available."""
    pass


def require_tensorflow() -> Any:
    """
    Ensure TensorFlow is available, raise error if not.
    
    Returns:
        The tensorflow module
        
    Raises:
        FrameworkError: If TensorFlow is not installed
    """
    if not HAS_TENSORFLOW:
        raise FrameworkError(
            "TensorFlow is required for this operation. "
            "Install with: pip install 'paramlake[tf]' or pip install tensorflow"
        )
    return tf


def require_torch() -> Any:
    """
    Ensure PyTorch is available, raise error if not.
    
    Returns:
        The torch module
        
    Raises:
        FrameworkError: If PyTorch is not installed
    """
    if not HAS_TORCH:
        raise FrameworkError(
            "PyTorch is required for this operation. "
            "Install with: pip install 'paramlake[torch]' or pip install torch"
        )
    return torch


def require_jax() -> Any:
    """
    Ensure JAX is available, raise error if not.
    
    Returns:
        The jax module
        
    Raises:
        FrameworkError: If JAX is not installed
    """
    if not HAS_JAX:
        raise FrameworkError(
            "JAX is required for this operation. "
            "Install with: pip install 'paramlake[jax]' or pip install jax"
        )
    return jax


def require_transformers() -> Any:
    """
    Ensure Transformers is available, raise error if not.
    
    Returns:
        The transformers module
        
    Raises:
        FrameworkError: If Transformers is not installed
    """
    if not HAS_TRANSFORMERS:
        raise FrameworkError(
            "Transformers is required for this operation. "
            "Install with: pip install 'paramlake[hf]' or pip install transformers"
        )
    return transformers


def get_available_frameworks() -> List[str]:
    """
    Get list of available ML frameworks.
    
    Returns:
        List of available framework names
    """
    frameworks = []
    if HAS_TENSORFLOW:
        frameworks.append("tensorflow")
    if HAS_TORCH:
        frameworks.append("torch")
    if HAS_JAX:
        frameworks.append("jax")
    if HAS_TRANSFORMERS:
        frameworks.append("transformers")
    return frameworks


def detect_model_framework(model: Any) -> str:
    """
    Detect which framework a model belongs to.
    
    Args:
        model: The model to analyze
        
    Returns:
        Framework name ("tensorflow", "torch", "jax", "unknown")
    """
    if HAS_TENSORFLOW and isinstance(model, tf.keras.Model):
        return "tensorflow"
    elif HAS_TORCH and isinstance(model, nn.Module):
        return "torch"
    elif HAS_JAX and hasattr(model, '__class__') and 'flax' in str(type(model)):
        return "jax"
    elif HAS_TRANSFORMERS and hasattr(model, 'config') and hasattr(model, 'forward'):
        # Likely a Transformers model
        return "transformers"
    else:
        return "unknown"


def extract_model_parameters(model: Any) -> Dict[str, Any]:
    """
    Extract parameters from a model in a framework-agnostic way.
    
    Args:
        model: The model to extract parameters from
        
    Returns:
        Dictionary mapping parameter names to arrays
        
    Raises:
        FrameworkError: If model framework is not supported
    """
    framework = detect_model_framework(model)
    
    if framework == "tensorflow":
        tf = require_tensorflow()
        parameters = {}
        for layer in model.layers:
            layer_name = layer.name
            for weight_idx, weight in enumerate(layer.weights):
                weight_name = weight.name.split(':')[0]  # Remove :0 suffix
                param_name = f"{layer_name}/{weight_name}"
                parameters[param_name] = weight.numpy()
        return parameters
        
    elif framework == "torch":
        torch = require_torch()
        parameters = {}
        for name, param in model.named_parameters():
            parameters[name.replace('.', '/')] = param.detach().cpu().numpy()
        return parameters
        
    elif framework == "jax":
        jax = require_jax()
        # JAX models typically use pytrees for parameters
        if hasattr(model, 'params'):
            parameters = {}
            def flatten_params(params_dict, prefix=""):
                for key, value in params_dict.items():
                    full_key = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten_params(value, full_key)
                    else:
                        parameters[full_key] = jnp.asarray(value)
            flatten_params(model.params)
            return parameters
        else:
            raise FrameworkError(f"JAX model does not have recognizable parameter structure")
            
    else:
        raise FrameworkError(f"Unsupported model framework: {framework}")


def apply_parameters_to_model(model: Any, parameters: Dict[str, Any]) -> None:
    """
    Apply parameters to a model in a framework-agnostic way.
    
    Args:
        model: The model to apply parameters to
        parameters: Dictionary mapping parameter names to arrays
        
    Raises:
        FrameworkError: If model framework is not supported
    """
    framework = detect_model_framework(model)
    
    if framework == "tensorflow":
        tf = require_tensorflow()
        # Create mapping of full parameter names to layer weights
        weight_mapping = {}
        for layer in model.layers:
            layer_name = layer.name
            for weight_idx, weight in enumerate(layer.weights):
                weight_name = weight.name.split(':')[0]
                param_name = f"{layer_name}/{weight_name}"
                weight_mapping[param_name] = weight
        
        # Apply parameters
        for param_name, param_value in parameters.items():
            if param_name in weight_mapping:
                weight_mapping[param_name].assign(param_value)
            else:
                warnings.warn(f"Parameter {param_name} not found in model")
                
    elif framework == "torch":
        torch = require_torch()
        model_state = model.state_dict()
        
        # Convert parameter names back to PyTorch format
        for param_name, param_value in parameters.items():
            torch_name = param_name.replace('/', '.')
            if torch_name in model_state:
                model_state[torch_name] = torch.from_numpy(param_value)
            else:
                warnings.warn(f"Parameter {param_name} not found in model")
        
        model.load_state_dict(model_state)
        
    elif framework == "jax":
        jax = require_jax()
        # JAX parameter application is more complex and model-specific
        if hasattr(model, 'params'):
            # Reconstruct parameter tree
            def unflatten_params(flat_params):
                params = {}
                for key, value in flat_params.items():
                    parts = key.split('/')
                    current = params
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                return params
            
            model.params = unflatten_params(parameters)
        else:
            raise FrameworkError(f"JAX model does not support parameter application")
            
    else:
        raise FrameworkError(f"Unsupported model framework: {framework}")


def warn_missing_framework(framework: str) -> None:
    """
    Issue a warning about missing framework.
    
    Args:
        framework: Name of the missing framework
    """
    install_map = {
        "tensorflow": "pip install 'paramlake[tf]'",
        "torch": "pip install 'paramlake[torch]'", 
        "jax": "pip install 'paramlake[jax]'",
        "transformers": "pip install 'paramlake[hf]'"
    }
    
    install_cmd = install_map.get(framework, f"pip install {framework}")
    
    warnings.warn(
        f"{framework} is not available. Some functionality may be limited. "
        f"Install with: {install_cmd}",
        UserWarning,
        stacklevel=2
    )


def framework_info() -> Dict[str, Any]:
    """
    Get information about available frameworks.
    
    Returns:
        Dictionary with framework availability and versions
    """
    info = {
        "available": get_available_frameworks(),
        "versions": {},
        "support": {
            "tensorflow": HAS_TENSORFLOW,
            "torch": HAS_TORCH, 
            "jax": HAS_JAX,
            "transformers": HAS_TRANSFORMERS
        }
    }
    
    if HAS_TENSORFLOW:
        info["versions"]["tensorflow"] = tf.__version__
    if HAS_TORCH:
        info["versions"]["torch"] = torch.__version__
    if HAS_JAX:
        info["versions"]["jax"] = jax.__version__
    if HAS_TRANSFORMERS:
        info["versions"]["transformers"] = transformers.__version__
    
    return info


# Convenience functions for common operations
def safe_import_tensorflow():
    """Safely import TensorFlow with helpful error message."""
    try:
        return require_tensorflow()
    except FrameworkError as e:
        warn_missing_framework("tensorflow")
        return None


def safe_import_torch():
    """Safely import PyTorch with helpful error message."""
    try:
        return require_torch()
    except FrameworkError as e:
        warn_missing_framework("torch")
        return None


def safe_import_jax():
    """Safely import JAX with helpful error message."""
    try:
        return require_jax()
    except FrameworkError as e:
        warn_missing_framework("jax")
        return None 