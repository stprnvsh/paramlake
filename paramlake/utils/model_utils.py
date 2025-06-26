"""
Model utility functions for ParamLake.

This module provides utilities for working with models across different frameworks,
with a focus on TensorFlow/Keras models when available.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import subprocess
import platform

import numpy as np

from paramlake.utils.framework_utils import HAS_TENSORFLOW, require_tensorflow

# Optional TensorFlow import
if HAS_TENSORFLOW:
    import tensorflow as tf
else:
    tf = None


def get_all_layers(model: Any) -> List[Any]:
    """
    Get all layers from a model, handling nested models.
    
    Args:
        model: Model to extract layers from (TensorFlow model if TF is available)
        
    Returns:
        List of all layers
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    layers = []
    
    def _get_layers_recursive(layer_or_model):
        if hasattr(layer_or_model, 'layers'):
            # This is a model or a layer with sub-layers
            for sublayer in layer_or_model.layers:
                layers.append(sublayer)
                _get_layers_recursive(sublayer)
        else:
            # This is a regular layer
            layers.append(layer_or_model)
    
    _get_layers_recursive(model)
    return layers


def process_tensors_batch(storage, layer_group, tensor_data_pairs, tensor_type, step=None):
    """
    Process multiple tensors for a layer in batch to improve efficiency.
    
    Args:
        storage: Storage manager instance
        layer_group: Layer group to store tensors in
        tensor_data_pairs: List of (tensor_name, tensor_data) tuples
        tensor_type: Type of tensors (weights, gradients, activations)
        step: Current step
    """
    for tensor_name, tensor_data in tensor_data_pairs:
        try:
            # Convert TensorFlow tensors to numpy if needed
            if HAS_TENSORFLOW and hasattr(tensor_data, 'numpy'):
                tensor_data = tensor_data.numpy()
            elif not isinstance(tensor_data, np.ndarray):
                tensor_data = np.array(tensor_data)
            
            # Store the tensor
            storage.store_tensor(
                layer_group=layer_group,
                tensor_name=tensor_name,
                tensor_type=tensor_type,
                tensor_data=tensor_data,
                step=step
            )
        except Exception as e:
            print(f"Error processing tensor {tensor_name}: {e}")


def extract_model_parameters(model: Any, include_metadata: bool = True) -> Dict[str, Any]:
    """
    Extract all parameters from a model for storage.
    
    Args:
        model: Model to extract parameters from (TensorFlow model if TF is available)
        include_metadata: Whether to include model metadata
        
    Returns:
        Dictionary with model parameters and optionally metadata
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    parameters = {}
    
    # Extract layer weights
    for layer in model.layers:
        layer_weights = layer.get_weights()
        for i, weight in enumerate(layer_weights):
            param_name = f"{layer.name}/weight_{i}"
            parameters[param_name] = weight
    
    # Extract optimizer weights if available
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            optimizer_weights = model.optimizer.get_weights()
            for i, weight in enumerate(optimizer_weights):
                param_name = f"optimizer/weight_{i}"
                parameters[param_name] = weight
        except Exception as e:
            print(f"Warning: Could not extract optimizer parameters: {e}")
    
    # Include metadata if requested
    if include_metadata:
        parameters['_metadata'] = get_model_metadata(model)
    
    return parameters


def get_model_metadata(model: Any) -> Dict[str, Any]:
    """
    Extract metadata from a model.
    
    Args:
        model: Model to extract metadata from (TensorFlow model if TF is available)
        
    Returns:
        Dictionary with model metadata
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    metadata = {
        'model_class': model.__class__.__name__,
        'framework': 'tensorflow',
        'framework_version': tf.__version__,
        'total_params': model.count_params() if hasattr(model, 'count_params') else 0,
        'trainable_params': sum(1 for layer in model.layers for weight in layer.trainable_weights),
        'layers_count': len(model.layers),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add layer information
    layers_info = []
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'class': layer.__class__.__name__,
            'params': layer.count_params() if hasattr(layer, 'count_params') else 0,
            'trainable': layer.trainable,
        }
        
        # Add shapes if available
        try:
            if hasattr(layer, 'input_shape'):
                layer_info['input_shape'] = layer.input_shape
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = layer.output_shape
        except:
            pass
        
        layers_info.append(layer_info)
    
    metadata['layers'] = layers_info
    
    # Add optimizer info if available
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            optimizer_config = model.optimizer.get_config()
            metadata['optimizer'] = {
                'class': model.optimizer.__class__.__name__,
                'config': optimizer_config
            }
        except Exception as e:
            metadata['optimizer'] = {'error': str(e)}
    
    # Add compile info if available
    try:
        if hasattr(model, '_compile_config'):
            metadata['compile_config'] = model._compile_config
        elif hasattr(model, 'get_compile_config'):
            metadata['compile_config'] = model.get_compile_config()
    except Exception as e:
        metadata['compile_config'] = {'error': str(e)}
    
    return metadata


def compare_model_architectures(model1: Any, model2: Any) -> Dict[str, Any]:
    """
    Compare the architectures of two models.
    
    Args:
        model1: First model (TensorFlow model if TF is available)
        model2: Second model (TensorFlow model if TF is available)
        
    Returns:
        Dictionary with comparison results
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model1, tf.keras.Model) or not isinstance(model2, tf.keras.Model):
        raise ValueError("Both models must be TensorFlow/Keras models when TensorFlow is available")
    
    comparison = {
        'identical': True,
        'differences': [],
        'summary': {}
    }
    
    # Compare basic metrics
    meta1 = get_model_metadata(model1)
    meta2 = get_model_metadata(model2)
    
    for key in ['total_params', 'trainable_params', 'layers_count']:
        if meta1[key] != meta2[key]:
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'basic_metric',
                'metric': key,
                'model1': meta1[key],
                'model2': meta2[key]
            })
    
    # Compare layer by layer
    layers1 = model1.layers
    layers2 = model2.layers
    
    if len(layers1) != len(layers2):
        comparison['identical'] = False
        comparison['differences'].append({
            'type': 'layer_count',
            'model1': len(layers1),
            'model2': len(layers2)
        })
    
    # Compare individual layers
    min_layers = min(len(layers1), len(layers2))
    for i in range(min_layers):
        layer1, layer2 = layers1[i], layers2[i]
        
        if layer1.__class__.__name__ != layer2.__class__.__name__:
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'layer_class',
                'layer_index': i,
                'model1': layer1.__class__.__name__,
                'model2': layer2.__class__.__name__
            })
        
        if layer1.name != layer2.name:
            comparison['differences'].append({
                'type': 'layer_name',
                'layer_index': i,
                'model1': layer1.name,
                'model2': layer2.name
            })
        
        # Compare parameter counts
        params1 = layer1.count_params() if hasattr(layer1, 'count_params') else 0
        params2 = layer2.count_params() if hasattr(layer2, 'count_params') else 0
        
        if params1 != params2:
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'layer_params',
                'layer_index': i,
                'layer_name': layer1.name,
                'model1': params1,
                'model2': params2
            })
    
    # Summary
    comparison['summary'] = {
        'total_differences': len(comparison['differences']),
        'models_identical': comparison['identical'],
        'model1_layers': len(layers1),
        'model2_layers': len(layers2),
        'model1_params': meta1['total_params'],
        'model2_params': meta2['total_params']
    }
    
    return comparison


def load_model_from_parameters(
    model_template: Any, 
    parameters: Dict[str, np.ndarray],
    strict: bool = True
) -> Any:
    """
    Load parameters into a model from a parameter dictionary.
    
    Args:
        model_template: Template model to load parameters into (TensorFlow model if TF is available)
        parameters: Dictionary of parameter arrays
        strict: Whether to enforce strict parameter matching
        
    Returns:
        Model with loaded parameters
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model_template, tf.keras.Model):
        raise ValueError("Model template must be a TensorFlow/Keras model when TensorFlow is available")
    
    # Filter out metadata
    param_dict = {k: v for k, v in parameters.items() if not k.startswith('_')}
    
    # Group parameters by layer
    layer_params = {}
    optimizer_params = {}
    
    for param_name, param_data in param_dict.items():
        if param_name.startswith("optimizer/"):
            optimizer_params[param_name] = param_data
        else:
            # Extract layer name
            if "/" in param_name:
                layer_name = param_name.split("/")[0]
                if layer_name not in layer_params:
                    layer_params[layer_name] = {}
                layer_params[layer_name][param_name] = param_data
    
    # Load layer parameters
    loaded_count = 0
    failed_count = 0
    
    for layer in model_template.layers:
        if layer.name in layer_params:
            layer_weights = []
            
            # Collect weights for this layer in order
            for i in range(len(layer.get_weights())):
                weight_name = f"{layer.name}/weight_{i}"
                if weight_name in layer_params[layer.name]:
                    layer_weights.append(layer_params[layer.name][weight_name])
                else:
                    if strict:
                        raise ValueError(f"Weight {weight_name} not found in parameters")
                    else:
                        print(f"Warning: Weight {weight_name} not found")
                        failed_count += 1
                        break
            
            # Set weights if all were found
            if len(layer_weights) == len(layer.get_weights()):
                try:
                    layer.set_weights(layer_weights)
                    loaded_count += len(layer_weights)
                except Exception as e:
                    if strict:
                        raise ValueError(f"Failed to load weights for layer {layer.name}: {e}")
                    else:
                        print(f"Warning: Failed to load weights for layer {layer.name}: {e}")
                        failed_count += len(layer_weights)
    
    # Load optimizer parameters if available
    if optimizer_params and hasattr(model_template, 'optimizer') and model_template.optimizer is not None:
        try:
            optimizer_weights = []
            for i in range(len(optimizer_params)):
                weight_name = f"optimizer/weight_{i}"
                if weight_name in optimizer_params:
                    optimizer_weights.append(optimizer_params[weight_name])
            
            if optimizer_weights:
                model_template.optimizer.set_weights(optimizer_weights)
                print(f"Loaded optimizer parameters: {len(optimizer_weights)} weights")
        except Exception as e:
            print(f"Warning: Could not load optimizer parameters: {e}")
    
    print(f"Parameter loading complete: {loaded_count} loaded, {failed_count} failed")
    
    return model_template


def create_model_diff_summary(
    model1: Any, 
    model2: Any,
    parameter_changes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive diff summary between two models.
    
    Args:
        model1: First model (TensorFlow model if TF is available)
        model2: Second model (TensorFlow model if TF is available)
        parameter_changes: Optional pre-computed parameter changes
        
    Returns:
        Dictionary with diff summary
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model1, tf.keras.Model) or not isinstance(model2, tf.keras.Model):
        raise ValueError("Both models must be TensorFlow/Keras models when TensorFlow is available")
    
    diff_summary = {
        'timestamp': datetime.now().isoformat(),
        'architecture_comparison': compare_model_architectures(model1, model2),
        'parameter_changes': parameter_changes or {},
        'statistics': {}
    }
    
    # Calculate parameter statistics
    params1 = extract_model_parameters(model1, include_metadata=False)
    params2 = extract_model_parameters(model2, include_metadata=False)
    
    changed_params = 0
    total_params = 0
    param_norm_changes = []
    
    common_params = set(params1.keys()) & set(params2.keys())
    
    for param_name in common_params:
        p1, p2 = params1[param_name], params2[param_name]
        total_params += 1
        
        if not np.array_equal(p1, p2):
            changed_params += 1
            # Calculate L2 norm of change
            diff_norm = np.linalg.norm(p2 - p1)
            param_norm = np.linalg.norm(p1)
            relative_change = diff_norm / (param_norm + 1e-8)
            
            param_norm_changes.append({
                'name': param_name,
                'absolute_change': float(diff_norm),
                'relative_change': float(relative_change),
                'param_norm': float(param_norm)
            })
    
    # Sort by relative change magnitude
    param_norm_changes.sort(key=lambda x: x['relative_change'], reverse=True)
    
    diff_summary['statistics'] = {
        'total_parameters': total_params,
        'changed_parameters': changed_params,
        'unchanged_parameters': total_params - changed_params,
        'change_percentage': (changed_params / max(total_params, 1)) * 100,
        'top_changes': param_norm_changes[:10],  # Top 10 largest changes
        'added_parameters': len(set(params2.keys()) - set(params1.keys())),
        'removed_parameters': len(set(params1.keys()) - set(params2.keys()))
    }
    
    return diff_summary


def validate_model_compatibility(
    model: Any, 
    checkpoint_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that a model is compatible with checkpoint metadata.
    
    Args:
        model: Model to validate (TensorFlow model if TF is available)
        checkpoint_metadata: Metadata from a checkpoint
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    validation = {
        'compatible': True,
        'warnings': [],
        'errors': [],
        'details': {}
    }
    
    current_metadata = get_model_metadata(model)
    
    # Check basic compatibility
    checks = [
        ('total_params', 'Total parameter count'),
        ('layers_count', 'Number of layers'),
        ('framework', 'Framework'),
    ]
    
    for key, description in checks:
        if key in checkpoint_metadata:
            current_val = current_metadata.get(key)
            checkpoint_val = checkpoint_metadata.get(key)
            
            if current_val != checkpoint_val:
                if key in ['total_params', 'layers_count']:
                    validation['compatible'] = False
                    validation['errors'].append(
                        f"{description} mismatch: model has {current_val}, "
                        f"checkpoint has {checkpoint_val}"
                    )
                else:
                    validation['warnings'].append(
                        f"{description} mismatch: model has {current_val}, "
                        f"checkpoint has {checkpoint_val}"
                    )
    
    # Check layer compatibility
    if 'layers' in checkpoint_metadata:
        checkpoint_layers = checkpoint_metadata['layers']
        current_layers = current_metadata['layers']
        
        if len(current_layers) != len(checkpoint_layers):
            validation['compatible'] = False
            validation['errors'].append(
                f"Layer count mismatch: model has {len(current_layers)}, "
                f"checkpoint has {len(checkpoint_layers)}"
            )
        else:
            # Check individual layers
            for i, (current_layer, checkpoint_layer) in enumerate(zip(current_layers, checkpoint_layers)):
                if current_layer['class'] != checkpoint_layer['class']:
                    validation['compatible'] = False
                    validation['errors'].append(
                        f"Layer {i} class mismatch: model has {current_layer['class']}, "
                        f"checkpoint has {checkpoint_layer['class']}"
                    )
                
                if current_layer['params'] != checkpoint_layer['params']:
                    validation['compatible'] = False
                    validation['errors'].append(
                        f"Layer {i} ({current_layer['name']}) parameter count mismatch: "
                        f"model has {current_layer['params']}, "
                        f"checkpoint has {checkpoint_layer['params']}"
                    )
    
    # Framework version check
    if 'framework_version' in checkpoint_metadata:
        current_version = current_metadata.get('framework_version')
        checkpoint_version = checkpoint_metadata.get('framework_version')
        
        if current_version != checkpoint_version:
            validation['warnings'].append(
                f"Framework version mismatch: current {current_version}, "
                f"checkpoint {checkpoint_version}"
            )
    
    validation['details'] = {
        'current_metadata': current_metadata,
        'checkpoint_metadata': checkpoint_metadata,
        'total_errors': len(validation['errors']),
        'total_warnings': len(validation['warnings'])
    }
    
    return validation


def get_git_commit_metadata(
    model: Any,
    training_info: Optional[Dict[str, Any]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate git-style commit metadata for a model.
    
    Args:
        model: Model to generate metadata for (TensorFlow model if TF is available)
        training_info: Optional training information
        custom_metadata: Optional custom metadata to include
        
    Returns:
        Dictionary with git-style commit metadata
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()
    
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_metadata': get_model_metadata(model),
        'system_info': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'tensorflow_version': tf.__version__ if HAS_TENSORFLOW else None,
        }
    }
    
    # Add git information if available
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        ).strip()
        metadata['git_commit'] = git_commit
        
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        ).strip()
        metadata['git_branch'] = git_branch
        
        # Check for uncommitted changes
        git_status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            universal_newlines=True
        )
        metadata['git_has_uncommitted_changes'] = bool(git_status.strip())
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repository
        metadata['git_commit'] = None
        metadata['git_branch'] = None
        metadata['git_has_uncommitted_changes'] = None
    
    # Add training info if provided
    if training_info:
        metadata['training_info'] = training_info
    
    # Add custom metadata if provided
    if custom_metadata:
        metadata['custom'] = custom_metadata
    
    return metadata 