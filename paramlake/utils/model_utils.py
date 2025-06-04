"""
Shared utility functions for TensorFlow models with git-like version control support.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf


def get_all_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """
    Recursively get all layers in a model, including nested layers.
    
    Args:
        model: TensorFlow model
        
    Returns:
        List of all layers
    """
    all_layers = []
    
    for layer in model.layers:
        all_layers.append(layer)
        
        # If the layer is a model or sequential, get its layers
        if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
            all_layers.extend(get_all_layers(layer))
    
    return all_layers


def process_tensors_batch(storage, layer_group, tensor_data_pairs, tensor_type, step=None):
    """
    Process and store multiple tensors in a batch.
    
    Args:
        storage: Storage manager instance
        layer_group: Layer group to store tensors in
        tensor_data_pairs: List of (tensor_name, tensor_data) tuples
        tensor_type: Type of tensor (weights, gradients, activations)
        step: Current step or epoch (if None, uses storage's internal counter)
        
    Returns:
        Number of tensors successfully stored
    """
    successful_writes = 0
    
    # Process tensors in a batch
    for tensor_name, tensor_data in tensor_data_pairs:
        try:
            # Store tensor using storage interface
            storage.store_tensor(layer_group, tensor_name, tensor_type, tensor_data, step)
            successful_writes += 1
        except Exception as e:
            import traceback
            print(f"Error storing tensor {tensor_name}: {e}")
            traceback.print_exc()
    
    return successful_writes


def extract_model_parameters(model: tf.keras.Model, include_metadata: bool = True) -> Dict[str, Any]:
    """
    Extract model parameters for git-like commits.
    
    Args:
        model: TensorFlow model
        include_metadata: Whether to include model metadata
        
    Returns:
        Dictionary with model parameters and optional metadata
    """
    # Create a mapping of weight objects to their layer information
    weight_to_layer = {}
    for layer in model.layers:
        for weight in layer.weights:
            weight_to_layer[id(weight)] = layer.name
    
    params = {}
    for i, weight_var in enumerate(model.weights):
        # Create unique name using layer name and weight name
        layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
        weight_name = weight_var.name if weight_var.name else f"weight_{i}"
        
        # Combine layer name and weight name for uniqueness
        unique_name = f"{layer_name}/{weight_name}"
        params[unique_name] = weight_var.numpy()
    
    if include_metadata:
        result = {
            "parameters": params,
            "metadata": get_model_metadata(model)
        }
        return result
    else:
        return params


def get_model_metadata(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Extract comprehensive model metadata for version control.
    
    Args:
        model: TensorFlow model
        
    Returns:
        Dictionary with model metadata
    """
    metadata = {
        "architecture": {
            "num_layers": len(model.layers),
            "layer_types": {},
            "total_params": model.count_params(),
            "trainable_params": sum(tf.keras.utils.count_params(w) for w in model.trainable_weights),
            "non_trainable_params": sum(tf.keras.utils.count_params(w) for w in model.non_trainable_weights),
        },
        "compilation": {},
        "input_output": {},
    }
    
    # Layer type counts
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        metadata["architecture"]["layer_types"][layer_type] = metadata["architecture"]["layer_types"].get(layer_type, 0) + 1
    
    # Compilation info
    if hasattr(model, '_is_compiled') and model._is_compiled:
        metadata["compilation"] = {
            "optimizer": model.optimizer.__class__.__name__ if model.optimizer else None,
            "loss": str(model.loss) if model.loss else None,
            "metrics": [str(m) for m in model.metrics] if model.metrics else [],
        }
        
        # Optimizer config if available
        if model.optimizer:
            try:
                metadata["compilation"]["optimizer_config"] = model.optimizer.get_config()
            except:
                pass
    
    # Input/output shapes
    try:
        if hasattr(model, 'input_shape'):
            metadata["input_output"]["input_shape"] = model.input_shape
        if hasattr(model, 'output_shape'):
            metadata["input_output"]["output_shape"] = model.output_shape
    except:
        pass
    
    return metadata


def compare_model_architectures(model1: tf.keras.Model, model2: tf.keras.Model) -> Dict[str, Any]:
    """
    Compare architectures of two models for diff operations.
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Dictionary with comparison results
    """
    meta1 = get_model_metadata(model1)
    meta2 = get_model_metadata(model2)
    
    comparison = {
        "architectures_identical": True,
        "parameter_count_diff": meta2["architecture"]["total_params"] - meta1["architecture"]["total_params"],
        "layer_count_diff": meta2["architecture"]["num_layers"] - meta1["architecture"]["num_layers"],
        "differences": {},
    }
    
    # Compare layer counts
    layer_types1 = meta1["architecture"]["layer_types"]
    layer_types2 = meta2["architecture"]["layer_types"]
    
    all_layer_types = set(layer_types1.keys()) | set(layer_types2.keys())
    for layer_type in all_layer_types:
        count1 = layer_types1.get(layer_type, 0)
        count2 = layer_types2.get(layer_type, 0)
        if count1 != count2:
            comparison["architectures_identical"] = False
            comparison["differences"][f"{layer_type}_layers"] = {
                "model1": count1,
                "model2": count2,
                "diff": count2 - count1
            }
    
    # Compare compilation settings
    comp1 = meta1.get("compilation", {})
    comp2 = meta2.get("compilation", {})
    
    for key in ["optimizer", "loss"]:
        if comp1.get(key) != comp2.get(key):
            comparison["architectures_identical"] = False
            comparison["differences"][key] = {
                "model1": comp1.get(key),
                "model2": comp2.get(key)
            }
    
    return comparison


def load_model_from_parameters(
    model_template: tf.keras.Model, 
    parameters: Dict[str, np.ndarray],
    strict: bool = True
) -> tf.keras.Model:
    """
    Load parameters into a model template for git checkout operations.
    
    Args:
        model_template: Model to load parameters into
        parameters: Parameter dictionary from git storage
        strict: Whether to enforce strict parameter matching
        
    Returns:
        Model with loaded parameters
    """
    # Create mapping from weight objects to their layer information
    weight_to_layer = {}
    for layer in model_template.layers:
        for weight in layer.weights:
            weight_to_layer[id(weight)] = layer.name
    
    weights_to_set = []
    found_weights = {}
    
    # First pass: try exact name matching
    for i, weight_var in enumerate(model_template.weights):
        # Recreate the unique name using the same logic as extraction
        layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
        weight_name = weight_var.name if weight_var.name else f"weight_{i}"
        unique_name = f"{layer_name}/{weight_name}"
        
        # Try to find the parameter with the unique name
        sanitized_name = unique_name.replace("/", "_").replace(":", "_")
        
        param_data = None
        matched_key = None
        
        # Try multiple matching strategies
        for key_to_try in [unique_name, sanitized_name, weight_var.name]:
            if key_to_try in parameters:
                param_data = parameters[key_to_try]
                matched_key = key_to_try
                break
        
        if param_data is not None:
            # Verify shapes match to prevent mismatch errors
            if param_data.shape == weight_var.shape:
                weights_to_set.append(param_data)
                found_weights[unique_name] = True
            else:
                if strict:
                    raise ValueError(f"Shape mismatch for {unique_name}. Expected {weight_var.shape}, got {param_data.shape}")
                else:
                    print(f"Warning: Shape mismatch for {unique_name}. Using existing weights.")
                    weights_to_set.append(weight_var.numpy())
        else:
            if strict:
                raise ValueError(f"Parameter {unique_name} not found in checkpoint")
            else:
                weights_to_set.append(weight_var.numpy())
    
    # Set the weights
    if len(weights_to_set) == len(model_template.weights):
        model_template.set_weights(weights_to_set)
    else:
        raise ValueError(f"Weight count mismatch. Model has {len(model_template.weights)}, checkpoint has {len(weights_to_set)}")
    
    return model_template


def create_model_diff_summary(
    model1: tf.keras.Model, 
    model2: tf.keras.Model,
    parameter_changes: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive diff summary between two models.
    
    Args:
        model1: First model (baseline)
        model2: Second model (comparison)
        parameter_changes: Optional parameter change information
        
    Returns:
        Detailed diff summary
    """
    arch_comparison = compare_model_architectures(model1, model2)
    
    summary = {
        "architecture_changes": arch_comparison,
        "parameter_statistics": {},
        "training_impact": {},
    }
    
    if parameter_changes:
        summary["parameter_changes"] = parameter_changes
    
    # Calculate parameter statistics
    params1 = extract_model_parameters(model1, include_metadata=False)
    params2 = extract_model_parameters(model2, include_metadata=False)
    
    param_stats = {
        "total_parameters_changed": 0,
        "parameter_norm_diff": 0.0,
        "layer_changes": {}
    }
    
    common_params = set(params1.keys()) & set(params2.keys())
    for param_name in common_params:
        p1 = params1[param_name]
        p2 = params2[param_name]
        
        if p1.shape == p2.shape:
            diff = p2 - p1
            norm_diff = float(np.linalg.norm(diff))
            
            if norm_diff > 1e-8:  # Consider changed if difference is significant
                param_stats["total_parameters_changed"] += 1
                param_stats["parameter_norm_diff"] += norm_diff
                
                layer_name = param_name.split('/')[0]
                if layer_name not in param_stats["layer_changes"]:
                    param_stats["layer_changes"][layer_name] = {
                        "changed_tensors": 0,
                        "total_norm_diff": 0.0
                    }
                param_stats["layer_changes"][layer_name]["changed_tensors"] += 1
                param_stats["layer_changes"][layer_name]["total_norm_diff"] += norm_diff
    
    summary["parameter_statistics"] = param_stats
    
    return summary


def validate_model_compatibility(
    model: tf.keras.Model, 
    checkpoint_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate if a model is compatible with checkpoint metadata.
    
    Args:
        model: Model to validate
        checkpoint_metadata: Metadata from checkpoint
        
    Returns:
        Validation results
    """
    current_meta = get_model_metadata(model)
    
    validation = {
        "compatible": True,
        "warnings": [],
        "errors": [],
        "architecture_match": True,
        "compilation_match": True,
    }
    
    # Check architecture compatibility
    if "architecture" in checkpoint_metadata:
        ckpt_arch = checkpoint_metadata["architecture"]
        curr_arch = current_meta["architecture"]
        
        if ckpt_arch.get("num_layers") != curr_arch.get("num_layers"):
            validation["architecture_match"] = False
            validation["errors"].append(f"Layer count mismatch: checkpoint has {ckpt_arch.get('num_layers')}, model has {curr_arch.get('num_layers')}")
        
        if ckpt_arch.get("total_params") != curr_arch.get("total_params"):
            validation["architecture_match"] = False
            validation["errors"].append(f"Parameter count mismatch: checkpoint has {ckpt_arch.get('total_params')}, model has {curr_arch.get('total_params')}")
    
    # Check compilation compatibility
    if "compilation" in checkpoint_metadata:
        ckpt_comp = checkpoint_metadata["compilation"]
        curr_comp = current_meta["compilation"]
        
        if ckpt_comp.get("optimizer") != curr_comp.get("optimizer"):
            validation["compilation_match"] = False
            validation["warnings"].append(f"Optimizer mismatch: checkpoint has {ckpt_comp.get('optimizer')}, model has {curr_comp.get('optimizer')}")
        
        if ckpt_comp.get("loss") != curr_comp.get("loss"):
            validation["compilation_match"] = False
            validation["warnings"].append(f"Loss function mismatch: checkpoint has {ckpt_comp.get('loss')}, model has {curr_comp.get('loss')}")
    
    # Overall compatibility
    if validation["errors"]:
        validation["compatible"] = False
    
    return validation


def get_git_commit_metadata(
    model: tf.keras.Model,
    training_info: Optional[Dict[str, Any]] = None,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for git commits.
    
    Args:
        model: Model being committed
        training_info: Optional training information (epochs, loss, metrics, etc.)
        custom_metadata: Optional custom metadata
        
    Returns:
        Complete metadata for git commit
    """
    metadata = {
        "model": get_model_metadata(model),
        "timestamp": tf.timestamp().numpy().item(),
        "framework": {
            "name": "tensorflow",
            "version": tf.__version__,
        }
    }
    
    if training_info:
        metadata["training"] = training_info
    
    if custom_metadata:
        metadata["custom"] = custom_metadata
    
    return metadata 