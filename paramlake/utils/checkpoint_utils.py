"""
Checkpoint utilities for ParamLake.

This module provides high-level utilities for saving and loading model checkpoints
with git-like version control capabilities.
"""

import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np

from paramlake.storage.storage_interface import StorageInterface
from paramlake.utils.framework_utils import HAS_TENSORFLOW, require_tensorflow

# Optional TensorFlow import
if HAS_TENSORFLOW:
    import tensorflow as tf
else:
    tf = None


def save_checkpoint(
    model: Any,  # Changed from tf.keras.Model to Any for compatibility
    storage_manager: StorageInterface,
    step: Optional[int] = None,
    include_optimizer: bool = True,
    name: Optional[str] = None,
    description: Optional[str] = None,
    create_tag: bool = False,
    tag_name: Optional[str] = None,
) -> str:
    """
    Save a model checkpoint with version control.
    
    Args:
        model: Model to save (TensorFlow model if TF is available)
        storage_manager: Storage manager instance
        step: Current training step
        include_optimizer: Whether to include optimizer state
        name: Custom checkpoint name
        description: Checkpoint description
        create_tag: Whether to create a tag for this checkpoint
        tag_name: Custom tag name
        
    Returns:
        Checkpoint ID/snapshot ID
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()  # This will raise the appropriate error
    
    # Verify model is TensorFlow model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    # Generate step if not provided
    if step is None:
        step = storage_manager.current_step
    
    # Extract model parameters
    weights_data = []
    weights_names = []
    weights_shapes = []
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        for i, weight in enumerate(layer_weights):
            weights_data.append(weight)
            weights_names.append(f"{layer.name}/weight_{i}")
            weights_shapes.append(weight.shape)
    
    # Extract optimizer state if requested
    optimizer_data = []
    optimizer_config = None
    
    if include_optimizer and hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            # Get optimizer weights
            optimizer_weights = model.optimizer.get_weights()
            optimizer_data = optimizer_weights
            
            # Get optimizer config
            optimizer_config = model.optimizer.get_config()
        except Exception as e:
            print(f"Warning: Could not extract optimizer state: {e}")
            optimizer_data = []
            optimizer_config = None
    
    # Get model compile configuration
    compile_config = None
    try:
        if hasattr(model, '_compile_config'):
            compile_config = model._compile_config
        elif hasattr(model, 'get_compile_config'):
            compile_config = model.get_compile_config()
    except Exception as e:
        print(f"Warning: Could not extract compile config: {e}")
    
    # Create metadata
    metadata = {
        'name': name or f"checkpoint_{step}",
        'description': description or f"Model checkpoint at step {step}",
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__,
        'total_params': model.count_params() if hasattr(model, 'count_params') else 0,
        'trainable_params': sum(1 for layer in model.layers for weight in layer.trainable_weights),
        'layers_count': len(model.layers),
        'framework': 'tensorflow',
        'framework_version': tf.__version__,
    }
    
    # Save checkpoint using storage manager
    checkpoint_id = storage_manager.save_checkpoint(
        weights_data=weights_data,
        weights_names=weights_names,
        weights_shapes=weights_shapes,
        optimizer_data=optimizer_data,
        optimizer_config=optimizer_config,
        compile_config=compile_config,
        metadata=metadata,
        step=step
    )
    
    # Create tag if requested
    if create_tag and hasattr(storage_manager, 'create_tag'):
        tag = tag_name or f"checkpoint_{step}"
        try:
            storage_manager.create_tag(tag, checkpoint_id)
            print(f"✓ Created tag: {tag}")
        except Exception as e:
            print(f"Warning: Could not create tag {tag}: {e}")
    
    return checkpoint_id


def load_checkpoint(
    model: Any,  # Changed from tf.keras.Model to Any for compatibility
    storage_manager: StorageInterface,
    checkpoint_id: str = None,
    step: int = None,
    by_name: bool = False,
    include_optimizer: bool = True,
    recompile: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load checkpoint into (TensorFlow model if TF is available)
        storage_manager: Storage manager instance
        checkpoint_id: Checkpoint ID to load
        step: Step number to load (alternative to checkpoint_id)
        by_name: Load weights by layer name matching
        include_optimizer: Whether to restore optimizer state
        recompile: Whether to recompile the model
        strict: Whether to enforce strict loading
        
    Returns:
        Dictionary with loading information
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()  # This will raise the appropriate error
    
    # Verify model is TensorFlow model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    # Determine which checkpoint to load
    if checkpoint_id is None and step is not None:
        checkpoint_data = storage_manager.load_checkpoint_by_step(step)
    elif checkpoint_id is not None:
        checkpoint_data = storage_manager.load_checkpoint(checkpoint_id)
    else:
        checkpoint_data = storage_manager.load_latest_checkpoint()
    
    # Extract checkpoint data
    weights_data = checkpoint_data["weights_data"]
    weights_names = checkpoint_data["weights_names"]
    weights_shapes = checkpoint_data["weights_shapes"]
    optimizer_data = checkpoint_data.get("optimizer_data", [])
    optimizer_config = checkpoint_data.get("optimizer_config")
    compile_config = checkpoint_data.get("compile_config")
    metadata = checkpoint_data.get("metadata", {})
    
    # Load weights into model
    loaded_weights = 0
    failed_weights = 0
    
    if by_name:
        # Load weights by matching layer names
        weight_dict = dict(zip(weights_names, weights_data))
        
        for layer in model.layers:
            layer_weights = []
            layer_found = False
            
            for i in range(len(layer.get_weights())):
                weight_name = f"{layer.name}/weight_{i}"
                if weight_name in weight_dict:
                    layer_weights.append(weight_dict[weight_name])
                    layer_found = True
                else:
                    if strict:
                        raise ValueError(f"Weight {weight_name} not found in checkpoint")
                    else:
                        print(f"Warning: Weight {weight_name} not found, skipping")
                        failed_weights += 1
                        break
            
            if layer_found and len(layer_weights) == len(layer.get_weights()):
                try:
                    layer.set_weights(layer_weights)
                    loaded_weights += len(layer_weights)
                except Exception as e:
                    if strict:
                        raise ValueError(f"Failed to load weights for layer {layer.name}: {e}")
                    else:
                        print(f"Warning: Failed to load weights for layer {layer.name}: {e}")
                        failed_weights += len(layer_weights)
    else:
        # Load weights sequentially
        weight_idx = 0
        for layer in model.layers:
            layer_weights = []
            num_layer_weights = len(layer.get_weights())
            
            if weight_idx + num_layer_weights <= len(weights_data):
                for i in range(num_layer_weights):
                    layer_weights.append(weights_data[weight_idx + i])
                
                try:
                    layer.set_weights(layer_weights)
                    loaded_weights += num_layer_weights
                    weight_idx += num_layer_weights
                except Exception as e:
                    if strict:
                        raise ValueError(f"Failed to load weights for layer {layer.name}: {e}")
                    else:
                        print(f"Warning: Failed to load weights for layer {layer.name}: {e}")
                        failed_weights += num_layer_weights
                        weight_idx += num_layer_weights
            else:
                if strict:
                    raise ValueError(f"Not enough weights in checkpoint for layer {layer.name}")
                else:
                    print(f"Warning: Not enough weights for layer {layer.name}")
                    failed_weights += num_layer_weights
                break
    
    # Restore optimizer state if requested
    optimizer_restored = False
    if include_optimizer and optimizer_data and len(optimizer_data) > 0:
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            try:
                # If optimizer config is available, create new optimizer
                if optimizer_config:
                    # Get optimizer class from config
                    optimizer_class = optimizer_config.get('class_name', 'Adam')
                    if hasattr(tf.keras.optimizers, optimizer_class):
                        optimizer_cls = getattr(tf.keras.optimizers, optimizer_class)
                        new_optimizer = optimizer_cls.from_config(optimizer_config)
                        model.compile(optimizer=new_optimizer)
                
                # Set optimizer weights
                model.optimizer.set_weights(optimizer_data)
                optimizer_restored = True
            except Exception as e:
                print(f"Warning: Could not restore optimizer state: {e}")
    
    # Recompile model if requested and compile config is available
    if recompile and compile_config:
        try:
            model.compile(**compile_config)
        except Exception as e:
            print(f"Warning: Could not recompile model: {e}")
    
    return {
        'checkpoint_id': checkpoint_id or metadata.get('name', 'unknown'),
        'step': metadata.get('step', 0),
        'loaded_weights': loaded_weights,
        'failed_weights': failed_weights,
        'optimizer_restored': optimizer_restored,
        'metadata': metadata,
        'success': failed_weights == 0
    }


def load_checkpoint_from_tag(
    model: Any,  # Changed from tf.keras.Model to Any for compatibility
    storage_manager: StorageInterface,
    tag_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Load a checkpoint from a tag.
    
    Args:
        model: Model to load checkpoint into (TensorFlow model if TF is available)
        storage_manager: Storage manager instance
        tag_name: Tag name to load from
        **kwargs: Additional arguments passed to load_checkpoint
        
    Returns:
        Dictionary with loading information
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()  # This will raise the appropriate error
    
    # Look up the snapshot ID from the tag
    if hasattr(storage_manager, 'list_tags'):
        tags = storage_manager.list_tags()
        if tag_name not in tags:
            raise ValueError(f"Tag '{tag_name}' not found")
        
        snapshot_id = tags[tag_name]
        return load_checkpoint(model, storage_manager, checkpoint_id=snapshot_id, **kwargs)
    else:
        raise NotImplementedError("Storage manager does not support tags")


def list_checkpoints(storage_manager: StorageInterface) -> List[Dict[str, Any]]:
    """
    List all available checkpoints.
    
    Args:
        storage_manager: Storage manager instance
        
    Returns:
        List of checkpoint metadata
    """
    return storage_manager.list_checkpoints()


def save_checkpoint_on_branch(
    model: Any,  # Changed from tf.keras.Model to Any for compatibility
    storage_manager: StorageInterface,
    branch_name: str,
    message: str,
    step: Optional[int] = None,
    author: Optional[str] = None,
    create_tag: bool = False,
    tag_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Save a checkpoint on a specific branch with git-like semantics.
    
    Args:
        model: Model to save (TensorFlow model if TF is available)
        storage_manager: Storage manager instance
        branch_name: Branch to save on
        message: Commit message
        step: Current training step
        author: Author name
        create_tag: Whether to create a tag
        tag_name: Custom tag name
        **kwargs: Additional arguments passed to save_checkpoint
        
    Returns:
        Snapshot ID
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()  # This will raise the appropriate error
    
    # Verify model is TensorFlow model
    if not isinstance(model, tf.keras.Model):
        raise ValueError("Model must be a TensorFlow/Keras model when TensorFlow is available")
    
    # Check if storage manager supports git-like operations
    if not hasattr(storage_manager, 'commit_model_state'):
        raise NotImplementedError("Storage manager does not support git-like operations")
    
    # Generate step if not provided
    if step is None:
        step = storage_manager.current_step
    
    # Extract model parameters
    model_parameters = {}
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        for i, weight in enumerate(layer_weights):
            param_name = f"{layer.name}/weight_{i}"
            model_parameters[param_name] = weight
    
    # Add optimizer parameters if available
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            optimizer_weights = model.optimizer.get_weights()
            for i, weight in enumerate(optimizer_weights):
                param_name = f"optimizer/weight_{i}"
                model_parameters[param_name] = weight
        except Exception as e:
            print(f"Warning: Could not extract optimizer parameters: {e}")
    
    # Create additional metadata
    additional_metadata = {
        'step': step,
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__,
        'total_params': model.count_params() if hasattr(model, 'count_params') else 0,
        'framework': 'tensorflow',
        'framework_version': tf.__version__,
    }
    
    # Add any custom metadata from kwargs
    for key, value in kwargs.items():
        if key not in ['include_optimizer', 'name', 'description']:
            additional_metadata[key] = value
    
    # Commit model state to branch
    snapshot_id = storage_manager.commit_model_state(
        model_parameters=model_parameters,
        message=message,
        branch_name=branch_name,
        author=author,
        additional_metadata=additional_metadata
    )
    
    # Create tag if requested
    if create_tag and hasattr(storage_manager, 'create_tag'):
        tag = tag_name or f"{branch_name}_{step}"
        try:
            storage_manager.create_tag(tag, snapshot_id)
            print(f"✓ Created tag: {tag}")
        except Exception as e:
            print(f"Warning: Could not create tag {tag}: {e}")
    
    return snapshot_id


def load_checkpoint_from_branch(
    model: Any,  # Changed from tf.keras.Model to Any for compatibility
    storage_manager: StorageInterface,
    branch_name: str,
    snapshot_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Load a checkpoint from a specific branch.
    
    Args:
        model: Model to load checkpoint into (TensorFlow model if TF is available)
        storage_manager: Storage manager instance
        branch_name: Branch name to load from
        snapshot_id: Specific snapshot ID (if None, uses branch HEAD)
        **kwargs: Additional arguments passed to load_checkpoint
        
    Returns:
        Dictionary with loading information
        
    Raises:
        ImportError: If TensorFlow is required but not available
    """
    if not HAS_TENSORFLOW:
        require_tensorflow()  # This will raise the appropriate error
    
    # Check if storage manager supports git-like operations
    if not hasattr(storage_manager, 'load_parameters_from_snapshot'):
        raise NotImplementedError("Storage manager does not support git-like operations")
    
    # Determine snapshot to load
    if snapshot_id is None:
        # Use branch HEAD
        if hasattr(storage_manager, 'list_branches'):
            branches = storage_manager.list_branches()
            if branch_name not in branches:
                raise ValueError(f"Branch '{branch_name}' not found")
        
        # For git-like systems, we'd get the HEAD of the branch
        # For now, assume the branch name can be used as reference
        reference = branch_name
    else:
        reference = snapshot_id
    
    # Load parameters from snapshot
    try:
        parameters = storage_manager.load_parameters_from_snapshot(reference)
    except Exception as e:
        raise ValueError(f"Could not load parameters from {reference}: {e}")
    
    # Load parameters into model
    loaded_params = 0
    failed_params = 0
    
    # Group parameters by layer
    layer_params = {}
    optimizer_params = {}
    
    for param_name, param_data in parameters.items():
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
    for layer in model.layers:
        if layer.name in layer_params:
            layer_weights = []
            
            # Collect weights for this layer in order
            for i in range(len(layer.get_weights())):
                weight_name = f"{layer.name}/weight_{i}"
                if weight_name in layer_params[layer.name]:
                    layer_weights.append(layer_params[layer.name][weight_name])
                else:
                    print(f"Warning: Weight {weight_name} not found in checkpoint")
                    failed_params += 1
                    break
            
            # Set weights if all were found
            if len(layer_weights) == len(layer.get_weights()):
                try:
                    layer.set_weights(layer_weights)
                    loaded_params += len(layer_weights)
                except Exception as e:
                    print(f"Warning: Failed to load weights for layer {layer.name}: {e}")
                    failed_params += len(layer_weights)
    
    # Load optimizer parameters if available
    optimizer_restored = False
    if optimizer_params and hasattr(model, 'optimizer') and model.optimizer is not None:
        try:
            optimizer_weights = []
            for i in range(len(optimizer_params)):
                weight_name = f"optimizer/weight_{i}"
                if weight_name in optimizer_params:
                    optimizer_weights.append(optimizer_params[weight_name])
            
            if optimizer_weights:
                model.optimizer.set_weights(optimizer_weights)
                optimizer_restored = True
        except Exception as e:
            print(f"Warning: Could not restore optimizer state: {e}")
    
    return {
        'branch_name': branch_name,
        'snapshot_id': snapshot_id or reference,
        'loaded_params': loaded_params,
        'failed_params': failed_params,
        'optimizer_restored': optimizer_restored,
        'success': failed_params == 0
    }


def create_checkpoint_branch(
    storage_manager: StorageInterface,
    branch_name: str,
    from_checkpoint: Optional[str] = None,
    from_step: Optional[int] = None
) -> str:
    """
    Create a new branch for checkpoint management.
    
    Args:
        storage_manager: Storage manager instance
        branch_name: Name of the new branch
        from_checkpoint: Checkpoint ID to branch from (optional)
        from_step: Step number to branch from (optional)
        
    Returns:
        Snapshot ID of the new branch HEAD
    """
    if not hasattr(storage_manager, 'create_branch'):
        raise NotImplementedError("Storage manager does not support branching")
    
    # Determine reference to branch from
    reference = None
    if from_checkpoint:
        reference = from_checkpoint
    elif from_step is not None:
        # Find checkpoint at the specified step
        checkpoints = list_checkpoints(storage_manager)
        for checkpoint in checkpoints:
            if checkpoint.get('step') == from_step:
                reference = checkpoint.get('id') or checkpoint.get('snapshot_id')
                break
        
        if reference is None:
            raise ValueError(f"No checkpoint found at step {from_step}")
    
    # Create the branch
    storage_manager.create_branch(branch_name, from_reference=reference)
    
    # Return the snapshot ID of the new branch
    if hasattr(storage_manager, 'get_snapshot_id_for_reference'):
        return storage_manager.get_snapshot_id_for_reference(branch_name)
    else:
        return branch_name  # Fallback


def compare_checkpoints(
    storage_manager: StorageInterface,
    checkpoint1: str,
    checkpoint2: str,
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Compare two checkpoints and return differences.
    
    Args:
        storage_manager: Storage manager instance
        checkpoint1: First checkpoint ID/reference
        checkpoint2: Second checkpoint ID/reference
        detailed: Whether to include detailed parameter differences
        
    Returns:
        Dictionary with comparison results
    """
    if hasattr(storage_manager, 'diff_snapshots'):
        return storage_manager.diff_snapshots(checkpoint1, checkpoint2)
    else:
        # Fallback comparison by loading both checkpoints
        try:
            data1 = storage_manager.load_checkpoint(checkpoint1)
            data2 = storage_manager.load_checkpoint(checkpoint2)
            
            # Compare metadata
            metadata_diff = {}
            meta1 = data1.get('metadata', {})
            meta2 = data2.get('metadata', {})
            
            for key in set(meta1.keys()) | set(meta2.keys()):
                val1 = meta1.get(key)
                val2 = meta2.get(key)
                if val1 != val2:
                    metadata_diff[key] = {'old': val1, 'new': val2}
            
            # Compare weights
            weights1 = data1.get('weights_data', [])
            weights2 = data2.get('weights_data', [])
            names1 = data1.get('weights_names', [])
            names2 = data2.get('weights_names', [])
            
            weight_diff = {
                'weights_count_changed': len(weights1) != len(weights2),
                'weights_names_changed': names1 != names2,
            }
            
            if detailed and len(weights1) == len(weights2):
                weight_diff['parameter_changes'] = []
                for i, (w1, w2) in enumerate(zip(weights1, weights2)):
                    if not np.array_equal(w1, w2):
                        weight_diff['parameter_changes'].append({
                            'index': i,
                            'name': names1[i] if i < len(names1) else f'weight_{i}',
                            'shape': w1.shape,
                            'changed': True
                        })
            
            return {
                'checkpoint1': checkpoint1,
                'checkpoint2': checkpoint2,
                'metadata_changes': metadata_diff,
                'weight_changes': weight_diff,
                'summary': {
                    'has_changes': bool(metadata_diff) or weight_diff.get('weights_count_changed', False) or weight_diff.get('weights_names_changed', False)
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'checkpoint1': checkpoint1,
                'checkpoint2': checkpoint2
            }


def get_checkpoint_history(
    storage_manager: StorageInterface,
    branch: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get checkpoint history for a branch.
    
    Args:
        storage_manager: Storage manager instance
        branch: Branch name (None for default/main)
        limit: Maximum number of entries to return
        
    Returns:
        List of checkpoint history entries
    """
    if hasattr(storage_manager, 'get_history'):
        return storage_manager.get_history(reference=branch, limit=limit)
    else:
        # Fallback to listing checkpoints
        checkpoints = list_checkpoints(storage_manager)
        
        # Sort by timestamp or step
        checkpoints.sort(
            key=lambda c: c.get('timestamp', c.get('step', 0)),
            reverse=True
        )
        
        if limit:
            checkpoints = checkpoints[:limit]
        
        return checkpoints


def merge_checkpoint_branches(
    storage_manager: StorageInterface,
    source_branch: str,
    target_branch: str,
    strategy: str = 'auto',
    message: Optional[str] = None
) -> str:
    """
    Merge checkpoint branches.
    
    Args:
        storage_manager: Storage manager instance
        source_branch: Source branch to merge from
        target_branch: Target branch to merge into
        strategy: Merge strategy ('auto', 'ours', 'theirs')
        message: Custom merge message
        
    Returns:
        Snapshot ID of merge commit
    """
    if not hasattr(storage_manager, 'merge_branches'):
        raise NotImplementedError("Storage manager does not support branch merging")
    
    merge_message = message or f"Merge branch '{source_branch}' into '{target_branch}'"
    
    return storage_manager.merge_branches(
        source_branch=source_branch,
        target_branch=target_branch,
        strategy=strategy,
        commit_message=merge_message
    )


def tag_checkpoint(
    storage_manager: StorageInterface,
    tag_name: str,
    checkpoint_reference: str,
    message: Optional[str] = None
) -> None:
    """
    Create a tag for a checkpoint.
    
    Args:
        storage_manager: Storage manager instance
        tag_name: Name of the tag
        checkpoint_reference: Checkpoint ID, branch name, or snapshot ID
        message: Optional tag message
    """
    if not hasattr(storage_manager, 'create_tag'):
        raise NotImplementedError("Storage manager does not support tagging")
    
    storage_manager.create_tag(tag_name, checkpoint_reference, message)


def list_checkpoint_tags(storage_manager: StorageInterface) -> Dict[str, str]:
    """
    List all checkpoint tags.
    
    Args:
        storage_manager: Storage manager instance
        
    Returns:
        Dictionary mapping tag names to snapshot IDs
    """
    if hasattr(storage_manager, 'list_tags'):
        return storage_manager.list_tags()
    else:
        return {}


def _get_model_summary(model: Any) -> Dict[str, Any]:
    """
    Get a summary of model architecture (TensorFlow-specific).
    
    Args:
        model: Model to summarize (TensorFlow model if TF is available)
        
    Returns:
        Dictionary with model summary
    """
    if not HAS_TENSORFLOW:
        return {'error': 'TensorFlow not available'}
    
    if not isinstance(model, tf.keras.Model):
        return {'error': 'Not a TensorFlow model'}
    
    summary = {
        'total_params': model.count_params() if hasattr(model, 'count_params') else 0,
        'trainable_params': sum(1 for layer in model.layers for weight in layer.trainable_weights),
        'layers_count': len(model.layers),
        'model_class': model.__class__.__name__,
    }
    
    # Add layer information
    layers_info = []
    for layer in model.layers:
        layer_info = {
            'name': layer.name,
            'class': layer.__class__.__name__,
            'params': layer.count_params() if hasattr(layer, 'count_params') else 0,
        }
        
        # Add input/output shapes if available
        try:
            if hasattr(layer, 'input_shape'):
                layer_info['input_shape'] = layer.input_shape
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = layer.output_shape
        except:
            pass
        
        layers_info.append(layer_info)
    
    summary['layers'] = layers_info
    
    return summary 