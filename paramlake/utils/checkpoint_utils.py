"""
Checkpoint utilities for saving and loading model state with git-like version control integration.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from paramlake.storage.storage_interface import StorageInterface


def save_checkpoint(
    model: tf.keras.Model,
    storage_manager: StorageInterface,
    step: Optional[int] = None,
    include_optimizer: bool = True,
    name: Optional[str] = None,
    description: Optional[str] = None,
    create_tag: bool = False,
    tag_name: Optional[str] = None,
) -> str:
    """
    Save a checkpoint of model weights and optimizer state.
    
    Args:
        model: TensorFlow model to save
        storage_manager: Storage manager instance
        step: Step number (if None, uses storage manager's current step)
        include_optimizer: Whether to include optimizer state
        name: Name for the checkpoint (if None, uses "checkpoint_{step}")
        description: Optional description of the checkpoint
        create_tag: Whether to create a Git-like tag for this checkpoint
        tag_name: Name for the tag (if None and create_tag=True, uses checkpoint name)
        
    Returns:
        Checkpoint ID or snapshot ID
    """
    # Use provided step or current step
    current_step = step if step is not None else storage_manager.current_step
    
    # Generate checkpoint name if not provided
    checkpoint_name = name if name is not None else f"checkpoint_{current_step}"
    
    # Create a dictionary with metadata
    metadata = {
        "name": checkpoint_name,
        "step": current_step,
        "description": description,
        "timestamp": None,  # Will be filled by storage manager
        "include_optimizer": include_optimizer,
        "framework": "tensorflow",
        "model_summary": _get_model_summary(model),
    }
    
    # Save the model weights
    # We convert to a simple list of numpy arrays for storage
    weights_data = []
    weights_shapes = []
    weights_names = []
    
    for layer in model.layers:
        for weight in layer.weights:
            weights_data.append(weight.numpy())
            weights_shapes.append(weight.shape)
            weights_names.append(weight.name)
    
    # Save optimizer state if requested
    optimizer_data = []
    optimizer_config = None
    
    if include_optimizer and hasattr(model, 'optimizer') and model.optimizer is not None:
        optimizer = model.optimizer
        
        # Get optimizer config
        optimizer_config = optimizer.get_config()
        
        # Get optimizer weights
        if hasattr(optimizer, 'weights') and optimizer.weights:
            for weight in optimizer.weights:
                optimizer_data.append(weight.numpy())
    
    # Compile configuration if model is compiled
    compile_config = None
    if hasattr(model, '_is_compiled') and model._is_compiled:
        compile_config = {
            'optimizer_config': optimizer_config,
            'loss': model.loss,
            'metrics': [m.name if hasattr(m, 'name') else m for m in model.metrics],
            'weighted_metrics': model.weighted_metrics,
            'loss_weights': model.loss_weights,
        }
        
        # Add any other necessary compile information
        if hasattr(model, 'sample_weight_mode'):
            compile_config['sample_weight_mode'] = model.sample_weight_mode
        if hasattr(model, 'target_tensors'):
            compile_config['target_tensors'] = model.target_tensors
    
    # Call the storage manager to save the checkpoint
    checkpoint_id = storage_manager.save_checkpoint(
        weights_data=weights_data,
        weights_names=weights_names,
        weights_shapes=weights_shapes,
        optimizer_data=optimizer_data,
        optimizer_config=optimizer_config,
        compile_config=compile_config,
        metadata=metadata,
        step=current_step,
    )
    
    # Create a Git-like tag if requested and supported
    if create_tag and hasattr(storage_manager, 'create_tag'):
        try:
            final_tag_name = tag_name or checkpoint_name
            storage_manager.create_tag(final_tag_name, checkpoint_id)
            print(f"Created tag '{final_tag_name}' for checkpoint")
        except Exception as e:
            print(f"Warning: Could not create tag for checkpoint: {e}")
    
    return checkpoint_id


def load_checkpoint(
    model: tf.keras.Model,
    storage_manager: StorageInterface,
    checkpoint_id: str = None,
    step: int = None,
    by_name: bool = False,
    include_optimizer: bool = True,
    recompile: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a checkpoint into a model.
    
    Args:
        model: TensorFlow model to load weights into
        storage_manager: Storage manager instance
        checkpoint_id: ID of the checkpoint to load (ignored if step is provided)
        step: Step number to load (if provided, loads the checkpoint from this step)
        by_name: Whether to load weights by name instead of order
        include_optimizer: Whether to load optimizer state if available
        recompile: Whether to recompile the model with saved compile config
        strict: Whether to enforce strict loading (fail on mismatches)
        
    Returns:
        Dictionary with metadata about the loaded checkpoint
    """
    # Determine which checkpoint to load
    if step is not None:
        checkpoint_data = storage_manager.load_checkpoint_by_step(step)
    elif checkpoint_id is not None:
        checkpoint_data = storage_manager.load_checkpoint(checkpoint_id)
    else:
        # Load the latest checkpoint if neither is specified
        checkpoint_data = storage_manager.load_latest_checkpoint()
    
    if checkpoint_data is None:
        raise ValueError("No checkpoint found")
    
    # Extract checkpoint data
    weights_data = checkpoint_data.get('weights_data', [])
    weights_names = checkpoint_data.get('weights_names', [])
    optimizer_data = checkpoint_data.get('optimizer_data', [])
    optimizer_config = checkpoint_data.get('optimizer_config', None)
    compile_config = checkpoint_data.get('compile_config', None)
    metadata = checkpoint_data.get('metadata', {})
    
    # Validate model compatibility if strict mode
    if strict and 'model_summary' in metadata:
        current_summary = _get_model_summary(model)
        if current_summary != metadata['model_summary']:
            print(f"Warning: Model architecture mismatch detected")
            print(f"Current: {current_summary}")
            print(f"Checkpoint: {metadata['model_summary']}")
            if strict:
                raise ValueError("Model architecture mismatch - use strict=False to force loading")
    
    # Load weights into the model
    if by_name:
        # Create a dict mapping names to weights
        named_weights = {name: weight for name, weight in zip(weights_names, weights_data)}
        
        # Assign weights by name
        for layer in model.layers:
            for weight in layer.weights:
                name = weight.name
                if name in named_weights:
                    try:
                        weight.assign(named_weights[name])
                    except Exception as e:
                        print(f"Error assigning weight {name}: {e}")
                        if strict:
                            raise
    else:
        # Assign weights directly
        # This assumes the model structure is exactly the same as when saved
        if len(weights_data) == 0:
            print("Warning: No weights found in checkpoint")
        elif len(weights_data) != len(model.weights):
            print(f"Warning: Checkpoint has {len(weights_data)} weights, but model has {len(model.weights)} weights")
            if strict:
                raise ValueError("Weight count mismatch - use by_name=True or strict=False")
            print("Will attempt to set weights directly from checkpoint data")
            model.set_weights(weights_data)
        else:
            # Assign weights directly when counts match
            model.set_weights(weights_data)
    
    # Load optimizer state if available and requested
    if include_optimizer and optimizer_data and optimizer_config:
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            optimizer = model.optimizer
            
            # Check if optimizer class matches
            if optimizer.__class__.__name__ == optimizer_config.get('name', ''):
                # Set weights directly if possible
                if hasattr(optimizer, 'set_weights') and callable(optimizer.set_weights):
                    try:
                        optimizer.set_weights(optimizer_data)
                    except Exception as e:
                        print(f"Warning: Failed to set optimizer weights: {e}")
                        
                # Alternative method: try to set weights one by one
                elif hasattr(optimizer, 'weights') and len(optimizer.weights) == len(optimizer_data):
                    for weight, value in zip(optimizer.weights, optimizer_data):
                        weight.assign(value)
    
    # Recompile model if requested and compile config is available
    if recompile and compile_config and hasattr(model, 'compile'):
        # Extract necessary compile parameters
        optimizer = model.optimizer  # Keep existing optimizer with restored weights
        loss = compile_config.get('loss')
        metrics = compile_config.get('metrics', [])
        loss_weights = compile_config.get('loss_weights')
        weighted_metrics = compile_config.get('weighted_metrics')
        
        # Recompile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics
        )
    
    return metadata


def load_checkpoint_from_tag(
    model: tf.keras.Model,
    storage_manager: StorageInterface,
    tag_name: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Load a checkpoint using a Git-like tag.
    
    Args:
        model: TensorFlow model to load weights into
        storage_manager: Storage manager instance
        tag_name: Name of the tag pointing to the checkpoint
        **kwargs: Additional arguments passed to load_checkpoint
        
    Returns:
        Dictionary with metadata about the loaded checkpoint
    """
    # Resolve tag to snapshot/checkpoint ID
    if hasattr(storage_manager, 'get_snapshot_id_for_reference'):
        checkpoint_id = storage_manager.get_snapshot_id_for_reference(tag_name)
        if not checkpoint_id:
            raise ValueError(f"Tag '{tag_name}' not found")
    else:
        # Fallback: assume tag_name is the checkpoint_id
        checkpoint_id = tag_name
    
    return load_checkpoint(model, storage_manager, checkpoint_id=checkpoint_id, **kwargs)


def list_checkpoints(storage_manager: StorageInterface) -> List[Dict[str, Any]]:
    """
    List all available checkpoints.
    
    Args:
        storage_manager: Storage manager instance
        
    Returns:
        List of dictionaries with checkpoint metadata
    """
    return storage_manager.list_checkpoints()


# Git-aware checkpoint functions

def save_checkpoint_on_branch(
    model: tf.keras.Model,
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
    Save a checkpoint on a specific branch with git-like commit.
    
    Args:
        model: TensorFlow model to save
        storage_manager: Storage manager instance
        branch_name: Branch to save checkpoint on
        message: Commit message
        step: Step number (if None, uses storage manager's current step)
        author: Author of the checkpoint
        create_tag: Whether to create a tag for this checkpoint
        tag_name: Name of the tag (if create_tag is True)
        **kwargs: Additional arguments for save_checkpoint
        
    Returns:
        Checkpoint snapshot ID
    """
    # Switch to the target branch if we have git features
    if hasattr(storage_manager, 'switch_branch'):
        storage_manager.switch_branch(branch_name, create_if_missing=True)
    
    # Extract model parameters for git commit
    from paramlake.utils.model_utils import extract_model_parameters, get_git_commit_metadata
    
    model_params = extract_model_parameters(model, include_metadata=False)
    training_info = {
        "step": step if step is not None else storage_manager.current_step,
        "checkpoint_message": message
    }
    commit_metadata = get_git_commit_metadata(model, training_info=training_info)
    
    # Commit the model state if git features are available
    if hasattr(storage_manager, 'commit_model_state'):
        snapshot_id = storage_manager.commit_model_state(
            model_parameters=model_params,
            message=message,
            branch_name=branch_name,
            author=author,
            additional_metadata=commit_metadata
        )
        
        # Create tag if requested
        if create_tag and hasattr(storage_manager, 'create_tag'):
            try:
                final_tag_name = tag_name or f"checkpoint_{step or storage_manager.current_step}"
                storage_manager.create_tag(final_tag_name, snapshot_id)
                print(f"Created tag '{final_tag_name}' for checkpoint on branch '{branch_name}'")
            except Exception as e:
                print(f"Warning: Could not create tag: {e}")
        
        return snapshot_id
    else:
        # Fallback to regular checkpoint saving
        return save_checkpoint(
            model, storage_manager, step=step, 
            create_tag=create_tag, tag_name=tag_name, **kwargs
        )


def load_checkpoint_from_branch(
    model: tf.keras.Model,
    storage_manager: StorageInterface,
    branch_name: str,
    snapshot_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Load a checkpoint from a specific branch.
    
    Args:
        model: TensorFlow model to load weights into
        storage_manager: Storage manager instance
        branch_name: Branch to load checkpoint from
        snapshot_id: Specific snapshot ID on the branch (if None, uses latest)
        **kwargs: Additional arguments for load_checkpoint
        
    Returns:
        Dictionary with metadata about the loaded checkpoint
    """
    # Get the snapshot ID for the branch
    if snapshot_id is None and hasattr(storage_manager, 'get_snapshot_id_for_reference'):
        snapshot_id = storage_manager.get_snapshot_id_for_reference(branch_name)
        if not snapshot_id:
            raise ValueError(f"Branch '{branch_name}' not found")
    
    # Load parameters from the snapshot if git features are available
    if hasattr(storage_manager, 'load_parameters_from_snapshot') and snapshot_id:
        from paramlake.utils.model_utils import load_model_from_parameters
        
        parameters = storage_manager.load_parameters_from_snapshot(snapshot_id)
        load_model_from_parameters(model, parameters, strict=kwargs.get('strict', True))
        
        # Return metadata
        return {
            "branch": branch_name,
            "snapshot_id": snapshot_id,
            "loaded_from": "git_snapshot"
        }
    else:
        # Fallback to regular checkpoint loading
        if snapshot_id:
            return load_checkpoint(model, storage_manager, checkpoint_id=snapshot_id, **kwargs)
        else:
            return load_checkpoint(model, storage_manager, **kwargs)


def create_checkpoint_branch(
    storage_manager: StorageInterface,
    branch_name: str,
    from_checkpoint: Optional[str] = None,
    from_step: Optional[int] = None
) -> str:
    """
    Create a new branch from a checkpoint for experimental training.
    
    Args:
        storage_manager: Storage manager instance
        branch_name: Name of the new branch
        from_checkpoint: Checkpoint ID to branch from (if None, uses current HEAD)
        from_step: Step number to branch from (if None, uses current HEAD)
        
    Returns:
        Snapshot ID of the branch point
    """
    if not hasattr(storage_manager, 'create_branch'):
        raise NotImplementedError("Git features not available in this storage manager")
    
    # Determine the reference to branch from
    from_reference = None
    if from_checkpoint:
        from_reference = from_checkpoint
    elif from_step is not None:
        # Try to find a checkpoint at this step
        checkpoints = list_checkpoints(storage_manager)
        for checkpoint in checkpoints:
            if checkpoint.get('step') == from_step:
                from_reference = checkpoint.get('snapshot_id') or checkpoint.get('id')
                break
        if not from_reference:
            raise ValueError(f"No checkpoint found at step {from_step}")
    
    # Create the branch
    storage_manager.create_branch(branch_name, from_reference=from_reference)
    
    # Return the snapshot ID we branched from
    if from_reference:
        return from_reference
    else:
        # Get current HEAD if no specific reference was used
        return storage_manager.get_snapshot_id_for_reference("main")


def compare_checkpoints(
    storage_manager: StorageInterface,
    checkpoint1: str,
    checkpoint2: str,
    detailed: bool = False
) -> Dict[str, Any]:
    """
    Compare two checkpoints using git-like diff.
    
    Args:
        storage_manager: Storage manager instance
        checkpoint1: First checkpoint reference (ID, tag, or branch)
        checkpoint2: Second checkpoint reference (ID, tag, or branch)
        detailed: Whether to include detailed parameter differences
        
    Returns:
        Comparison results
    """
    if not hasattr(storage_manager, 'diff_snapshots'):
        raise NotImplementedError("Diff features not available in this storage manager")
    
    # Resolve checkpoint references to snapshot IDs
    if hasattr(storage_manager, 'get_snapshot_id_for_reference'):
        snap1 = storage_manager.get_snapshot_id_for_reference(checkpoint1)
        snap2 = storage_manager.get_snapshot_id_for_reference(checkpoint2)
    else:
        snap1, snap2 = checkpoint1, checkpoint2
    
    # Perform diff
    diff_result = storage_manager.diff_snapshots(snap1, snap2)
    
    if not detailed:
        # Return simplified summary
        return {
            "checkpoint1": checkpoint1,
            "checkpoint2": checkpoint2,
            "summary": diff_result.get("summary", {}),
            "metadata_changes": len(diff_result.get("metadata_changes", {})),
            "layer_changes": len(diff_result.get("layer_changes", {})),
        }
    
    return diff_result


def get_checkpoint_history(
    storage_manager: StorageInterface,
    branch: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get the history of checkpoints on a branch.
    
    Args:
        storage_manager: Storage manager instance
        branch: Branch name (if None, uses current branch)
        limit: Maximum number of checkpoints to return
        
    Returns:
        List of checkpoint history entries
    """
    if not hasattr(storage_manager, 'get_history'):
        # Fallback to listing regular checkpoints
        return list_checkpoints(storage_manager)
    
    return storage_manager.get_history(reference=branch, limit=limit)


def merge_checkpoint_branches(
    storage_manager: StorageInterface,
    source_branch: str,
    target_branch: str,
    strategy: str = 'auto',
    message: Optional[str] = None
) -> str:
    """
    Merge checkpoints from one branch into another.
    
    Args:
        storage_manager: Storage manager instance
        source_branch: Branch to merge from
        target_branch: Branch to merge into
        strategy: Merge strategy ('auto', 'ours', 'theirs')
        message: Merge commit message
        
    Returns:
        Snapshot ID of the merge commit
    """
    if not hasattr(storage_manager, 'merge_branches'):
        raise NotImplementedError("Merge features not available in this storage manager")
    
    merge_message = message or f"Merge checkpoint branch '{source_branch}' into '{target_branch}'"
    
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
        checkpoint_reference: Checkpoint ID, branch, or other reference
        message: Optional tag message
    """
    if not hasattr(storage_manager, 'create_tag'):
        raise NotImplementedError("Tagging features not available in this storage manager")
    
    storage_manager.create_tag(tag_name, checkpoint_reference, message=message)


def list_checkpoint_tags(storage_manager: StorageInterface) -> Dict[str, str]:
    """
    List all checkpoint tags.
    
    Args:
        storage_manager: Storage manager instance
        
    Returns:
        Dictionary mapping tag names to snapshot IDs
    """
    if not hasattr(storage_manager, 'list_tags'):
        return {}
    
    return storage_manager.list_tags()


def _get_model_summary(model: tf.keras.Model) -> Dict[str, Any]:
    """
    Get a summary of the model architecture for compatibility checking.
    
    Args:
        model: TensorFlow model
        
    Returns:
        Dictionary with model summary information
    """
    try:
        summary = {
            "num_layers": len(model.layers),
            "num_parameters": model.count_params(),
            "input_shape": model.input_shape if hasattr(model, 'input_shape') else None,
            "output_shape": model.output_shape if hasattr(model, 'output_shape') else None,
        }
        
        # Add layer types summary
        layer_types = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        summary["layer_types"] = layer_types
        
        return summary
    except Exception:
        # Return minimal summary if detailed extraction fails
        return {
            "num_layers": len(model.layers) if hasattr(model, 'layers') else 0,
            "summary_available": False
        } 