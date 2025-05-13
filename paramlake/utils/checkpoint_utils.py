"""
Checkpoint utilities for saving and loading model state.
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
    
    return checkpoint_id


def load_checkpoint(
    model: tf.keras.Model,
    storage_manager: StorageInterface,
    checkpoint_id: str = None,
    step: int = None,
    by_name: bool = False,
    include_optimizer: bool = True,
    recompile: bool = True,
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
    else:
        # Assign weights directly
        # This assumes the model structure is exactly the same as when saved
        if len(weights_data) == 0:
            print("Warning: No weights found in checkpoint")
        elif len(weights_data) != len(model.weights):
            print(f"Warning: Checkpoint has {len(weights_data)} weights, but model has {len(model.weights)} weights")
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


def list_checkpoints(storage_manager: StorageInterface) -> List[Dict[str, Any]]:
    """
    List all available checkpoints.
    
    Args:
        storage_manager: Storage manager instance
        
    Returns:
        List of dictionaries with checkpoint metadata
    """
    return storage_manager.list_checkpoints() 