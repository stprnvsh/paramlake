"""
Shared utility functions for TensorFlow models.
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