"""Tests for the storage components."""

import os
from typing import Dict

import numpy as np
import pytest
import zarr

from paramlake.storage.factory import create_storage_manager
from paramlake.storage.storage_interface import StorageInterface
from paramlake.storage.zarr_manager import ZarrStorageManager


def test_create_storage_manager(config):
    """Test creation of storage manager."""
    storage = create_storage_manager(config)
    assert isinstance(storage, StorageInterface)
    assert isinstance(storage, ZarrStorageManager)
    storage.close()


def test_zarr_storage_manager_initialization(config):
    """Test initialization of ZarrStorageManager."""
    storage = ZarrStorageManager(config)
    
    # Check that the zarr store was created
    assert os.path.exists(config["output_path"])
    
    # Check that the run group exists
    assert config["run_id"] in storage.store
    
    # Check that the standard groups were created
    assert "layers" in storage.run_group
    assert "metrics" in storage.run_group
    
    # Clean up
    storage.close()


def test_create_layer_group(config):
    """Test creating a layer group."""
    storage = ZarrStorageManager(config)
    
    # Create a layer group
    layer_name = "test_layer"
    layer_type = "Dense"
    layer_group = storage.create_or_get_layer_group(layer_name, layer_type)
    
    # Check that the group was created with correct attributes
    assert layer_group is not None
    assert layer_group.attrs["name"] == layer_name
    assert layer_group.attrs["type"] == layer_type
    
    # Clean up
    storage.close()


def test_store_tensor(config):
    """Test storing a tensor."""
    storage = ZarrStorageManager(config)
    
    # Create a layer group
    layer_name = "test_layer"
    layer_type = "Dense"
    layer_group = storage.create_or_get_layer_group(layer_name, layer_type)
    
    # Create a test tensor
    tensor_name = "kernel"
    tensor_type = "weights"
    tensor_data = np.random.random((10, 5)).astype(np.float32)
    
    # Store the tensor
    storage.store_tensor(layer_group, tensor_name, tensor_type, tensor_data)
    
    # Verify the tensor was stored correctly
    assert tensor_type in layer_group
    assert tensor_name in layer_group[tensor_type]
    
    # Check that the data matches
    stored_tensor = layer_group[tensor_type][tensor_name]
    np.testing.assert_array_equal(stored_tensor[0], tensor_data)
    
    # Clean up
    storage.close()


def test_store_metric(config):
    """Test storing a metric."""
    storage = ZarrStorageManager(config)
    
    # Store a test metric
    metric_name = "accuracy"
    metric_value = 0.95
    step = 0
    
    storage.store_metric(metric_name, metric_value, step)
    
    # Verify the metric was stored correctly
    assert metric_name in storage.metrics_group
    assert storage.metrics_group[metric_name][step] == metric_value
    
    # Store another value at a different step
    step = 1
    metric_value = 0.97
    storage.store_metric(metric_name, metric_value, step)
    
    # Verify both values are stored
    assert storage.metrics_group[metric_name][0] == 0.95
    assert storage.metrics_group[metric_name][1] == 0.97
    
    # Clean up
    storage.close()


def test_incremental_storage(config):
    """Test incremental storage of tensors."""
    storage = ZarrStorageManager(config)
    
    # Create a layer group
    layer_name = "test_layer"
    layer_type = "Dense"
    layer_group = storage.create_or_get_layer_group(layer_name, layer_type)
    
    # Create test tensors
    tensor_name = "kernel"
    tensor_type = "weights"
    
    # Store at step 0
    tensor_data_0 = np.random.random((10, 5)).astype(np.float32)
    storage.store_tensor(layer_group, tensor_name, tensor_type, tensor_data_0, step=0)
    
    # Store at step 1
    tensor_data_1 = np.random.random((10, 5)).astype(np.float32)
    storage.store_tensor(layer_group, tensor_name, tensor_type, tensor_data_1, step=1)
    
    # Verify both tensors were stored correctly
    stored_tensor = layer_group[tensor_type][tensor_name]
    assert stored_tensor.shape[0] >= 2  # Should have at least 2 steps
    np.testing.assert_array_equal(stored_tensor[0], tensor_data_0)
    np.testing.assert_array_equal(stored_tensor[1], tensor_data_1)
    
    # Clean up
    storage.close()


def test_step_management(config):
    """Test step management functionality."""
    storage = ZarrStorageManager(config)
    
    # Check initial step
    assert storage.current_step == 0
    
    # Increment step
    storage.increment_step()
    assert storage.current_step == 1
    
    # Set step
    storage.set_step(5)
    assert storage.current_step == 5
    
    # Verify step is stored in metadata
    assert storage.run_group.attrs["current_step"] == 5
    
    # Clean up
    storage.close()
    
    # Reopen and check that step was restored
    storage = ZarrStorageManager(config)
    assert storage.current_step == 5
    storage.close()


def test_metadata_storage(config):
    """Test storing layer metadata."""
    storage = ZarrStorageManager(config)
    
    # Create a layer group
    layer_name = "test_layer"
    layer_type = "Dense"
    layer_group = storage.create_or_get_layer_group(layer_name, layer_type)
    
    # Create test metadata
    metadata = {
        "name": layer_name,
        "type": layer_type,
        "trainable": True,
        "input_shape": [None, 10],
        "output_shape": [None, 5],
        "config": {
            "units": 5,
            "activation": "relu",
            "use_bias": True
        }
    }
    
    # Store metadata
    storage.store_layer_metadata(layer_group, metadata)
    
    # Verify metadata was stored correctly
    for key, value in metadata.items():
        assert key in layer_group.attrs
        if isinstance(value, dict):
            # For nested dictionaries, we serialize as strings
            assert layer_group.attrs[key] is not None
        else:
            assert layer_group.attrs[key] == value
    
    # Clean up
    storage.close()


def test_storage_with_real_model(config, simple_model):
    """Test storage with a real TensorFlow model."""
    storage = ZarrStorageManager(config)
    
    # Get a layer from the model
    layer = simple_model.layers[0]
    layer_name = layer.name
    layer_type = layer.__class__.__name__
    
    # Create a layer group
    layer_group = storage.create_or_get_layer_group(layer_name, layer_type)
    
    # Store a weight tensor
    weight = layer.weights[0]  # Kernel
    tensor_name = "kernel"
    tensor_type = "weights"
    tensor_data = weight.numpy()
    
    storage.store_tensor(layer_group, tensor_name, tensor_type, tensor_data)
    
    # Verify the tensor was stored correctly
    assert tensor_type in layer_group
    assert tensor_name in layer_group[tensor_type]
    
    # Check that the data matches
    stored_tensor = layer_group[tensor_type][tensor_name]
    np.testing.assert_array_equal(stored_tensor[0], tensor_data)
    
    # Store metadata
    metadata = {
        "name": layer_name,
        "type": layer_type,
        "trainable": layer.trainable,
    }
    
    storage.store_layer_metadata(layer_group, metadata)
    
    # Verify metadata was stored
    assert layer_group.attrs["name"] == layer_name
    assert layer_group.attrs["type"] == layer_type
    assert layer_group.attrs["trainable"] == layer.trainable
    
    # Clean up
    storage.close() 