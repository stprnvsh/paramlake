"""Tests for the collector components."""

import numpy as np
import pytest
import tensorflow as tf

from paramlake.collectors.activation_collector import ActivationCollector
from paramlake.collectors.gradient_collector import GradientCollector
from paramlake.collectors.weight_collector import WeightCollector
from paramlake.storage.zarr_manager import ZarrStorageManager


def test_weight_collector_initialization(config):
    """Test initialization of WeightCollector."""
    storage = ZarrStorageManager(config)
    
    # Initialize collector
    collector = WeightCollector(storage)
    
    # Check basic properties
    assert collector.storage == storage
    assert collector.capture_trainable is True
    assert collector.capture_non_trainable is True
    assert collector.include_layers is None
    assert collector.exclude_layers is None
    assert collector.include_types is None
    
    # With custom parameters
    collector = WeightCollector(
        storage,
        capture_trainable=False,
        capture_non_trainable=False,
        include_layers=["dense*"],
        exclude_layers=["dropout*"],
        include_types=["Dense"]
    )
    
    assert collector.capture_trainable is False
    assert collector.capture_non_trainable is False
    assert collector.include_layers == ["dense*"]
    assert collector.exclude_layers == ["dropout*"]
    assert collector.include_types == ["Dense"]
    
    # Clean up
    storage.close()


def test_should_capture_layer(config):
    """Test layer filtering logic."""
    storage = ZarrStorageManager(config)
    
    # Create test layers
    dense_layer = tf.keras.layers.Dense(10, name="dense_1")
    dropout_layer = tf.keras.layers.Dropout(0.5, name="dropout_1")
    
    # Test with no filters
    collector = WeightCollector(storage)
    assert collector.should_capture_layer(dense_layer) is True
    assert collector.should_capture_layer(dropout_layer) is True
    
    # Test with include_layers
    collector = WeightCollector(storage, include_layers=["dense*"])
    assert collector.should_capture_layer(dense_layer) is True
    assert collector.should_capture_layer(dropout_layer) is False
    
    # Test with exclude_layers
    collector = WeightCollector(storage, exclude_layers=["dropout*"])
    assert collector.should_capture_layer(dense_layer) is True
    assert collector.should_capture_layer(dropout_layer) is False
    
    # Test with include_types
    collector = WeightCollector(storage, include_types=["Dense"])
    assert collector.should_capture_layer(dense_layer) is True
    assert collector.should_capture_layer(dropout_layer) is False
    
    # Clean up
    storage.close()


def test_weight_collector_capture(config, simple_model):
    """Test capturing weights."""
    storage = ZarrStorageManager(config)
    
    collector = WeightCollector(storage)
    
    # Capture weights for a specific layer
    layer = simple_model.layers[0]
    collector.capture_layer_weights(layer)
    
    # Verify weights were captured
    layer_name = layer.name
    layer_type = layer.__class__.__name__
    layer_group = storage.layers_group[layer_name.replace("/", "_").replace(":", "_")]
    
    assert layer_group is not None
    assert "weights" in layer_group
    assert "kernel" in layer_group["weights"]
    
    # Test capture_model_weights
    collector.capture_model_weights(simple_model)
    
    # Verify all layers were captured
    for layer in simple_model.layers:
        layer_name = layer.name
        sanitized_name = layer_name.replace("/", "_").replace(":", "_")
        assert sanitized_name in storage.layers_group
    
    # Clean up
    storage.close()


def test_tensor_conversion(config):
    """Test tensor conversion utilities."""
    storage = ZarrStorageManager(config)
    collector = WeightCollector(storage)
    
    # Create a test tensor
    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    
    # Convert to numpy
    numpy_array = collector._tensor_to_numpy(tensor)
    
    # Check conversion
    assert isinstance(numpy_array, np.ndarray)
    np.testing.assert_array_equal(numpy_array, np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    # Test size estimation
    size = collector._estimate_tensor_size(numpy_array)
    assert size == numpy_array.size * numpy_array.dtype.itemsize
    
    # Clean up
    storage.close()


def test_activation_collector_initialization(config):
    """Test initialization of ActivationCollector."""
    storage = ZarrStorageManager(config)
    
    # Initialize collector
    collector = ActivationCollector(storage)
    
    # Check basic properties
    assert collector.storage == storage
    assert collector.include_layers is None
    assert collector.exclude_layers is None
    assert collector.include_types is None
    assert collector.activation_functions == {}
    assert collector.sample_input is None
    
    # With custom parameters
    collector = ActivationCollector(
        storage,
        include_layers=["dense*"],
        exclude_layers=["dropout*"],
        include_types=["Dense"]
    )
    
    assert collector.include_layers == ["dense*"]
    assert collector.exclude_layers == ["dropout*"]
    assert collector.include_types == ["Dense"]
    
    # Clean up
    storage.close()


def test_activation_collector_setup(config, simple_model):
    """Test setting up activation capture."""
    storage = ZarrStorageManager(config)
    collector = ActivationCollector(storage)
    
    # Set up activation capture
    collector.setup_activation_capture(simple_model)
    
    # Check that activation functions were registered
    assert "functional" in collector.activation_functions
    assert "layer_mapping" in collector.activation_functions
    
    # Clean up
    storage.close()


def test_activation_collection(config, simple_model):
    """Test capturing activations."""
    storage = ZarrStorageManager(config)
    collector = ActivationCollector(storage)
    
    # Set up activation capture
    collector.setup_activation_capture(simple_model)
    
    # Create sample input
    sample_input = np.random.random((1, 5)).astype(np.float32)
    
    # Capture activations
    collector.capture_activations(simple_model, sample_input)
    
    # Verify activations were captured for at least one layer
    assert len(storage.layers_group) > 0
    
    # Check that activations group exists
    for layer_name in storage.layers_group:
        layer_group = storage.layers_group[layer_name]
        if "activations" in layer_group:
            assert "activation" in layer_group["activations"]
            activation_array = layer_group["activations"]["activation"]
            assert activation_array.shape[0] >= 1  # At least one step
            break
    
    # Clean up
    storage.close()


def test_gradient_collector_initialization(config):
    """Test initialization of GradientCollector."""
    storage = ZarrStorageManager(config)
    
    # Initialize collector
    collector = GradientCollector(storage)
    
    # Check basic properties
    assert collector.storage == storage
    assert collector.include_layers is None
    assert collector.exclude_layers is None
    assert collector.include_types is None
    assert collector._var_to_layer_map == {}
    assert collector._original_train_step is None
    assert collector._original_apply_gradients is None
    
    # With custom parameters
    collector = GradientCollector(
        storage,
        include_layers=["dense*"],
        exclude_layers=["dropout*"],
        include_types=["Dense"]
    )
    
    assert collector.include_layers == ["dense*"]
    assert collector.exclude_layers == ["dropout*"]
    assert collector.include_types == ["Dense"]
    
    # Clean up
    storage.close()


def test_variable_mapping(config, simple_model):
    """Test building variable mapping."""
    storage = ZarrStorageManager(config)
    collector = GradientCollector(storage)
    
    # Build variable mapping
    collector.build_variable_mapping(simple_model)
    
    # Check that variables were mapped
    assert len(collector._var_to_layer_map) > 0
    
    # Verify mapping structure
    for var_name, (layer_name, tensor_name) in collector._var_to_layer_map.items():
        assert isinstance(layer_name, str)
        assert isinstance(tensor_name, str)
    
    # Clean up
    storage.close()


@pytest.mark.parametrize("batch_mode", [False, True])
def test_gradient_computation(config, simple_model, sample_data, batch_mode):
    """Test computing and capturing gradients."""
    storage = ZarrStorageManager(config)
    collector = GradientCollector(storage)
    
    # Get input and target data
    x_data = sample_data["x_train"][:1]  # Just use the first sample
    y_data = sample_data["y_train"][:1]
    
    # Compute and capture gradients
    if batch_mode and hasattr(collector, "compute_and_capture_gradients_batch"):
        collector.compute_and_capture_gradients_batch(
            simple_model, x_data, y_data, step=0
        )
    else:
        collector.compute_and_capture_gradients(
            simple_model, x_data, y_data, step=0
        )
    
    # Verify gradients were captured
    found_gradients = False
    for layer_name in storage.layers_group:
        layer_group = storage.layers_group[layer_name]
        if "gradients" in layer_group:
            found_gradients = True
            # Should have gradients for at least one tensor
            assert len(layer_group["gradients"]) > 0
            break
    
    assert found_gradients, "No gradients were captured"
    
    # Clean up
    storage.close()


@pytest.mark.parametrize("batch_mode", [False, True])
def test_weight_collection_batch(config, simple_model, batch_mode):
    """Test batch processing for weight collection."""
    storage = ZarrStorageManager(config)
    collector = WeightCollector(storage)
    
    # Capture weights for the whole model
    if batch_mode and hasattr(collector, "capture_model_weights_batch"):
        collector.capture_model_weights_batch(simple_model, step=0)
    else:
        collector.capture_model_weights(simple_model, step=0)
    
    # Verify weights were captured for all layers
    for layer in simple_model.layers:
        if layer.weights:  # Skip layers without weights
            layer_name = layer.name
            sanitized_name = layer_name.replace("/", "_").replace(":", "_")
            assert sanitized_name in storage.layers_group
            
            layer_group = storage.layers_group[sanitized_name]
            assert "weights" in layer_group
            
            # Check if weights are available
            for weight in layer.weights:
                name = weight.name.split("/")[-1]
                if ":" in name:
                    name = name.split(":")[0]
                
                # At least one of the weights should be captured
                if name in layer_group["weights"]:
                    assert layer_group["weights"][name].shape[0] >= 1
    
    # Clean up
    storage.close()


@pytest.mark.parametrize("batch_mode", [False, True])
def test_activation_collection_batch(config, simple_model, batch_mode):
    """Test batch processing for activation collection."""
    storage = ZarrStorageManager(config)
    collector = ActivationCollector(storage)
    
    # Set up activation capture
    collector.setup_activation_capture(simple_model)
    
    # Create sample input
    sample_input = np.random.random((1, 5)).astype(np.float32)
    
    # Capture activations
    if batch_mode and hasattr(collector, "capture_activations_batch"):
        collector.capture_activations_batch(simple_model, sample_input, step=0)
    else:
        collector.capture_activations(simple_model, sample_input, step=0)
    
    # Verify activations were captured for at least one layer
    found_activations = False
    for layer_name in storage.layers_group:
        layer_group = storage.layers_group[layer_name]
        if "activations" in layer_group:
            found_activations = True
            assert "activation" in layer_group["activations"]
            activation_array = layer_group["activations"]["activation"]
            assert activation_array.shape[0] >= 1  # At least one step
            break
    
    assert found_activations, "No activations were captured"
    
    # Clean up
    storage.close() 