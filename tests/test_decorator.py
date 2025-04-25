"""Tests for the decorator components."""

import os
from typing import Dict

import numpy as np
import pytest
import tensorflow as tf

from paramlake.decorators.model_decorator import ParamLakeCallback, ModelWrapper, paramlake, load_config
from paramlake.storage.zarr_analyzer import ZarrModelAnalyzer


def test_load_config(temp_dir):
    """Test loading configuration from different sources."""
    # Test with dictionary
    config_dict = {
        "output_path": os.path.join(temp_dir, "test_paramlake.zarr"),
        "run_id": "test_run",
        "capture_weights": True,
    }
    
    config = load_config(config_dict)
    assert config["output_path"] == os.path.join(temp_dir, "test_paramlake.zarr")
    assert config["run_id"] == "test_run"
    assert config["capture_weights"] is True
    
    # Test with keyword arguments
    config = load_config(output_path=os.path.join(temp_dir, "test_paramlake2.zarr"), run_id="test_run2")
    assert config["output_path"] == os.path.join(temp_dir, "test_paramlake2.zarr")
    assert config["run_id"] == "test_run2"
    
    # Test overriding with kwargs
    config = load_config(config_dict, output_path=os.path.join(temp_dir, "override.zarr"))
    assert config["output_path"] == os.path.join(temp_dir, "override.zarr")
    assert config["run_id"] == "test_run"


def test_paramlake_callback_initialization(config, simple_model):
    """Test initialization of ParamLakeCallback."""
    callback = ParamLakeCallback(config)
    
    # Check basic properties
    assert callback.config == config
    assert callback.capture_frequency == 1
    assert callback.capture_gradients is True
    
    # Set the model and check
    callback.model = simple_model
    assert callback.model == simple_model


def test_paramlike_callback_training(config, simple_model, sample_data):
    """Test callback during model training."""
    # Create callback
    callback = ParamLakeCallback(config)
    callback.model = simple_model
    
    # Get training data
    x_train = sample_data["x_train"]
    y_train = sample_data["y_train"]
    
    # Test callbacks
    callback.on_train_begin()
    
    # Train for a few epochs with the callback
    simple_model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=32,
        callbacks=[callback],
        verbose=0
    )
    
    # Verify data was captured
    analyzer = ZarrModelAnalyzer(config["output_path"], config["run_id"])
    
    # Check that layers were captured
    layer_names = analyzer.get_layer_names()
    assert len(layer_names) > 0
    
    # Check that metrics were captured
    metrics = analyzer.get_metrics()
    assert len(metrics) > 0
    
    # Clean up
    callback.on_train_end()
    analyzer.close()
    callback.close()


def test_model_wrapper(config, simple_model, sample_data):
    """Test ModelWrapper functionality."""
    # Create wrapper
    wrapper = ModelWrapper(simple_model, config)
    
    # Check properties
    assert wrapper.model == simple_model
    assert wrapper.config == config
    
    # Test forwarding through the wrapper
    x_input = np.random.random((1, 5)).astype(np.float32)
    
    # Original model prediction
    original_pred = simple_model(x_input).numpy()
    
    # Wrapper prediction
    wrapped_pred = wrapper(x_input).numpy()
    
    # Should be identical
    np.testing.assert_array_equal(original_pred, wrapped_pred)
    
    # Clean up
    wrapper.close()


@pytest.mark.parametrize("with_params", [True, False])
def test_paramlake_decorator(temp_dir, sample_data, with_params):
    """Test the paramlake decorator."""
    # Define parameters
    output_path = os.path.join(temp_dir, "test_decorator.zarr")
    run_id = "test_decorator_run"
    
    if with_params:
        # Create decorated function with parameters
        @paramlake(output_path=output_path, run_id=run_id, capture_frequency=1)
        def train_model():
            # Create a model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,), name='dense_1'),
                tf.keras.layers.Dense(5, activation='relu', name='dense_2'),
                tf.keras.layers.Dense(2, activation='softmax', name='output')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for a few steps
            x_train = sample_data["x_train"]
            y_train = sample_data["y_train"]
            
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
            
            return model
    else:
        # Create decorated function without parameters
        # Set parameters on the decorator directly
        paramlake.params = {
            'output_path': output_path,
            'run_id': run_id,
            'capture_frequency': 1
        }
        
        @paramlake
        def train_model():
            # Create a model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(5,), name='dense_1'),
                tf.keras.layers.Dense(5, activation='relu', name='dense_2'),
                tf.keras.layers.Dense(2, activation='softmax', name='output')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train for a few steps
            x_train = sample_data["x_train"]
            y_train = sample_data["y_train"]
            
            model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
            
            return model
    
    # Run the decorated function
    model = train_model()
    
    # Verify that the model was trained and data was captured
    assert model is not None
    assert os.path.exists(output_path)
    
    # Check the captured data using analyzer
    analyzer = ZarrModelAnalyzer(output_path, run_id)
    
    # Should have captured at least one layer
    layer_names = analyzer.get_layer_names()
    assert len(layer_names) > 0
    
    # Should have captured metrics
    metrics = analyzer.get_metrics()
    assert len(metrics) > 0
    
    # Clean up
    analyzer.close()


def test_decorator_with_callback_param(temp_dir, sample_data):
    """Test the paramlake decorator with callback parameter."""
    # Define parameters
    output_path = os.path.join(temp_dir, "test_decorator_callback.zarr")
    run_id = "test_decorator_callback_run"
    
    # Create decorated function that accepts a callback
    @paramlake(output_path=output_path, run_id=run_id)
    def train_model_with_callback(paramlake_callback=None):
        # Create a model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,), name='dense_1'),
            tf.keras.layers.Dense(5, activation='relu', name='dense_2'),
            tf.keras.layers.Dense(2, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set the model on the callback
        if paramlake_callback:
            paramlake_callback.model = model
        
        # Train for a few steps
        x_train = sample_data["x_train"]
        y_train = sample_data["y_train"]
        
        # Use our own callback list to check that the ParamLake callback is included
        callbacks = [tf.keras.callbacks.History()]
        if paramlake_callback:
            callbacks.append(paramlake_callback)
        
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0, callbacks=callbacks)
        
        return model
    
    # Run the decorated function
    model = train_model_with_callback()
    
    # Verify that the model was trained and data was captured
    assert model is not None
    assert os.path.exists(output_path)
    
    # Check the captured data using analyzer
    analyzer = ZarrModelAnalyzer(output_path, run_id)
    
    # Should have captured at least one layer
    layer_names = analyzer.get_layer_names()
    assert len(layer_names) > 0
    
    # Clean up
    analyzer.close() 