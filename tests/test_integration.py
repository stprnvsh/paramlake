"""Integration tests for ParamLake."""

import os
import tensorflow as tf
import numpy as np
import pytest

from paramlake.decorators.model_decorator import paramlake
from paramlake.storage.zarr_analyzer import ZarrModelAnalyzer


def test_complete_paramlake_workflow(temp_dir, sample_data):
    """Test the complete ParamLake workflow from model creation to analysis."""
    # Define parameters
    output_path = os.path.join(temp_dir, "test_workflow.zarr")
    run_id = "test_workflow_run"
    
    # Define decorated training function
    @paramlake(
        output_path=output_path,
        run_id=run_id,
        capture_weights=True,
        capture_gradients=True,
        capture_activations=True,
        capture_frequency=1
    )
    def train_model(paramlake_callback=None):
        """Train a simple model with ParamLake tracking."""
        # Create a model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(5,), name='dense_1'),
            tf.keras.layers.BatchNormalization(name='bn_1'),
            tf.keras.layers.Dense(8, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(2, activation='softmax', name='output')
        ])
        
        # Use a custom optimizer to ensure we can track gradients
        optimizer = tf.keras.optimizers.Adam(0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Ensure the callback knows about our model
        if paramlake_callback:
            paramlake_callback.model = model
            
            # Set sample data for activation capture
            x_sample = sample_data["x_train"][:1]
            paramlake_callback.set_sample_data(x_sample)
        
        # Get training data
        x_train = sample_data["x_train"]
        y_train = sample_data["y_train"]
        x_test = sample_data["x_test"]
        y_test = sample_data["y_test"]
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=32,
            validation_data=(x_test, y_test),
            verbose=0
        )
        
        # Evaluate the model
        results = model.evaluate(x_test, y_test, verbose=0)
        
        return model, history, results
    
    # Train the model with ParamLake tracking
    model, history, results = train_model()
    
    # Verify model was trained
    assert model is not None
    assert history is not None
    assert len(history.history["loss"]) == 3
    assert results[0] > 0  # Loss
    
    # Verify data was captured by using the analyzer
    analyzer = ZarrModelAnalyzer(output_path, run_id)
    
    # 1. Check layers were captured
    layer_names = analyzer.get_layer_names()
    assert len(layer_names) >= 5  # Should have at least 5 layers
    
    # Verify layer types
    layer_types = {name: analyzer.get_layer_info(name)["type"] for name in layer_names}
    assert any("Dense" in layer_type for layer_type in layer_types.values())
    assert any("BatchNormalization" in layer_type for layer_type in layer_types.values())
    
    # 2. Check metrics were captured
    metrics = analyzer.get_metrics()
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert len(metrics["loss"]) >= 3  # Should have at least 3 epoch values
    
    # 3. Check weight tensors were captured
    for layer_name in layer_names:
        layer_info = analyzer.get_layer_info(layer_name)
        if "weights" in layer_info["tensor_types"]:
            # Get available weight tensors
            weight_tensors = layer_info["available_tensors"]["weights"]
            if "kernel" in weight_tensors:
                # Get kernel weights for all steps
                kernel_weights = analyzer.get_tensor_data(
                    layer_name=layer_name,
                    tensor_type="weights",
                    tensor_name="kernel"
                )
                assert kernel_weights.shape[0] >= 1  # Should have at least 1 step
                break
    
    # 4. Check activations were captured (might not be present for all layers)
    activations_captured = False
    for layer_name in layer_names:
        layer_info = analyzer.get_layer_info(layer_name)
        if "activations" in layer_info["tensor_types"]:
            activations_captured = True
            break
    assert activations_captured
    
    # 5. Check gradients were captured (might not be present for all layers)
    gradients_captured = False
    for layer_name in layer_names:
        layer_info = analyzer.get_layer_info(layer_name)
        if "gradients" in layer_info["tensor_types"]:
            gradients_captured = True
            break
    assert gradients_captured
            
    # Get a model summary
    model_summary = analyzer.get_model_summary()
    assert model_summary["total_layers"] >= 5
    
    # Clean up
    analyzer.close()

