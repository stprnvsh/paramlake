"""Common test fixtures for ParamLake tests."""

import os
import shutil
import tempfile
from typing import Dict, Generator

import numpy as np
import pytest
import tensorflow as tf

from paramlake.utils.config import ParamLakeConfig


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test data."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # Clean up
    shutil.rmtree(test_dir)


@pytest.fixture
def config(temp_dir: str) -> ParamLakeConfig:
    """Create a basic configuration for testing."""
    config_dict = {
        "output_path": os.path.join(temp_dir, "test_paramlake.zarr"),
        "run_id": "test_run",
        "capture_weights": True,
        "capture_non_trainable": True,
        "capture_gradients": True,
        "capture_activations": True,
        "capture_frequency": 1,
        "include_layers": None,
        "exclude_layers": None,
        "include_types": None,
        "compression": {
            "algorithm": "blosc",
            "level": 5,
            "shuffle": True,
        },
        "chunking": {
            "time_dimension": 10,
            "spatial_dimensions": "auto",
            "target_chunk_size": 1000000,  # 1MB
        },
        "async_writes": False,
        "buffer_size": 100,
        "gradients": {
            "enabled": True,
            "auto_tracking": True,
        },
        "activations": {
            "sample_batch": None,
        },
    }
    return ParamLakeConfig(config_dict)


@pytest.fixture
def simple_model() -> tf.keras.Model:
    """Create a simple model for testing."""
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
    
    return model


@pytest.fixture
def sample_data() -> Dict[str, np.ndarray]:
    """Create sample data for testing."""
    # Create some random data
    np.random.seed(42)
    x_train = np.random.random((100, 5))
    y_train = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, size=(100,)), 
        num_classes=2
    )
    
    x_test = np.random.random((20, 5))
    y_test = tf.keras.utils.to_categorical(
        np.random.randint(0, 2, size=(20,)), 
        num_classes=2
    )
    
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    } 