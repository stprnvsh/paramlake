"""
Schema definitions for ParamLake Zarr.
"""

from typing import Dict, List, Optional

# Standard schema for layer types across frameworks
# Maps framework-specific layer types to standard ParamLake types
STANDARD_LAYER_TYPES = {
    # TensorFlow layer types
    "Dense": "Dense",
    "Conv1D": "Conv1D",
    "Conv2D": "Conv2D", 
    "Conv3D": "Conv3D",
    "BatchNormalization": "BatchNorm",
    "LayerNormalization": "LayerNorm",
    "Dropout": "Dropout",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "Embedding": "Embedding",
    "Flatten": "Flatten",
    "MaxPooling1D": "MaxPool1D",
    "MaxPooling2D": "MaxPool2D",
    "MaxPooling3D": "MaxPool3D",
    "AveragePooling1D": "AvgPool1D",
    "AveragePooling2D": "AvgPool2D",
    "AveragePooling3D": "AvgPool3D",
    "GlobalMaxPooling1D": "GlobalMaxPool1D",
    "GlobalMaxPooling2D": "GlobalMaxPool2D",
    "GlobalMaxPooling3D": "GlobalMaxPool3D",
    "GlobalAveragePooling1D": "GlobalAvgPool1D",
    "GlobalAveragePooling2D": "GlobalAvgPool2D",
    "GlobalAveragePooling3D": "GlobalAvgPool3D",
    
    # PyTorch layer types (for future compatibility)
    "Linear": "Dense",
    "Conv1d": "Conv1D",
    "Conv2d": "Conv2D",
    "Conv3d": "Conv3D",
    "BatchNorm1d": "BatchNorm",
    "BatchNorm2d": "BatchNorm",
    "BatchNorm3d": "BatchNorm",
    "LayerNorm": "LayerNorm",
    "LSTM": "LSTM",
    "GRU": "GRU",
    "Dropout": "Dropout",
    "MaxPool1d": "MaxPool1D",
    "MaxPool2d": "MaxPool2D",
    "MaxPool3d": "MaxPool3D",
    "AvgPool1d": "AvgPool1D",
    "AvgPool2d": "AvgPool2D",
    "AvgPool3d": "AvgPool3D",
    "AdaptiveMaxPool1d": "AdaptiveMaxPool1D",
    "AdaptiveMaxPool2d": "AdaptiveMaxPool2D",
    "AdaptiveMaxPool3d": "AdaptiveMaxPool3D",
    "AdaptiveAvgPool1d": "AdaptiveAvgPool1D",
    "AdaptiveAvgPool2d": "AdaptiveAvgPool2D",
    "AdaptiveAvgPool3d": "AdaptiveAvgPool3D",
    
    # JAX/Flax layer types (for future compatibility)
    "Dense": "Dense",
    "Conv": "Conv2D",
    "BatchNorm": "BatchNorm",
    "LayerNorm": "LayerNorm",
    "Dropout": "Dropout",
    "Embed": "Embedding",
    "recurrent.LSTMCell": "LSTM",
    "recurrent.GRUCell": "GRU",
}

# Standard schema for tensor types across frameworks
# Maps framework-specific tensor names to standard ParamLake names
STANDARD_TENSOR_NAMES = {
    # TensorFlow tensor names
    "kernel": "weights",
    "bias": "bias",
    "gamma": "scale",
    "beta": "shift",
    "moving_mean": "running_mean",
    "moving_variance": "running_var",
    
    # PyTorch tensor names (for future compatibility)
    "weight": "weights",
    "bias": "bias",
    "running_mean": "running_mean",
    "running_var": "running_var",
    
    # JAX/Flax tensor names (for future compatibility)
    "kernel": "weights",
    "bias": "bias",
    "scale": "scale",
    "mean": "running_mean",
    "var": "running_var",
}

# Standard schema for tensor types
TENSOR_CATEGORIES = {
    "weights": "trainable parameter",
    "bias": "trainable parameter",
    "scale": "trainable parameter",
    "shift": "trainable parameter",
    "running_mean": "non-trainable statistic",
    "running_var": "non-trainable statistic",
    "activation": "intermediate value",
    "gradient": "gradient",
}

# Zarr schema constants
ZARR_SCHEMA = {
    "version": "1.0.0",
    "structure": {
        "run_group": {
            "description": "Top-level group for a single training run",
            "attributes": {
                "framework": "Name of the framework (tensorflow, pytorch, etc.)",
                "framework_version": "Version of the framework",
                "paramlake_version": "Version of ParamLake used to create the dataset",
                "timestamp": "ISO-format timestamp of when the run was created",
                "config": "JSON string of the configuration used for this run",
                "current_step": "Current step/epoch of the training run",
                "final_step": "Final step/epoch of the training run (set when closed)",
            },
            "subgroups": {
                "layers": {
                    "description": "Group containing all layer data",
                    "structure": {
                        "{layer_name}": {
                            "description": "Group for a specific layer",
                            "attributes": {
                                "name": "Original layer name",
                                "type": "Layer type (e.g., Dense, Conv2D)",
                                "trainable": "Whether the layer is trainable",
                                "input_shape": "Shape of the layer's input",
                                "output_shape": "Shape of the layer's output",
                                "config": "Layer configuration (if available)",
                            },
                            "subgroups": {
                                "weights": {
                                    "description": "Group containing trainable weights",
                                    "arrays": {
                                        "{tensor_name}": {
                                            "shape": "(steps, *tensor_shape)",
                                            "attributes": {
                                                "shape": "Shape of the tensor (without time dimension)",
                                                "dtype": "Data type of the tensor",
                                            },
                                        },
                                    },
                                },
                                "non_trainable": {
                                    "description": "Group containing non-trainable variables",
                                    "arrays": {
                                        "{tensor_name}": {
                                            "shape": "(steps, *tensor_shape)",
                                            "attributes": {
                                                "shape": "Shape of the tensor (without time dimension)",
                                                "dtype": "Data type of the tensor",
                                            },
                                        },
                                    },
                                },
                                "gradients": {
                                    "description": "Group containing gradients for trainable weights",
                                    "arrays": {
                                        "{tensor_name}": {
                                            "shape": "(steps, *tensor_shape)",
                                            "attributes": {
                                                "shape": "Shape of the tensor (without time dimension)",
                                                "dtype": "Data type of the tensor",
                                            },
                                        },
                                    },
                                },
                                "activations": {
                                    "description": "Group containing layer activations",
                                    "arrays": {
                                        "activation": {
                                            "shape": "(steps, batch_size, *activation_shape)",
                                            "attributes": {
                                                "shape": "Shape of the activation (without time dimension)",
                                                "dtype": "Data type of the tensor",
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "metrics": {
                    "description": "Group containing training metrics",
                    "arrays": {
                        "{metric_name}": {
                            "shape": "(steps,)",
                            "attributes": {},
                        },
                    },
                },
            },
        },
    },
}

def get_standard_layer_type(framework_layer_type: str) -> str:
    """
    Convert a framework-specific layer type to the standard ParamLake type.
    
    Args:
        framework_layer_type: Framework-specific layer type
        
    Returns:
        Standard ParamLake layer type
    """
    return STANDARD_LAYER_TYPES.get(framework_layer_type, framework_layer_type)

def get_standard_tensor_name(framework_tensor_name: str) -> str:
    """
    Convert a framework-specific tensor name to the standard ParamLake name.
    
    Args:
        framework_tensor_name: Framework-specific tensor name
        
    Returns:
        Standard ParamLake tensor name
    """
    return STANDARD_TENSOR_NAMES.get(framework_tensor_name, framework_tensor_name)

def get_tensor_category(tensor_name: str) -> str:
    """
    Get the category of a tensor from its standard name.
    
    Args:
        tensor_name: Standard ParamLake tensor name
        
    Returns:
        Category of the tensor
    """
    return TENSOR_CATEGORIES.get(tensor_name, "unknown") 