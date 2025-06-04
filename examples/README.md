# ParamLake Examples

This directory contains example scripts demonstrating the various capabilities of the ParamLake framework for tracking and analyzing deep learning model parameters, gradients, activations, and optimizer states.

## Available Examples

### 1. Optimizer State Capture
**File:** `test_optimizer_capture.py`

Demonstrates how to capture and analyze optimizer states during training. This is useful for:
- Monitoring optimizer moments (for Adam, RMSprop, etc.)
- Debugging optimization behavior
- Restoring training from exact optimizer state

### 2. Weights Capture
**File:** `test_weights_capture.py`

Shows how to track model weights throughout training. Use this to:
- Visualize weight evolution over time
- Calculate weight changes between iterations
- Compare initialization vs. final weights

### 3. Gradients Capture
**File:** `test_gradients_capture.py`

Illustrates gradient capturing during backpropagation. This helps with:
- Identifying vanishing/exploding gradients
- Analyzing gradient flow through the network
- Understanding how gradients change during training

### 4. Activations Capture
**File:** `test_activations_capture.py`

Shows how to record layer activations during training. Applications include:
- Analyzing neuron activation patterns
- Detecting dead/saturated neurons
- Understanding representation learning

## Running the Examples

Each example can be run directly:

```bash
python examples/test_optimizer_capture.py
python examples/test_weights_capture.py
python examples/test_gradients_capture.py
python examples/test_activations_capture.py
```

The examples will create their output files in the `test_output` directory, with each test saving data in a separate Zarr archive.

## Example Structure

All examples follow a similar pattern:
1. Create a simple regression dataset
2. Define a small neural network
3. Use the `@paramlake` decorator to capture the specified data during training
4. Analyze the captured data using `ZarrModelAnalyzer`

These examples serve as both demonstrations and tests of the ParamLake functionality. 