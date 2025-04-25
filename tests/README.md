# ParamLake Test Suite

This directory contains the test suite for the ParamLake library. The tests are organized into the following modules:

## Test Structure

- **test_storage.py**: Tests for the storage components (ZarrStorageManager, StorageInterface)
- **test_collectors.py**: Tests for the collector components (WeightCollector, ActivationCollector, GradientCollector)
- **test_decorator.py**: Tests for the decorator components (paramlake decorator, ParamLakeCallback, ModelWrapper)
- **test_integration.py**: End-to-end integration tests for the entire ParamLake workflow

## Common Fixtures

The common test fixtures are defined in `conftest.py` and include:

- `temp_dir`: Creates a temporary directory for test data that is cleaned up after the test
- `config`: Creates a basic ParamLake configuration for testing
- `simple_model`: Creates a simple TensorFlow model for testing
- `sample_data`: Creates sample training data for testing

## Running Tests

To run the tests, you can use pytest:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=paramlake

# Run a specific test file
pytest tests/test_storage.py

# Run a specific test
pytest tests/test_storage.py::test_store_tensor
```

## Test Coverage

The test suite is configured to generate coverage reports. You can view the coverage report in HTML format:

```bash
pytest --cov=paramlake --cov-report=html
# Then open htmlcov/index.html in your browser
``` 