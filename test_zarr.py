import numpy as np
import os
import shutil
from paramlake.storage.zarr_manager import ZarrStorageManager
from paramlake.utils.config import ParamLakeConfig

# Clean up any previous test data
if os.path.exists('/tmp/test_zarr'):
    shutil.rmtree('/tmp/test_zarr')

# Create config and manager
config = ParamLakeConfig({
    'run_id': 'test', 
    'output_path': '/tmp/test_zarr', 
    'chunking': {'time_dimension': 5, 'target_chunk_size': 1000000}
})
manager = ZarrStorageManager(config)

# Create a test layer group
layer_group = manager.create_or_get_layer_group('test_layer', 'Dense')

# Create test tensor data
tensor_data = np.random.rand(512, 256).astype(np.float32)

# Store the tensor
print('Storing tensor...')
manager.store_tensor(layer_group, 'weight', 'weights', tensor_data)

# Verify it was stored correctly
print('Verifying storage...')
tensor_group = layer_group['weights']
tensor_array = tensor_group['weight']
print(f'Tensor shape: {tensor_array.shape}')
print(f'Tensor dtype: {tensor_array.dtype}')
print(f'Storage successful: {np.allclose(tensor_array[0], tensor_data)}')

# Close the manager
manager.close()
print('Test completed successfully!') 