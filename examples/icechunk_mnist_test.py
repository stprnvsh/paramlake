"""
Simplified example using ParamLake with Icechunk for S3 storage.
"""

import os
import numpy as np
import tensorflow as tf
import icechunk
import zarr

# Set up AWS credentials
# IMPORTANT: Replace these with your own credentials or preferably set as environment variables
os.environ["AWS_ACCESS_KEY_ID"] = "YOUR_ACCESS_KEY"  # Replace with your own or use environment variables
os.environ["AWS_SECRET_ACCESS_KEY"] = "YOUR_SECRET_KEY"  # Replace with your own or use environment variables
os.environ["AWS_S3_ENDPOINT"] = "https://s3.amazonaws.com"

# Define storage parameters
BUCKET = "paramlake"
PREFIX = "mnist_test"
REGION = "us-east-1"

def create_mnist_repo():
    """Create a test repository for MNIST training."""
    print("Creating S3 storage for MNIST test...")
    
    # Configure repository storage
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        region=REGION,
        endpoint_url="https://s3.amazonaws.com",
        from_env=True
    )
    
    # Create repository configuration
    config = icechunk.RepositoryConfig.default()
    
    # Configure storage settings
    config.storage = icechunk.StorageSettings(
        concurrency=icechunk.StorageConcurrencySettings(
            max_concurrent_requests_for_object=10,
            ideal_concurrent_request_size=1000000,  # Use integer instead of float
        ),
        storage_class="STANDARD",
        metadata_storage_class="STANDARD",
        chunks_storage_class="STANDARD",
    )
    
    # Configure compression
    config.compression = icechunk.CompressionConfig(
        level=3,
        algorithm=icechunk.CompressionAlgorithm.Zstd,
    )
    
    # Configure caching
    config.caching = icechunk.CachingConfig(
        num_snapshot_nodes=100,
        num_chunk_refs=100,
        num_transaction_changes=100,
        num_bytes_attributes=10000,  # Use integer instead of float
        num_bytes_chunks=1000000,    # Use integer instead of float
    )
    
    # First check if repository exists
    if icechunk.Repository.exists(storage):
        print("Repository already exists, opening it...")
        repo = icechunk.Repository.open(storage, config=config)
    else:
        print("Creating new repository...")
        try:
            repo = icechunk.Repository.create(storage, config=config)
            # Save the configuration to persist it
            repo.save_config()
        except Exception as e:
            print(f"Error creating repository: {e}")
            try:
                print("Trying to open existing repository...")
                repo = icechunk.Repository.open(storage, config=config)
            except Exception as e2:
                print(f"Failed to open existing repository: {e2}")
                return None
    
    # Create a session for writing
    print("Creating writable session...")
    session = repo.writable_session("main")
    store = session.store
    
    # Create a Zarr group with metadata
    print("Creating Zarr structures...")
    root = zarr.group(store)
    
    # Create layers group
    if "layers" not in root:
        layers_group = root.create_group("layers")
    else:
        layers_group = root["layers"]
    
    # Create metrics group
    if "metrics" not in root:
        metrics_group = root.create_group("metrics")
    else:
        metrics_group = root["metrics"]
    
    # Add basic metadata
    from datetime import datetime
    root.attrs["paramlake_version"] = "0.1.0"
    root.attrs["framework"] = "tensorflow"
    root.attrs["framework_version"] = tf.__version__
    root.attrs["timestamp"] = datetime.now().isoformat()
    root.attrs["current_step"] = 0
    
    # Create a sample dataset with MNIST accuracy metrics
    accuracy_data = np.array([0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95], dtype=np.float32)
    if "accuracy" not in metrics_group:
        # Use create_dataset with shape parameter
        metrics_group.create_dataset("accuracy", data=accuracy_data, shape=accuracy_data.shape)
    
    # Commit the changes
    print("Committing changes...")
    snapshot_id = session.commit("Initialize repository for MNIST training")
    print(f"Committed with snapshot ID: {snapshot_id}")
    
    return repo

if __name__ == "__main__":
    # Create the repository
    repo = create_mnist_repo()
    
    if repo:
        print("Repository setup successful! Now you can run the full example.")
    else:
        print("Failed to set up repository.") 