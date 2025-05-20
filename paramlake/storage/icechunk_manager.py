"""
Icechunk storage manager for ParamLake.
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import zarr

try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

from paramlake.utils.config import ParamLakeConfig
from paramlake.storage.storage_interface import StorageInterface


class IcechunkStorageManager(StorageInterface):
    """Manages storage of model parameters using Icechunk as the backend."""

    def __init__(self, config: ParamLakeConfig):
        """
        Initialize the Icechunk storage manager.
        
        Args:
            config: ParamLake configuration
        """
        if not HAS_ICECHUNK:
            raise ImportError("Icechunk is required but not installed. Install it with 'pip install icechunk'.")
            
        self.config = config
        # Get the actual cloud storage type (s3, gcs, azure, local)
        storage_backend = config.get("storage_backend", config.get("storage_type", "s3"))
        
        # If storage_type is 'icechunk', check for storage_backend or default to 's3'
        if storage_backend == "icechunk":
            storage_backend = "s3"  # Default to s3 if not specified
        
        # Set up Icechunk repository
        if storage_backend == "s3":
            # Set environment variables if specified and not already set
            endpoint_url = config.get("endpoint_url", "https://s3.amazonaws.com")
            if not os.environ.get('AWS_S3_ENDPOINT'):
                os.environ['AWS_S3_ENDPOINT'] = endpoint_url
                
            storage = icechunk.s3_storage(
                bucket=config.get("bucket"),
                prefix=config.get("prefix", "paramlake_data"),
                region=config.get("region", "us-east-1"),
                endpoint_url=endpoint_url,
                from_env=True
            )
        elif storage_backend == "gcs":
            storage = icechunk.gcs_storage(
                bucket=config.get("bucket"),
                prefix=config.get("prefix", "paramlake_data"),
                from_env=True
            )
        elif storage_backend == "azure":
            storage = icechunk.azure_storage(
                account=config.get("account"),
                container=config.get("container"),
                prefix=config.get("prefix", "paramlake_data"),
                from_env=True
            )
        elif storage_backend == "local":
            storage = icechunk.local_filesystem_storage(config.get("output_path"))
        else:
            raise ValueError(f"Unsupported storage backend for Icechunk: {storage_backend}")
            
        # Create repository configuration
        repo_config = icechunk.RepositoryConfig.default()
        
        # Configure storage settings
        repo_config.storage = icechunk.StorageSettings(
            concurrency=icechunk.StorageConcurrencySettings(
                max_concurrent_requests_for_object=10,
                ideal_concurrent_request_size=1000000,
            ),
            storage_class="STANDARD",
            metadata_storage_class="STANDARD",
            chunks_storage_class="STANDARD",
        )
        
        # Configure compression
        compression_level = config.get("compression", {}).get("level", 3)
        compression_algorithm = config.get("compression", {}).get("algorithm", "zstd")
        
        # Map our algorithm names to Icechunk's enumeration
        if compression_algorithm.lower() in ["zstd", "blosc_zstd"]:
            icechunk_algorithm = icechunk.CompressionAlgorithm.Zstd
        elif compression_algorithm.lower() in ["lz4", "blosc_lz4"]:
            icechunk_algorithm = icechunk.CompressionAlgorithm.Lz4
        else:
            # Default to zstd
            icechunk_algorithm = icechunk.CompressionAlgorithm.Zstd
            
        repo_config.compression = icechunk.CompressionConfig(
            level=compression_level,
            algorithm=icechunk_algorithm,
        )
        
        # Configure caching
        repo_config.caching = icechunk.CachingConfig(
            num_snapshot_nodes=100,
            num_chunk_refs=100,
            num_transaction_changes=100,
            num_bytes_attributes=10000,
            num_bytes_chunks=1000000,
        )
        
        # Create or open repository
        if config.get("create_repo", False):
            try:
                self.repo = icechunk.Repository.create(storage, config=repo_config)
                # Save the configuration to persist it
                self.repo.save_config()
            except Exception as e:
                print(f"Error creating repository: {e}")
                print("Trying to open existing repository...")
                self.repo = icechunk.Repository.open(storage, config=repo_config)
        else:
            self.repo = icechunk.Repository.open(storage, config=repo_config)
        
        # Set up branch/tag handling
        self.run_id = config.get("run_id")
        
        # Create session for writing
        try:
            self.session = self.repo.writable_session("main")
            self.store = self.session.store
        except Exception as e:
            print(f"Error creating writable session: {e}. Creating 'main' branch first.")
            # Create the main branch if it doesn't exist
            init_snapshot = self.repo.initial_snapshot()
            self.repo.create_branch("main", snapshot_id=init_snapshot.id)
            self.session = self.repo.writable_session("main")
            self.store = self.session.store
        
        # Initialize zarr groups
        self._initialize_zarr_groups()
        
        # Track the current step
        self.current_step = 0
        
        # Commit frequency
        self.commit_frequency = config.get("icechunk", {}).get("commit_frequency", 10)
        self.last_commit_step = -1
        
        # Track created tags to avoid duplicates
        self.created_tags = set()
        
        # Tracked layers for debugging
        self.tracked_layers = set()

        # Create checkpoints group if it doesn't exist
        if "checkpoints" not in self.root_group:
            self.checkpoints_group = self.root_group.create_group("checkpoints")
        else:
            self.checkpoints_group = self.root_group["checkpoints"]

        # Track checkpoint snapshots
        self.checkpoint_snapshots = {}

    def _initialize_zarr_groups(self):
        """Initialize the basic zarr group structure needed for ParamLake."""
        # Root group is already created by icechunk
        self.root_group = zarr.open_group(self.store, mode="a")
            
        # Create layers group if it doesn't exist
        if "layers" not in self.root_group:
            self.layers_group = self.root_group.create_group("layers")
        else:
            self.layers_group = self.root_group["layers"]
            
        # Create metrics group if it doesn't exist
        if "metrics" not in self.root_group:
            self.metrics_group = self.root_group.create_group("metrics")
        else:
            self.metrics_group = self.root_group["metrics"]
            
        # Create optimizer_states group if it doesn't exist
        if "optimizer_states" not in self.root_group:
            self.optimizer_states_group = self.root_group.create_group("optimizer_states")
        else:
            self.optimizer_states_group = self.root_group["optimizer_states"]
            
        # Create optimizer_info group if it doesn't exist
        if "optimizer_info" not in self.root_group:
            self.optimizer_info_group = self.root_group.create_group("optimizer_info")
        else:
            self.optimizer_info_group = self.root_group["optimizer_info"]
        
        # Initialize run metadata if this is a new run
        self._initialize_run_metadata()
        
    def _initialize_run_metadata(self) -> None:
        """Initialize metadata for the current run."""
        import tensorflow as tf
        from datetime import datetime
        
        # Check if metadata already exists
        if "paramlake_version" in self.root_group.attrs:
            return
            
        # Store basic metadata
        self.root_group.attrs["paramlake_version"] = "0.1.0"
        self.root_group.attrs["framework"] = "tensorflow"
        self.root_group.attrs["framework_version"] = tf.__version__
        self.root_group.attrs["timestamp"] = datetime.now().isoformat()
        self.root_group.attrs["current_step"] = 0
        
        # Store configuration
        self.root_group.attrs["config"] = json.dumps(self.config.to_dict())
    
    def create_or_get_layer_group(self, layer_name: str, layer_type: str) -> zarr.Group:
        """
        Create or get a group for a layer.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of the layer
            
        Returns:
            Layer group
        """
        # Sanitize layer name for use as a path
        sanitized_name = layer_name.replace("/", "_").replace(":", "_")
        
        # Get or create layer group
        if sanitized_name not in self.layers_group:
            layer_group = self.layers_group.create_group(sanitized_name)
            # Set attributes
            layer_group.attrs["name"] = layer_name
            layer_group.attrs["type"] = layer_type
            # Add to tracked layers
            self.tracked_layers.add(layer_name)
        else:
            layer_group = self.layers_group[sanitized_name]
            # Make sure it's in tracked layers
            self.tracked_layers.add(layer_name)
            
        return layer_group
    
    def store_tensor(
        self,
        layer_group: zarr.Group,
        tensor_name: str,
        tensor_type: str,
        tensor_data: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Store a tensor in the Icechunk store.
        
        Args:
            layer_group: Layer group
            tensor_name: Name of the tensor
            tensor_type: Type of tensor (weights, gradients, non_trainable, activations)
            tensor_data: Tensor data
            step: Current step (if None, uses internal counter)
        """
        try:
            # Use provided step or current step
            current_step = step if step is not None else self.current_step
            
            # Create path for tensor
            if tensor_type not in layer_group:
                tensor_group = layer_group.create_group(tensor_type)
            else:
                tensor_group = layer_group[tensor_type]
            
            # Check for array in the tensor group
            create_new_array = False
            if tensor_name not in tensor_group:
                create_new_array = True
            else:
                # If array exists, check if we can write to it without resizing
                array = tensor_group[tensor_name]
                # Check if the array is writeable (might be read-only after a commit)
                try:
                    # Try to access store properties to check if it's accessible for writing
                    if array.shape[0] <= current_step and hasattr(array, '_store'):
                        # We'd need to resize which is problematic after commits, so create new
                        create_new_array = True
                except Exception:
                    # Any error suggests we should recreate
                    create_new_array = True
                    
            # If we need to create a new array (first time or after commit)
            if create_new_array:
                try:
                    # Calculate chunking strategy - pass tensor type for optimal chunking
                    chunks = self._determine_chunks((max(current_step + 1, 10),) + tensor_data.shape, tensor_type)
                    
                    # If array exists but we need to recreate it - ignore errors
                    # as we might be in a read-only state after commit
                    if tensor_name in tensor_group:
                        try:
                            del tensor_group[tensor_name]
                        except Exception as e:
                            # This might fail after a commit when store is read-only
                            # This is expected, just create the new array without deleting
                            print(f"Note: Could not delete existing array {tensor_name}, creating new one")
                            
                    # Create new array with initial shape large enough 
                    array = tensor_group.create_dataset(
                        tensor_name,
                        shape=(current_step + 1,) + tensor_data.shape,
                        chunks=chunks,
                        dtype=tensor_data.dtype
                    )
                    
                    # Track metadata about when this tensor type was created
                    if tensor_type == "gradients":
                        array.attrs["first_gradient_step"] = step
                        layer_group.attrs["has_gradients"] = True
                        print(f"Created new gradient array for {layer_group.name}/{tensor_name} with shape {array.shape}")
                except Exception as e:
                    # If we can't create the array either, we're likely after a commit
                    # and need to re-initialize the session
                    print(f"Could not create array {tensor_name}, trying to refresh session")
                    self._refresh_session_after_commit()
                    
                    # Try again with refreshed session - this is the key fix
                    try:
                        # Get reference to tensor group after session refresh
                        layer_group = self.layers_group[layer_group.name.split('/')[-1]]
                        tensor_group = layer_group[tensor_type]
                        
                        # Recalculate chunks with tensor type
                        chunks = self._determine_chunks((max(current_step + 1, 10),) + tensor_data.shape, tensor_type)
                        
                        # Now create the dataset
                        array = tensor_group.create_dataset(
                            tensor_name,
                            shape=(current_step + 1,) + tensor_data.shape,
                            chunks=chunks,
                            dtype=tensor_data.dtype
                        )
                        
                        # Add gradient metadata if applicable
                        if tensor_type == "gradients":
                            array.attrs["first_gradient_step"] = step
                            layer_group.attrs["has_gradients"] = True
                            print(f"Created new gradient array after session refresh for {layer_group.name}/{tensor_name}")
                    except Exception as e2:
                        print(f"Failed to create array even after session refresh: {e2}")
                        return
            
            # Only write data if the array shape is large enough
            if array.shape[0] > current_step:
                try:
                    # Store tensor data at the appropriate step
                    array[current_step] = tensor_data
                    
                    # For gradients, verify the write was successful
                    if tensor_type == "gradients":
                        # Try to verify the data was written by reading it back
                        try:
                            verification_data = array[current_step]
                            # Check if shapes match
                            if verification_data.shape != tensor_data.shape:
                                print(f"Warning: Gradient shape mismatch after write for {layer_group.name}/{tensor_name}")
                            # Additional verification could be added here
                        except Exception as e_verify:
                            print(f"Warning: Could not verify gradient write for {layer_group.name}/{tensor_name}: {e_verify}")
                    
                    # Log storage of gradient data if it's the first time and verbose is enabled
                    if tensor_type == "gradients" and self.config.get("verbose", False):
                        if not hasattr(self, "_logged_first_gradient") or tensor_name not in self._logged_first_gradient:
                            if not hasattr(self, "_logged_first_gradient"):
                                self._logged_first_gradient = set()
                            self._logged_first_gradient.add(tensor_name)
                            print(f"Stored first gradient for {layer_group.name}/{tensor_name} at step {current_step}")
                            
                            # Check and report statistics about gradient
                            try:
                                abs_mean = np.abs(tensor_data).mean()
                                if abs_mean < 1e-10:
                                    print(f"Warning: Very small gradient magnitude ({abs_mean:.2e}) for {layer_group.name}/{tensor_name}")
                                elif abs_mean > 100:
                                    print(f"Warning: Very large gradient magnitude ({abs_mean:.2e}) for {layer_group.name}/{tensor_name}")
                            except:
                                pass
                    
                    # For gradients, consider forcing a commit more frequently
                    if tensor_type == "gradients" and current_step > 0 and current_step % 5 == 0:
                        # Attempt to flush the store if supported
                        if hasattr(self.store, 'flush'):
                            try:
                                self.store.flush()
                            except Exception as e_flush:
                                print(f"Warning: Could not flush store after gradient write: {e_flush}")
                except Exception as e:
                    print(f"Error writing data to array {tensor_name}: {e}")
            
            # Check if we should commit changes based on the epoch (step)
            # Epochs are 0-indexed from Keras, but we display them as 1-indexed
            epoch = step # Use the step passed from the callback (which is the epoch)
            epoch_num = epoch + 1  # Convert to 1-indexed for comparison
            
            # For a commit_frequency of 5, we want commits at epoch numbers 5, 10, 15, 20
            # So we check if the 1-indexed epoch number is divisible by commit_frequency
            if (self.commit_frequency > 0 and epoch is not None and 
                epoch >= 0 and epoch_num % self.commit_frequency == 0 and 
                epoch != self.last_commit_step):
                
                # Use the 1-indexed epoch number in the commit message
                self.commit_changes(f"Commit at epoch {epoch_num}")
                self.last_commit_step = epoch
                
        except Exception as e:
            import traceback
            print(f"Error storing tensor {tensor_name} for layer {layer_group.name}:")
            traceback.print_exc()
    
    def get_compressor(self) -> Any:
        """Get the configured compressor."""
        # For Zarr V3 compatibility, return None instead of a compressor
        # The error "Expected a BytesBytesCodec" happens because Icechunk expects a
        # different compressor format than what numcodecs provides
        return None
    
    def _determine_chunks(self, shape: Tuple[int, ...], tensor_type: str = "weights") -> Tuple[int, ...]:
        """
        Determine optimal chunk size for a given tensor shape.
        
        Args:
            shape: Shape of the tensor with time dimension added
            tensor_type: Type of tensor (weights, gradients, activations)
            
        Returns:
            Tuple of chunk sizes for each dimension
        """
        # Get chunking configuration
        chunking = self.config.get("chunking", {})
        time_chunks = chunking.get("time_dimension", 10)
        target_size = chunking.get("target_chunk_size", 1000000)
        
        # Gradients may need different chunking strategy for more efficient storage
        # since they can change more rapidly between steps
        if tensor_type == "gradients":
            time_chunks = min(1, time_chunks)  # Store each gradient step separately by default
            
            # Adjust target size for gradients (potentially smaller chunks)
            if "gradient_chunk_size" in chunking:
                target_size = chunking.get("gradient_chunk_size", target_size)
        
        # Get element size (assuming float32)
        element_size = 4  # bytes per element
        
        # Always chunk along time dimension first
        chunks = [time_chunks]
        
        if len(shape) == 1:
            # Just time dimension, return as is
            return (time_chunks,)
        
        remaining_dimensions = shape[1:]
        
        spatial_dims = chunking.get("spatial_dimensions", "auto")
        if spatial_dims == "auto":
            # Calculate target elements per chunk based on target size
            target_elements = target_size / element_size
            
            # Divide target elements by time chunks
            target_elements_per_time = target_elements / time_chunks
            
            # Calculate spatial chunking based on actual shape
            total_elements = 1
            for dim in remaining_dimensions:
                total_elements *= dim
                
            if total_elements <= target_elements_per_time:
                # If the total size is smaller than target, use the full dimensions
                return (time_chunks,) + remaining_dimensions
            else:
                # Otherwise, calculate a proportional chunking
                ratio = (target_elements_per_time / total_elements) ** (1 / len(remaining_dimensions))
                for dim in remaining_dimensions:
                    chunk_size = max(1, int(dim * ratio))
                    chunks.append(chunk_size)
                return tuple(chunks)
        else:
            # Use user-specified spatial dimensions
            if isinstance(spatial_dims, list):
                return (time_chunks,) + tuple(spatial_dims)
            else:
                return (time_chunks,) + tuple([spatial_dims] * len(remaining_dimensions))
    
    def store_layer_metadata(
        self,
        layer_group: zarr.Group,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store metadata for a layer.
        
        Args:
            layer_group: Layer group
            metadata: Metadata dictionary
        """
        try:
            for key, value in metadata.items():
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, (tuple, list)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    value = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
                    
                # Store attribute
                layer_group.attrs[key] = value
        except Exception as e:
            print(f"Error storing metadata for layer {layer_group.name}: {e}")
    
    def store_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Store a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Current step (if None, uses internal counter)
        """
        try:
            current_step = step if step is not None else self.current_step
            
            # Create or recreate metric array
            if metric_name in self.metrics_group:
                # If array exists, check shape compatibility
                metric_array = self.metrics_group[metric_name]
                try:
                    if current_step >= metric_array.shape[0]:
                        # Try to resize - this might fail if shapes are incompatible
                        metric_array.resize((current_step + 1,))
                except Exception as e:
                    # If resize fails, delete and recreate the array
                    print(f"Recreating metric array {metric_name} due to shape mismatch")
                    try:
                        del self.metrics_group[metric_name]
                        # Create a new array with compatible shape
                        metric_array = self.metrics_group.create_dataset(
                            metric_name,
                            shape=(current_step + 1,),
                            chunks=(min(100, max(1, current_step + 1)),),
                            dtype=np.float32
                        )
                    except Exception as e2:
                        # If we can't delete/recreate, refresh the session and try again
                        print(f"Error recreating metric array, refreshing session: {e2}")
                        self._refresh_session_after_commit()
                        
                        # Try one more time with refreshed session
                        try:
                            if metric_name in self.metrics_group:
                                del self.metrics_group[metric_name]
                            metric_array = self.metrics_group.create_dataset(
                                metric_name,
                                shape=(current_step + 1,),
                                chunks=(min(100, max(1, current_step + 1)),),
                                dtype=np.float32
                            )
                        except Exception as e3:
                            print(f"Failed to recreate metric array even after session refresh: {e3}")
                            return
            else:
                # For new arrays, ensure we allocate enough space
                try:
                    metric_array = self.metrics_group.create_dataset(
                        metric_name,
                        shape=(current_step + 1,),
                        chunks=(min(100, max(1, current_step + 1)),),
                        dtype=np.float32
                    )
                except Exception as e:
                    # If we can't create, refresh the session and try again
                    print(f"Error creating metric array, refreshing session: {e}")
                    self._refresh_session_after_commit()
                    
                    # Try one more time with refreshed session
                    try:
                        metric_array = self.metrics_group.create_dataset(
                            metric_name,
                            shape=(current_step + 1,),
                            chunks=(min(100, max(1, current_step + 1)),),
                            dtype=np.float32
                        )
                    except Exception as e2:
                        print(f"Failed to create metric array even after session refresh: {e2}")
                        return
            
            # Store metric value
            try:
                metric_array[current_step] = value
            except Exception as e:
                print(f"Error writing metric value: {e}")
                # Try to refresh and write again if needed
                try:
                    self._refresh_session_after_commit()
                    # Need to get a fresh reference after session refresh
                    if metric_name in self.metrics_group:
                        metric_array = self.metrics_group[metric_name]
                        if metric_array.shape[0] > current_step:
                            metric_array[current_step] = value
                except Exception as e2:
                    print(f"Failed to write metric value even after session refresh: {e2}")
                
        except Exception as e:
            print(f"Error storing metric {metric_name}: {e}")
    
    def increment_step(self) -> None:
        """Increment the current step."""
        self.current_step += 1
        self.root_group.attrs["current_step"] = self.current_step
    
    def set_step(self, step: int) -> None:
        """
        Set the current step.
        
        Args:
            step: Step value (0-indexed epoch number from Keras)
        """
        try:
            # Add 1 to align with 1-indexed epoch numbers shown to users
            self.current_step = step + 1
            self.root_group.attrs["current_step"] = self.current_step
        except Exception as e:
            # If we can't set the attribute, the store might be read-only after a commit
            print(f"Error setting current_step attribute: {e}")
            print("Attempting to refresh session and try again")
            try:
                self._refresh_session_after_commit()
                # Now try again with the fresh session
                self.root_group.attrs["current_step"] = self.current_step
                print("Successfully set current_step after refreshing session")
            except Exception as e2:
                print(f"Failed to set current_step even after session refresh: {e2}")
                # Continue anyway, as this isn't critical for functionality
    
    def _tag_exists(self, tag_name: str) -> bool:
        """Check if a tag already exists in the repository."""
        try:
            self.repo.lookup_tag(tag_name)
            return True
        except:
            return False
    
    def commit_changes(self, message=None):
        """
        Commit the current transaction to create a new snapshot.
        
        Args:
            message: Commit message
            
        Returns:
            Snapshot ID
        """
        # Use the current step from the storage manager
        step_to_commit = self.current_step
        
        if not message:
            # Use the actual 1-indexed epoch number in the message
            message = f"Update at epoch {step_to_commit}"
        # Don't modify the epoch number in the commit message anymore as it's already correct
        
        if self.config.get("icechunk", {}).get("verbose", False):
            print(f"Committing changes with message: '{message}'")
            print(f"Current step: {step_to_commit}")
        
        # Commit the current session
        try:
            # Ensure all data is flushed before committing
            if hasattr(self.store, 'flush'):
                self.store.flush()
                
            # Do the actual commit
            snapshot_id = self.session.commit(message)
            
            if self.config.get("icechunk", {}).get("verbose", False):
                print(f"Successfully committed snapshot: {snapshot_id}")
            
            # Always refresh the session after a commit to ensure we can keep writing
            self._refresh_session_after_commit()
            
            # Tag important snapshots if configured
            tag_snapshots = self.config.get("icechunk", {}).get("tag_snapshots", False)
            if tag_snapshots:
                # Use 1-indexed step count in tag name for consistency with epoch display
                tag_name = f"epoch_{step_to_commit}"
                
                # Only create the tag if it doesn't already exist
                if tag_name not in self.created_tags and not self._tag_exists(tag_name):
                    try:
                        self.repo.create_tag(tag_name, snapshot_id=snapshot_id)
                        self.created_tags.add(tag_name)
                        if self.config.get("icechunk", {}).get("verbose", False):
                            print(f"Created tag: {tag_name}")
                    except Exception as e:
                        print(f"Warning: Could not create tag {tag_name}: {e}")
                
            return snapshot_id
        except Exception as e:
            print(f"Error committing changes for step {step_to_commit}: {e}")
            # Try to reopen the session
            try:
                self._refresh_session_after_commit()
                return None
            except Exception as e:
                print(f"Error reopening session: {e}")
                return None
    
    def _refresh_session_after_commit(self):
        """
        Refresh the session after a commit to ensure we can continue writing.
        This is necessary because Icechunk creates a read-only view after commit.
        """
        try:
            # Don't try to close the session - Icechunk Session objects don't have a close method
            
            # Create a new session
            self.session = self.repo.writable_session("main")
            self.store = self.session.store
            
            # Reinitialize Zarr groups with the new store
            self._initialize_zarr_groups()
            
            print("Successfully refreshed session after commit")
        except Exception as e:
            print(f"Error refreshing session: {e}")
    
    def close(self) -> None:
        """Close the storage manager."""
        # Commit any outstanding changes
        if hasattr(self, 'session') and self.session is not None:
            try:
                # Only commit if we have made changes since the last commit
                if self.current_step != self.last_commit_step:
                    print(f"[ParamLake Debug] Performing final commit on close.")
                    final_step = self.current_step # Use the last step recorded
                    print(f"[ParamLake Debug] Attempting commit: 'Final update on close' at step {final_step}, last commit was step {self.last_commit_step}")
                    self.commit_changes(f"Final update on close") # commit_changes uses self.current_step
                    self.last_commit_step = final_step # Update last commit step after successful final commit
                else:
                    print("[ParamLake Debug] No changes since last commit, skipping final commit.")
            except Exception as e:
                print(f"Error committing final changes on close: {e}")
                
        # Mark the run as completed by setting final_step
        if hasattr(self, 'root_group') and self.root_group is not None:
            try:
                print(f"[ParamLake Debug] Setting final_step attribute to {self.current_step}")
                self.root_group.attrs["final_step"] = self.current_step
                if hasattr(self.store, 'flush'):
                    self.store.flush() # Ensure attributes are written
                print("[ParamLake Debug] final_step attribute set.")
            except Exception as e:
                print(f"Error setting final_step attribute: {e}")
                
        print("[ParamLake Debug] Storage manager close sequence finished.")
            
    def get_tracked_layers(self) -> List[str]:
        """Get list of tracked layers."""
        return list(self.tracked_layers)

    def store_optimizer_state(
        self,
        optimizer_weights: List[np.ndarray],
        step: Optional[int] = None,
    ) -> None:
        """
        Store the optimizer's state (weights).

        Args:
            optimizer_weights: List of NumPy arrays representing optimizer state.
            step: Current step.
        """
        try:
            current_step = step if step is not None else self.current_step
            
            step_group_name = f"step_{current_step}"
            # For IceChunk, we need to handle potential read-only state after commit.
            # We ensure the group exists or create it. If creation fails, refresh session.
            try:
                if step_group_name not in self.optimizer_states_group:
                    step_group = self.optimizer_states_group.create_group(step_group_name)
                else:
                    step_group = self.optimizer_states_group[step_group_name]
            except Exception as e_group_create:
                print(f"Error creating/accessing step group {step_group_name} for optimizer state: {e_group_create}. Refreshing session.")
                self._refresh_session_after_commit()
                # After refresh, self.optimizer_states_group will be updated.
                if step_group_name not in self.optimizer_states_group:
                    step_group = self.optimizer_states_group.create_group(step_group_name)
                else:
                    step_group = self.optimizer_states_group[step_group_name]

            for idx, weight_data in enumerate(optimizer_weights):
                weight_array_name = f"weight_{idx}"
                try:
                    # Optimizer states are discrete per step, so create_dataset with overwrite is appropriate.
                    # IceChunk handles the backend Zarr array creation.
                    # The compressor is None for IceChunk as it manages its own compression.
                    step_group.create_dataset(
                        weight_array_name,
                        data=weight_data,
                        chunks=True,  # Let Zarr (via IceChunk) decide chunking for these individual arrays
                        dtype=weight_data.dtype,
                        overwrite=True,
                        # compressor=self.get_compressor("optimizer_state") # IceChunk handles compression
                    )
                    # step_group[weight_array_name].attrs["dtype"] = str(weight_data.dtype) # Not strictly needed if dtype is in array
                except Exception as e_array_create:
                    print(f"Error storing optimizer weight {weight_array_name} for step {current_step}: {e_array_create}. Attempting session refresh.")
                    self._refresh_session_after_commit()
                    # Retry after refresh for this specific weight
                    if step_group_name not in self.optimizer_states_group: # Re-check step_group after refresh
                        step_group = self.optimizer_states_group.create_group(step_group_name)
                    else:
                        step_group = self.optimizer_states_group[step_group_name]
                    
                    step_group.create_dataset(
                        weight_array_name,
                        data=weight_data,
                        chunks=True,
                        dtype=weight_data.dtype,
                        overwrite=True,
                    )

        except Exception as e:
            import traceback
            print(f"Error storing optimizer state at step {step}:")
            traceback.print_exc()

    def store_optimizer_config(
        self,
        optimizer_name: str,
        optimizer_config: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Store the optimizer's configuration in IceChunk.
        The config is stored once per run, identified by optimizer_name.
        The step argument is ignored.
        """
        try:
            # Ensure the optimizer_info group exists, refreshing session if necessary
            try:
                if "optimizer_info" not in self.root_group:
                    self.optimizer_info_group = self.root_group.create_group("optimizer_info")
                else:
                    self.optimizer_info_group = self.root_group["optimizer_info"]
            except Exception as e_info_create:
                print(f"Error accessing/creating optimizer_info group: {e_info_create}. Refreshing session.")
                self._refresh_session_after_commit() # This re-initializes self.root_group and sub-groups
                # self.optimizer_info_group should now be valid or recreated by _initialize_zarr_groups
                if "optimizer_info" not in self.root_group: # Re-check after refresh
                     self.optimizer_info_group = self.root_group.create_group("optimizer_info")
                else:
                    self.optimizer_info_group = self.root_group["optimizer_info"]


            # Store config as attributes of a group named after the optimizer
            # Need to handle potential read-only state after commit for this group too.
            opt_config_group_name = optimizer_name
            try:
                if opt_config_group_name not in self.optimizer_info_group:
                    opt_config_group = self.optimizer_info_group.create_group(opt_config_group_name)
                else:
                    opt_config_group = self.optimizer_info_group[opt_config_group_name]
            except Exception as e_group_create:
                print(f"Error accessing/creating optimizer config group '{opt_config_group_name}': {e_group_create}. Refreshing session.")
                self._refresh_session_after_commit()
                if opt_config_group_name not in self.optimizer_info_group: # Re-check after refresh
                    opt_config_group = self.optimizer_info_group.create_group(opt_config_group_name)
                else:
                    opt_config_group = self.optimizer_info_group[opt_config_group_name]

            # Clear existing attributes before writing new ones
            opt_config_group.attrs.clear()
            
            for key, value in optimizer_config.items():
                try:
                    opt_config_group.attrs[key] = value
                except TypeError:
                    try:
                        opt_config_group.attrs[key] = json.dumps(value)
                    except Exception as e_json:
                        print(f"Warning: Could not store optimizer config key '{key}' for '{optimizer_name}'. Value: {value}. Error: {e_json}")
                        opt_config_group.attrs[key] = str(value) # Fallback to string
            
            # IceChunk requires a commit to persist attribute changes if they are part of a new snapshot.
            # However, this method might be called frequently. 
            # Committing here could be too frequent. The main commit_changes() method handles periodic commits.
            # If this config is meant to be set once, the next natural commit will save it.
            # If it can change and needs immediate persistence, a targeted commit would be needed.
            # For now, relying on the periodic commit_changes.

        except Exception as e:
            import traceback
            print(f"Error storing optimizer configuration for {optimizer_name}:")
            traceback.print_exc()

    def save_checkpoint(
        self,
        weights_data: List[np.ndarray],
        weights_names: List[str],
        weights_shapes: List[Tuple[int, ...]],
        optimizer_data: List[np.ndarray],
        optimizer_config: Optional[Dict[str, Any]],
        compile_config: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        step: Optional[int] = None,
    ) -> str:
        """
        Save a model checkpoint using IceChunk's native snapshot capabilities.
        
        Args:
            weights_data: List of weight arrays
            weights_names: List of weight names
            weights_shapes: List of weight shapes
            optimizer_data: List of optimizer state arrays
            optimizer_config: Optimizer configuration
            compile_config: Model compilation configuration
            metadata: Checkpoint metadata
            step: Current step (if None, uses internal counter)
            
        Returns:
            Checkpoint ID/snapshot ID
        """
        # Use provided step or current step
        current_step = step if step is not None else self.current_step
        
        # Generate checkpoint name
        checkpoint_name = metadata.get('name', None)
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{current_step}"
        
        # Add timestamp and step to metadata
        metadata["timestamp"] = datetime.now().isoformat()
        metadata["step"] = current_step
        
        # Create a new checkpoint entry in the checkpoint index
        checkpoint_index = self.checkpoints_group.create_group(checkpoint_name)
        
        # Store metadata in the checkpoint index
        for key, value in metadata.items():
            try:
                checkpoint_index.attrs[key] = value
            except Exception as e:
                print(f"Error storing checkpoint metadata key {key}: {e}")
                # Store as string if needed
                checkpoint_index.attrs[key] = str(value)
        
        # Store the weights in a temporary group that will be captured in the snapshot
        weights_group = self.root_group.create_group(f"temp_checkpoint_{int(time.time())}")
        
        # Store weights temporarily
        for i, (weight_data, weight_name, weight_shape) in enumerate(zip(weights_data, weights_names, weights_shapes)):
            try:
                # Store weight data
                weight_array = weights_group.create_dataset(
                    f"weight_{i}",
                    data=weight_data,
                    chunks=True
                )
                
                # Store metadata
                weight_array.attrs["name"] = weight_name
                weight_array.attrs["shape"] = str(weight_shape)  # Convert tuple to string
            except Exception as e:
                print(f"Error storing weight {i}: {e}")
        
        # Store optimizer data if available
        if optimizer_data and len(optimizer_data) > 0:
            try:
                optimizer_group = weights_group.create_group("optimizer")
                
                # Store optimizer config
                if optimizer_config:
                    try:
                        # Try to serialize the optimizer config
                        optimizer_group.attrs["config"] = json.dumps(optimizer_config)
                    except (TypeError, ValueError):
                        # If serialization fails, store config entries as strings
                        for key, value in optimizer_config.items():
                            optimizer_group.attrs[f"config_{key}"] = str(value)
                
                # Store optimizer weights
                for i, opt_weight in enumerate(optimizer_data):
                    optimizer_group.create_dataset(
                        f"weight_{i}",
                        data=opt_weight,
                        chunks=True
                    )
            except Exception as e:
                print(f"Error storing optimizer data: {e}")
        
        # Store compile config if available
        if compile_config:
            try:
                for key, value in compile_config.items():
                    try:
                        # Try to serialize complex values
                        if isinstance(value, dict):
                            checkpoint_index.attrs[f"compile_{key}"] = json.dumps(value)
                        else:
                            checkpoint_index.attrs[f"compile_{key}"] = str(value)
                    except Exception as e:
                        print(f"Error storing compile config key {key}: {e}")
                        checkpoint_index.attrs[f"compile_{key}"] = str(value)
            except Exception as e:
                print(f"Error storing compile config: {e}")
        
        # Make sure everything is flushed
        if hasattr(self.store, 'flush'):
            self.store.flush()
        
        # Commit to create a snapshot
        try:
            description = metadata.get('description', f"Checkpoint at step {current_step}")
            snapshot_id = self.session.commit(f"Checkpoint: {description}")
            
            # Store the snapshot ID in the checkpoint index
            checkpoint_index.attrs["snapshot_id"] = snapshot_id
            
            # Create a tag for this checkpoint if configured
            tag_name = f"checkpoint_{current_step}"
            try:
                # Only create the tag if it doesn't exist
                if not self._tag_exists(tag_name):
                    self.repo.create_tag(tag_name, snapshot_id=snapshot_id)
                    self.created_tags.add(tag_name)
            except Exception as e:
                print(f"Warning: Could not create tag for checkpoint: {e}")
            
            # Track the checkpoint
            self.checkpoint_snapshots[checkpoint_name] = snapshot_id
            
            # Clean up temporary group in a new session
            self._refresh_session_after_commit()
            
            # Remove temporary group using the new session
            try:
                if f"temp_checkpoint_{int(time.time())}" in self.root_group:
                    del self.root_group[f"temp_checkpoint_{int(time.time())}"]
            except Exception as e:
                print(f"Warning: Could not clean up temporary checkpoint group: {e}")
            
            return snapshot_id
        except Exception as e:
            print(f"Error committing checkpoint: {e}")
            # Clean up the temporary checkpoint group
            try:
                if f"temp_checkpoint_{int(time.time())}" in self.root_group:
                    del self.root_group[f"temp_checkpoint_{int(time.time())}"]
            except:
                pass
            return None
    
    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load a checkpoint by ID.
        
        Args:
            checkpoint_id: ID of the checkpoint or snapshot
            
        Returns:
            Dictionary with checkpoint data
        """
        # First, check if the ID is a snapshot ID
        snapshot_id = checkpoint_id
        
        # Check if checkpoint_id is a checkpoint name rather than a snapshot ID
        if checkpoint_id in self.checkpoints_group:
            checkpoint_group = self.checkpoints_group[checkpoint_id]
            if "snapshot_id" in checkpoint_group.attrs:
                snapshot_id = checkpoint_group.attrs["snapshot_id"]
        
        try:
            # Create a readonly session with this snapshot
            checkpoint_session = self.repo.readonly_session(snapshot_id=snapshot_id)
            checkpoint_store = checkpoint_session.store
            checkpoint_root = zarr.open_group(checkpoint_store, mode="r")
            
            # Look for checkpoint data in the snapshot
            # First try to find the temporary checkpoint group
            temp_groups = [k for k in checkpoint_root.keys() if k.startswith("temp_checkpoint_")]
            
            if temp_groups:
                # Use the first temp group found
                checkpoint_group = checkpoint_root[temp_groups[0]]
            else:
                # Fallback to looking for data in the main groups
                checkpoint_group = checkpoint_root
            
            # Load weights
            weights_data = []
            weights_names = []
            weights_shapes = []
            
            # Find all weight arrays in the checkpoint
            weight_keys = sorted(
                [k for k in checkpoint_group.keys() if k.startswith("weight_")],
                key=lambda k: int(k.split("_")[1]) if "_" in k else 0
            )
            
            for key in weight_keys:
                try:
                    weight = checkpoint_group[key]
                    # Load weight data
                    weights_data.append(weight[:])
                    # Load weight name
                    if "name" in weight.attrs:
                        weights_names.append(weight.attrs["name"])
                    else:
                        weights_names.append(key)
                    # Load weight shape
                    if "shape" in weight.attrs:
                        shape_str = weight.attrs["shape"]
                        # Try to parse shape string if it was stored as a string
                        if isinstance(shape_str, str):
                            try:
                                import ast
                                shape = ast.literal_eval(shape_str)
                                weights_shapes.append(shape)
                            except:
                                weights_shapes.append(weight.shape)
                        else:
                            weights_shapes.append(shape_str)
                    else:
                        weights_shapes.append(weight.shape)
                except Exception as e:
                    print(f"Error loading weight {key}: {e}")
            
            # Load optimizer data if available
            optimizer_data = []
            optimizer_config = None
            
            if "optimizer" in checkpoint_group:
                optimizer_group = checkpoint_group["optimizer"]
                
                # Load optimizer config
                if "config" in optimizer_group.attrs:
                    try:
                        optimizer_config = json.loads(optimizer_group.attrs["config"])
                    except (json.JSONDecodeError, TypeError):
                        # Try to build optimizer config from individual attributes
                        optimizer_config = {}
                        for key in optimizer_group.attrs:
                            if key.startswith("config_"):
                                config_key = key[7:]  # Remove "config_" prefix
                                optimizer_config[config_key] = optimizer_group.attrs[key]
                
                # Load optimizer weights
                opt_keys = sorted(
                    [k for k in optimizer_group.keys() if k.startswith("weight_")],
                    key=lambda k: int(k.split("_")[1]) if "_" in k else 0
                )
                
                for key in opt_keys:
                    try:
                        optimizer_data.append(optimizer_group[key][:])
                    except Exception as e:
                        print(f"Error loading optimizer weight {key}: {e}")
            
            # Look for compile config
            compile_config = {}
            
            # Check checkpoint index for compile config
            if checkpoint_id in self.checkpoints_group:
                index_group = self.checkpoints_group[checkpoint_id]
                for key in index_group.attrs:
                    if key.startswith("compile_"):
                        config_key = key[8:]  # Remove "compile_" prefix
                        try:
                            if index_group.attrs[key].startswith('{'):
                                compile_config[config_key] = json.loads(index_group.attrs[key])
                            else:
                                compile_config[config_key] = index_group.attrs[key]
                        except:
                            compile_config[config_key] = index_group.attrs[key]
            
            # Get metadata from checkpoint index
            metadata = {}
            if checkpoint_id in self.checkpoints_group:
                index_group = self.checkpoints_group[checkpoint_id]
                for key in index_group.attrs:
                    if not key.startswith("compile_"):
                        metadata[key] = index_group.attrs[key]
            
            return {
                "weights_data": weights_data,
                "weights_names": weights_names,
                "weights_shapes": weights_shapes,
                "optimizer_data": optimizer_data,
                "optimizer_config": optimizer_config,
                "compile_config": compile_config if compile_config else None,
                "metadata": metadata
            }
            
        except Exception as e:
            print(f"Error loading checkpoint from snapshot {snapshot_id}: {e}")
            raise ValueError(f"Could not load checkpoint {checkpoint_id}")
    
    def load_checkpoint_by_step(self, step: int) -> Dict[str, Any]:
        """
        Load a checkpoint by step number.
        
        Args:
            step: Step number
            
        Returns:
            Dictionary with checkpoint data
        """
        # Find checkpoints for this step in the index
        matching_checkpoints = []
        
        for checkpoint_id in self.checkpoints_group.keys():
            checkpoint_group = self.checkpoints_group[checkpoint_id]
            if checkpoint_group.attrs.get("step") == step:
                matching_checkpoints.append(checkpoint_id)
        
        if not matching_checkpoints:
            # Try to find by tag
            tag_name = f"checkpoint_{step}"
            try:
                snapshot_id = self.repo.lookup_tag(tag_name)
                # Use snapshot_id directly
                return self.load_checkpoint(snapshot_id)
            except:
                raise ValueError(f"No checkpoint found for step {step}")
        
        # If multiple checkpoints for this step, use the latest one
        if len(matching_checkpoints) > 1:
            # Sort by timestamp if available, otherwise by ID
            latest_checkpoint = max(
                matching_checkpoints,
                key=lambda cid: self.checkpoints_group[cid].attrs.get("timestamp", cid)
            )
        else:
            latest_checkpoint = matching_checkpoints[0]
        
        return self.load_checkpoint(latest_checkpoint)
    
    def load_latest_checkpoint(self) -> Dict[str, Any]:
        """
        Load the latest checkpoint.
        
        Returns:
            Dictionary with checkpoint data
        """
        # First, try to find the latest checkpoint from the index
        if self.checkpoints_group.keys():
            # Find the latest checkpoint by timestamp or step
            latest_checkpoint = None
            latest_timestamp = None
            latest_step = -1
            
            for checkpoint_id in self.checkpoints_group.keys():
                checkpoint_group = self.checkpoints_group[checkpoint_id]
                
                # First try to use timestamp
                if "timestamp" in checkpoint_group.attrs:
                    timestamp = checkpoint_group.attrs["timestamp"]
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_checkpoint = checkpoint_id
                # Fallback to using step
                elif "step" in checkpoint_group.attrs:
                    step = checkpoint_group.attrs["step"]
                    if step > latest_step:
                        latest_step = step
                        latest_checkpoint = checkpoint_id
            
            # If found a checkpoint in the index, load it
            if latest_checkpoint is not None:
                return self.load_checkpoint(latest_checkpoint)
        
        # If no checkpoints in index or loading failed, try to find latest tag
        checkpoint_tags = []
        
        # Get all tags that start with "checkpoint_"
        for tag_name in self.created_tags:
            if tag_name.startswith("checkpoint_"):
                try:
                    step = int(tag_name.split("_")[1])
                    checkpoint_tags.append((step, tag_name))
                except:
                    pass
        
        if checkpoint_tags:
            # Sort by step number
            checkpoint_tags.sort(reverse=True)
            latest_tag = checkpoint_tags[0][1]
            
            try:
                # Lookup snapshot from tag
                snapshot_id = self.repo.lookup_tag(latest_tag)
                return self.load_checkpoint(snapshot_id)
            except Exception as e:
                print(f"Error loading checkpoint from tag {latest_tag}: {e}")
        
        # Last resort: get the current head of the main branch
        try:
            head_snapshot_id = self.repo.lookup_branch("main")
            return self.load_checkpoint(head_snapshot_id)
        except Exception as e:
            print(f"Error loading latest snapshot: {e}")
            raise ValueError("No checkpoints found")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of dictionaries with checkpoint metadata
        """
        checkpoints = []
        
        # First, list checkpoints from the index
        for checkpoint_id in self.checkpoints_group.keys():
            try:
                checkpoint_group = self.checkpoints_group[checkpoint_id]
                metadata = dict(checkpoint_group.attrs)
                metadata["id"] = checkpoint_id
                
                # Add snapshot ID if available
                if "snapshot_id" in metadata:
                    metadata["snapshot_id"] = metadata["snapshot_id"]
                
                checkpoints.append(metadata)
            except Exception as e:
                print(f"Error listing checkpoint {checkpoint_id}: {e}")
        
        # Also look for checkpoint tags that might not be in the index
        checkpoint_tags = []
        
        # List all tags
        try:
            all_tags = []
            for tag_name in self.created_tags:
                if tag_name.startswith("checkpoint_"):
                    all_tags.append(tag_name)
            
            # Look up each tag
            for tag_name in all_tags:
                try:
                    snapshot_id = self.repo.lookup_tag(tag_name)
                    
                    # Check if this snapshot is already in our list
                    already_included = False
                    for checkpoint in checkpoints:
                        if checkpoint.get("snapshot_id") == snapshot_id:
                            already_included = True
                            break
                    
                    if not already_included:
                        # Get step from tag name
                        step = None
                        try:
                            if "_" in tag_name:
                                step = int(tag_name.split("_")[1])
                        except:
                            pass
                        
                        # Create metadata entry
                        tag_metadata = {
                            "id": tag_name,
                            "snapshot_id": snapshot_id,
                            "step": step,
                            "from_tag": True
                        }
                        
                        # Try to get more info from snapshot
                        try:
                            snapshot = self.repo.get_snapshot(snapshot_id)
                            tag_metadata["timestamp"] = snapshot.written_at.isoformat()
                            tag_metadata["message"] = snapshot.message
                        except:
                            pass
                        
                        checkpoints.append(tag_metadata)
                except Exception as e:
                    print(f"Error processing tag {tag_name}: {e}")
            
        except Exception as e:
            print(f"Error listing checkpoint tags: {e}")
        
        # Sort by step, then timestamp
        return sorted(
            checkpoints,
            key=lambda c: (c.get("step", 0), c.get("timestamp", ""))
        )

    # --- New Git-like feature implementations ---

    def get_snapshot_id_for_reference(self, reference: str) -> Optional[str]:
        """Resolves a branch name, tag name, or snapshot ID to a snapshot ID."""
        if not self.repo: return None
        try: # Is it a branch?
            return self.repo.lookup_branch(reference)
        except icechunk.IcechunkError: # Not a branch, or error
            pass
        try: # Is it a tag?
            return self.repo.lookup_tag(reference)
        except icechunk.IcechunkError:
            pass
        try: # Is it a direct snapshot ID?
            self.repo.get_snapshot(reference) # Validate if it's a snapshot ID
            return reference
        except icechunk.IcechunkError:
            pass
        print(f"Warning: Reference '{reference}' not found as a branch, tag, or snapshot ID.")
        return None

    def commit_model_state(
        self,
        model_parameters: Dict[str, np.ndarray],
        message: str,
        branch_name: str,
        author: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Commits the model's parameters as a new snapshot on the given branch."""
        if not self.repo: raise ConnectionError("Repository not open.")

        # Ensure the branch exists, or create it from main's head if it doesn't and is not main
        if branch_name != 'main':
            try:
                self.repo.lookup_branch(branch_name)
            except icechunk.IcechunkError: # Branch does not exist
                print(f"Branch '{branch_name}' does not exist. Creating it from main HEAD.")
                main_head_id = self.repo.lookup_branch('main')
                self.repo.create_branch(branch_name, snapshot_id=main_head_id)
        
        session = self.repo.writable_session(branch_name)
        store = session.store
        params_group_name = "parameters" # Define a group for model parameters

        try:
            if params_group_name not in store:
                params_group = store.create_group(params_group_name)
            else:
                params_group = store[params_group_name]

            for name, data in model_parameters.items():
                # Sanitize name for Zarr
                safe_name = name.replace("/", "_").replace(":", "_")
                if safe_name in params_group:
                    # Overwrite existing array
                    params_group[safe_name][...] = data
                else:
                    # Create new array
                    params_group.create_dataset(safe_name, data=data, chunks=True, compressor=None) # Icechunk handles compression
            
            # Add commit metadata as attributes to the snapshot (via session commit)
            # Icechunk's commit message is the primary place. Author/timestamp are auto by Icechunk.
            # Additional metadata can be stored in a separate metadata object if needed,
            # or as attributes on a specific group within this snapshot if Icechunk's commit doesn't support rich metadata.
            # For now, relying on message and Icechunk's built-in timestamp/author (if any).
            
            snapshot_id = session.commit(message)
            # self._refresh_session_after_commit() # Not strictly needed after every commit if session remains usable
                                                # but good practice if next ops might conflict.
                                                # The design doc says IceChunk session becomes read-only.
                                                # So, this IS needed.
            self._refresh_session_after_commit() # To get a new writable session
            
            # Update current_step if this is on the main lineage of tracking
            # This logic might be better handled at the Repo class or callback level.
            # For now, just commit.
            
            return snapshot_id
        except Exception as e:
            session.abort() # Abort on error
            self._refresh_session_after_commit() # Still refresh to clean up session state
            raise RuntimeError(f"Failed to commit model state to branch '{branch_name}': {e}")

    def create_branch(self, branch_name: str, from_reference: Optional[str] = None) -> None:
        """Creates a new branch, optionally from a specific snapshot or another reference."""
        if not self.repo: raise ConnectionError("Repository not open.")
        
        snapshot_id_to_branch_from = None
        if from_reference:
            snapshot_id_to_branch_from = self.get_snapshot_id_for_reference(from_reference)
            if not snapshot_id_to_branch_from:
                raise ValueError(f"Reference '{from_reference}' for branching not found.")
        else: # Default to current HEAD of 'main' branch if no reference
            try:
                snapshot_id_to_branch_from = self.repo.lookup_branch('main')
            except icechunk.IcechunkError: # If main doesn't exist (e.g. new repo, no commits yet)
                snapshot_id_to_branch_from = self.repo.initial_snapshot().id

        try:
            self.repo.create_branch(branch_name, snapshot_id=snapshot_id_to_branch_from)
        except icechunk.IcechunkError as e:
            raise ValueError(f"Failed to create branch '{branch_name}': {e}")

    def list_branches(self) -> List[str]:
        """Lists all branches in the repository."""
        if not self.repo: return []
        try:
            # Icechunk returns a BranchView, convert to list of names
            return [branch.name for branch in self.repo.branches()]
        except Exception as e:
            print(f"Error listing branches: {e}")
            return []
            
    def delete_branch(self, branch_name: str) -> None:
        """Deletes a branch."""
        if not self.repo: return
        try:
            self.repo.delete_branch(branch_name)
        except icechunk.IcechunkError as e:
            # Catch if branch does not exist, etc.
            raise ValueError(f"Failed to delete branch '{branch_name}': {e}")

    def create_tag(self, tag_name: str, reference: str, message: Optional[str] = None) -> None:
        """Creates a tag pointing to a specific reference (snapshot_id, branch, or tag)."""
        if not self.repo: raise ConnectionError("Repository not open.")
        snapshot_id = self.get_snapshot_id_for_reference(reference)
        if not snapshot_id:
            raise ValueError(f"Reference '{reference}' for tagging not found.")
        try:
            # Icechunk tags don't have a separate message AFAIK, message is part of snapshot
            self.repo.create_tag(tag_name, snapshot_id=snapshot_id)
        except icechunk.IcechunkError as e:
            raise ValueError(f"Failed to create tag '{tag_name}': {e}")

    def list_tags(self) -> Dict[str, str]:
        """Lists all tags and their associated snapshot IDs."""
        if not self.repo: return {}
        try:
            # Icechunk returns a TagView, convert to dict
            return {tag.name: tag.snapshot_id for tag in self.repo.tags()}
        except Exception as e:
            print(f"Error listing tags: {e}")
            return {}

    def delete_tag(self, tag_name: str) -> None:
        """Deletes a tag."""
        if not self.repo: return
        try:
            self.repo.delete_tag(tag_name)
        except icechunk.IcechunkError as e:
            raise ValueError(f"Failed to delete tag '{tag_name}': {e}")

    def get_history(self, reference: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Gets the commit history for a given reference (branch, tag, or snapshot_id)."""
        if not self.repo: return []
        
        snapshot_id_to_log = None
        if reference:
            snapshot_id_to_log = self.get_snapshot_id_for_reference(reference)
            if not snapshot_id_to_log:
                print(f"Warning: Reference '{reference}' not found. Showing history for default (main).")
                snapshot_id_to_log = self.repo.lookup_branch('main') # Fallback
        else: # Default to main branch HEAD
            snapshot_id_to_log = self.repo.lookup_branch('main')

        history = []
        try:
            ancestry = self.repo.ancestry(snapshot_id=snapshot_id_to_log)
            count = 0
            for ancestor_snapshot_info in ancestry:
                if limit is not None and count >= limit:
                    break
                
                # Get tags pointing to this snapshot
                tags_for_snapshot = []
                for tag in self.repo.tags():
                    if tag.snapshot_id == ancestor_snapshot_info.id:
                        tags_for_snapshot.append(tag.name)

                history_entry = {
                    "id": ancestor_snapshot_info.id,
                    "message": ancestor_snapshot_info.message,
                    "timestamp": ancestor_snapshot_info.written_at.isoformat() if ancestor_snapshot_info.written_at else None,
                    "author": None, # Icechunk snapshots don't store author directly in SnapshotInfo; could be in message/metadata
                    "tags": tags_for_snapshot
                }
                history.append(history_entry)
                count += 1
        except Exception as e:
            print(f"Error retrieving history for '{reference}': {e}")
        return history
        
    def load_parameters_from_snapshot(self, reference: str) -> Dict[str, np.ndarray]:
        """Loads all model parameters from a given reference (snapshot ID, branch, or tag)."""
        if not self.repo: raise ConnectionError("Repository not open.")
        snapshot_id = self.get_snapshot_id_for_reference(reference)
        if not snapshot_id:
            raise ValueError(f"Reference '{reference}' not found.")

        params: Dict[str, np.ndarray] = {}
        try:
            # Open a read-only session for the specific snapshot
            read_session = self.repo.readonly_session(snapshot_id=snapshot_id)
            store = read_session.store
            params_group_name = "parameters"

            if params_group_name in store:
                params_group = store[params_group_name]
                for name in params_group.keys():
                    params[name] = params_group[name][...] # Load the full array
            else:
                print(f"Warning: No '{params_group_name}' group found in snapshot '{snapshot_id}'.")
            return params
        except Exception as e:
            raise RuntimeError(f"Failed to load parameters from snapshot '{snapshot_id}': {e}")

    # Stubs for future features
    def diff_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Diffing snapshots is not yet implemented for IcechunkStorageManager.")

    def merge_branches(
        self, 
        source_branch: str, 
        target_branch: str, 
        strategy: str = 'manual',
        commit_message: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError("Merging branches is not yet implemented for IcechunkStorageManager.")

    def import_model_from_path(
        self, 
        source_path: str, 
        source_format: str, 
        branch_name: str, 
        commit_message: Optional[str] = None
    ) -> str:
        # Basic HDF5 import as a starting point
        if source_format.lower() == 'hdf5':
            try:
                import h5py
                model_params = {}
                with h5py.File(source_path, 'r') as hf:
                    def extract_weights(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            # Use a sanitized version of the HDF5 path as the parameter name
                            safe_name = name.replace("/", "_").replace(":", "_")
                            model_params[safe_name] = obj[()] # Read numpy array
                    hf.visititems(extract_weights)
                
                if not model_params:
                    raise ValueError("No datasets found in HDF5 file or failed to extract.")

                msg = commit_message or f"Import model from HDF5: {os.path.basename(source_path)}"
                return self.commit_model_state(model_params, msg, branch_name)
            except ImportError:
                raise ImportError("h5py is required to import HDF5 files. pip install h5py")
            except Exception as e:
                raise RuntimeError(f"Failed to import HDF5 model: {e}")
        else:
            raise NotImplementedError(f"Import for format '{source_format}' is not yet implemented.") 