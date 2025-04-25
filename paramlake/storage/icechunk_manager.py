"""
Icechunk storage manager for ParamLake.
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
                    except Exception as e2:
                        print(f"Failed to create array even after session refresh: {e2}")
                        return
            
            # Only write data if the array shape is large enough
            if array.shape[0] > current_step:
                try:
                    # Store tensor data at the appropriate step
                    array[current_step] = tensor_data
                    
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