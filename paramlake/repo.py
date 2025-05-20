"""
Provides the Repo class for interacting with a ParamLake version-controlled repository.
"""
from typing import Any, Dict, List, Optional, Union
import tensorflow as tf # Assuming TensorFlow for now for model type hint
import numpy as np

from paramlake.storage.storage_interface import StorageInterface
from paramlake.storage.factory import create_storage_manager
from paramlake.utils.config import ParamLakeConfig

class Repo:
    """Manages a ParamLake version-controlled model repository."""

    def __init__(
        self,
        path: str,
        run_id: Optional[str] = None,
        storage_type: str = 'icechunk', # Default to icechunk for versioning features
        config: Optional[Union[Dict[str, Any], ParamLakeConfig]] = None,
        **kwargs
    ):
        """
        Initialize or open a ParamLake repository.

        Args:
            path: Path to the repository (local or cloud URI if supported by backend).
            run_id: A specific run ID to operate on. If None, a default or new one might be used.
            storage_type: Backend storage type ('icechunk' or 'zarr'). 
                          Git-like features are primarily supported by 'icechunk'.
            config: A dictionary or ParamLakeConfig object for advanced configuration.
            **kwargs: Additional configuration options for ParamLakeConfig.
        """
        if not HAS_ICECHUNK and storage_type == 'icechunk':
            raise ImportError("Icechunk is required for full version control features but not installed. Install with 'pip install icechunk'.")

        if isinstance(config, ParamLakeConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = ParamLakeConfig(config, **kwargs)
        else:
            self.config = ParamLakeConfig(kwargs)

        # Ensure essential config for Repo operation
        self.config.config['output_path'] = path
        if run_id:
            self.config.config['run_id'] = run_id
        if storage_type:
            self.config.config['storage_type'] = storage_type # Override if passed

        self.storage_manager: StorageInterface = create_storage_manager(self.config)
        self.current_branch: Optional[str] = 'main' # Default branch
        # TODO: Load current branch from a .paramlake/HEAD file or similar

    def _extract_model_parameters(self, model: Any) -> Dict[str, np.ndarray]:
        """Helper to extract parameters from a model object."""
        # This needs to be framework-specific. Placeholder for Keras:
        if isinstance(model, tf.keras.Model):
            params = {}
            for weight_var in model.weights:
                params[weight_var.name] = weight_var.numpy()
            return params
        # TODO: Add support for PyTorch (model.state_dict()), etc.
        raise NotImplementedError("Model parameter extraction not implemented for this model type.")

    def commit(
        self,
        model: Any, 
        message: str, 
        branch: Optional[str] = None, 
        author: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Commits the current state of the model to the repository."""
        if not hasattr(self.storage_manager, 'commit_model_state'):
            raise NotImplementedError("The configured storage manager does not support full commit history.")
        
        params_to_commit = self._extract_model_parameters(model)
        target_branch = branch if branch is not None else self.current_branch
        if target_branch is None:
             target_branch = 'main' # Fallback to main

        snapshot_id = self.storage_manager.commit_model_state(
            model_parameters=params_to_commit,
            message=message,
            branch_name=target_branch,
            author=author,
            additional_metadata=additional_metadata
        )
        # If committing to the current branch, update its HEAD implicitly with the new snapshot_id
        # (IceChunk handles this internally for its branch reference)
        print(f"Committed to branch '{target_branch}' - Snapshot ID: {snapshot_id}")
        return snapshot_id

    def create_branch(self, branch_name: str, from_reference: Optional[str] = None) -> None:
        """Creates a new branch."""
        if not hasattr(self.storage_manager, 'create_branch'):
            raise NotImplementedError("Storage manager does not support branching.")
        self.storage_manager.create_branch(branch_name, from_reference)
        print(f"Branch '{branch_name}' created.")

    def list_branches(self) -> List[str]:
        """Lists all branches."""
        if not hasattr(self.storage_manager, 'list_branches'):
            raise NotImplementedError("Storage manager does not support listing branches.")
        return self.storage_manager.list_branches()

    def delete_branch(self, branch_name: str) -> None:
        """Deletes a branch."""
        if not hasattr(self.storage_manager, 'delete_branch'):
            raise NotImplementedError("Storage manager does not support deleting branches.")
        if branch_name == 'main':
            raise ValueError("Cannot delete the main branch.")
        self.storage_manager.delete_branch(branch_name)
        print(f"Branch '{branch_name}' deleted.")
        if self.current_branch == branch_name:
            self.current_branch = 'main' # Switch to main if current branch was deleted

    def create_tag(self, tag_name: str, reference: str) -> None:
        """Creates a new tag for a given reference (snapshot_id, branch, or another tag)."""
        if not hasattr(self.storage_manager, 'create_tag'):
            raise NotImplementedError("Storage manager does not support tagging.")
        self.storage_manager.create_tag(tag_name, reference)
        print(f"Tag '{tag_name}' created for reference '{reference}'.")

    def list_tags(self) -> Dict[str, str]:
        """Lists all tags and the snapshots they point to."""
        if not hasattr(self.storage_manager, 'list_tags'):
            raise NotImplementedError("Storage manager does not support listing tags.")
        return self.storage_manager.list_tags()

    def delete_tag(self, tag_name: str) -> None:
        """Deletes a tag."""
        if not hasattr(self.storage_manager, 'delete_tag'):
            raise NotImplementedError("Storage manager does not support deleting tags.")
        self.storage_manager.delete_tag(tag_name)
        print(f"Tag '{tag_name}' deleted.")

    def log(self, reference: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Shows the commit history for a reference (branch/tag) or current branch."""
        if not hasattr(self.storage_manager, 'get_history'):
            raise NotImplementedError("Storage manager does not support history/log.")
        target_ref = reference if reference is not None else self.current_branch
        history = self.storage_manager.get_history(target_ref, limit)
        print(f"History for '{target_ref}':")
        for entry in history:
            print(f"  Commit: {entry.get('id')}")
            print(f"    Author: {entry.get('author', 'N/A')}")
            print(f"    Date: {entry.get('timestamp', 'N/A')}")
            print(f"    Message: {entry.get('message', '')}")
            if entry.get('tags'):
                print(f"    Tags: {', '.join(entry.get('tags'))}")
        return history

    def checkout(self, reference: str, target_model_instance: Optional[Any] = None) -> None:
        """Checks out a specific version (branch, tag, or snapshot ID)."""
        if not hasattr(self.storage_manager, 'load_parameters_from_snapshot'):
            raise NotImplementedError("Storage manager does not support loading parameters from arbitrary snapshots.")
        
        # Resolve reference to a snapshot_id (storage manager should handle this)
        # For now, assume `reference` can be directly used or resolved by `load_parameters_from_snapshot`
        parameters = self.storage_manager.load_parameters_from_snapshot(reference)
        
        if target_model_instance:
            if isinstance(target_model_instance, tf.keras.Model):
                # This is a simplified assignment. Real assignment needs care with names/order.
                current_model_weights = target_model_instance.get_weights()
                if len(current_model_weights) == len(parameters):
                    # Assuming parameters is a list of np.arrays in the correct order
                    # Or if parameters is a dict, we need to match by name
                    if isinstance(parameters, dict):
                        # Match by name, requires model weights to have names
                        weights_to_set = []
                        for weight_var in target_model_instance.weights:
                            if weight_var.name in parameters:
                                weights_to_set.append(parameters[weight_var.name])
                            else:
                                # Keep existing weight if not in checkpoint
                                print(f"Warning: Weight {weight_var.name} not found in checkpoint, using existing.")
                                weights_to_set.append(weight_var.numpy())
                        if len(weights_to_set) == len(target_model_instance.weights):
                             target_model_instance.set_weights(weights_to_set)
                        else:
                            print("Error: Mismatch in number of weights found in checkpoint for named assignment.")
                    elif isinstance(parameters, list):
                        target_model_instance.set_weights(parameters) # Assumes direct list of numpy arrays
                else:
                    print(f"Warning: Weight count mismatch. Model has {len(current_model_weights)}, checkpoint has {len(parameters)}. Cannot set weights.")
            else:
                # TODO: Implement for other frameworks
                raise NotImplementedError("Checkout to this model type not implemented.")
            print(f"Loaded parameters from '{reference}' into model.")
        
        # Update current branch if checking out a branch name
        # This is a simplified check; real check involves asking storage_manager if 'reference' is a branch
        # For now, we assume if it doesn't look like a typical snapshot ID, it might be a branch/tag.
        if not (len(reference) > 10 and reference.isalnum()): # Basic heuristic for snapshot ID
            # Potentially a branch or tag. If it's a branch, update current_branch.
            # This needs proper resolution via storage_manager.is_branch(reference)
            # For now, if checking out a branch name explicitly, update self.current_branch
            # (This part of logic needs refinement with storage_manager's help)
            pass 
        print(f"Checked out '{reference}'.")

    def diff(self, ref1: str, ref2: Optional[str] = None):
        """Shows differences between two references (commits, branches, tags)."""
        print(f"Diff between '{ref1}' and '{ref2 if ref2 else 'WORKING_COPY (Not Implemented)'}' (Not Implemented Yet)")
        # TODO: Implement using self.storage_manager.diff_snapshots(ref1_snap_id, ref2_snap_id)
        return []

    def merge(self, source_branch: str, strategy: str = 'manual'):
        """Merges source_branch into the current branch."""
        print(f"Merging branch '{source_branch}' into '{self.current_branch}' (Not Implemented Yet - Strategy: {strategy})")
        # TODO: Implement using self.storage_manager.merge_branches(...)
        return None

    def import_model(self, source_path: str, source_format: str, message: Optional[str] = None):
        """Imports a model from an external format into the repository."""
        print(f"Importing model from '{source_path}' (Format: {source_format}) (Not Implemented Yet)")
        # TODO: Implement using self.storage_manager.import_model_from_path(...)
        return None

    def export_model(self, reference: str, target_path: str, target_format: str):
        """Exports a model version to an external format."""
        print(f"Exporting '{reference}' to '{target_path}' (Format: {target_format}) (Not Implemented Yet)")
        # TODO: Implement using self.storage_manager.export_model_to_path(...)
        return None
    
    def status(self):
        """Shows the current repository status (current branch, uncommitted changes - conceptual)."""
        print(f"On branch: {self.current_branch}")
        print("Uncommitted changes detection: Not Implemented Yet")
        # TODO: Query storage_manager for last commit on current_branch, etc.
        return {}

# Helper to check if Icechunk is available at module level
try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False 