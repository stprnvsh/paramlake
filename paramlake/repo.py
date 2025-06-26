"""
Provides the Repo class for interacting with a ParamLake version-controlled repository.
"""
from typing import Any, Dict, List, Optional, Union, Callable
import functools
import os
import json
from pathlib import Path
import numpy as np

from paramlake.storage.storage_interface import StorageInterface
from paramlake.storage.factory import create_storage_manager
from paramlake.utils.config import ParamLakeConfig
from paramlake.utils.framework_utils import (
    detect_model_framework,
    extract_model_parameters,
    apply_parameters_to_model,
    framework_info,
    safe_import_tensorflow,
    TensorFlowModel,
    TorchModel,
    JAXArray,
    HAS_TENSORFLOW,
    require_tensorflow
)

# Helper to check if Icechunk is available at module level
try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False


class RepositoryError(Exception):
    """Custom exception for repository-related errors."""
    pass


class Repo:
    """Manages a ParamLake version-controlled model repository."""

    @staticmethod
    def is_repository(path: str) -> bool:
        """
        Check if a path contains a valid ParamLake repository.
        
        Args:
            path: Path to check
            
        Returns:
            True if path contains a valid repository
        """
        if not os.path.exists(path):
            return False
            
        # Check for common ParamLake repository indicators
        repo_indicators = [
            # Icechunk repository files
            ".icechunk",
            "icechunk.json",
            # Zarr repository files
            ".zarray",
            ".zgroup",
            "zarr.json",
            # ParamLake metadata
            ".paramlake",
            "paramlake.json"
        ]
        
        for indicator in repo_indicators:
            if os.path.exists(os.path.join(path, indicator)):
                return True
        
        # Check if it's a zarr directory structure
        if path.endswith('.zarr') and os.path.isdir(path):
            return True
            
        return False

    @staticmethod
    def find_repository(start_path: str = ".") -> Optional[str]:
        """
        Find a ParamLake repository by walking up the directory tree.
        
        Args:
            start_path: Directory to start searching from
            
        Returns:
            Path to repository root, or None if not found
        """
        current_path = os.path.abspath(start_path)
        
        while current_path != os.path.dirname(current_path):  # Not at filesystem root
            if Repo.is_repository(current_path):
                return current_path
            
            # Check for .paramlake directory indicating repo root
            paramlake_dir = os.path.join(current_path, ".paramlake")
            if os.path.exists(paramlake_dir):
                config_file = os.path.join(paramlake_dir, "config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            repo_path = config.get('repository_path')
                            if repo_path and Repo.is_repository(repo_path):
                                return repo_path
                    except:
                        pass
                return current_path
            
            current_path = os.path.dirname(current_path)
        
        return None

    @staticmethod
    def get_repository_info(path: str) -> Dict[str, Any]:
        """
        Get information about a repository at the given path.
        
        Args:
            path: Path to the repository
            
        Returns:
            Dictionary with repository information
        """
        if not Repo.is_repository(path):
            raise RepositoryError(f"No ParamLake repository found at: {path}")
        
        info = {
            "path": os.path.abspath(path),
            "exists": True,
            "storage_type": "unknown",
            "has_git_features": False,
            "branches": [],
            "tags": {},
            "current_branch": None,
            "last_commit": None,
            "repository_size": 0
        }
        
        try:
            # Try to determine storage type and get basic info
            if os.path.exists(os.path.join(path, ".icechunk")) or any(f.startswith("icechunk") for f in os.listdir(path)):
                info["storage_type"] = "icechunk"
                info["has_git_features"] = True
            elif any(f.endswith('.zarr') for f in os.listdir(path)) or path.endswith('.zarr'):
                info["storage_type"] = "zarr"
            
            # Calculate repository size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
            info["repository_size"] = total_size
            
            # Try to load more detailed info if possible
            temp_config = ParamLakeConfig({"output_path": path, "storage_type": info["storage_type"]})
            try:
                storage_manager = create_storage_manager(temp_config)
                
                # Get branches
                if hasattr(storage_manager, 'list_branches'):
                    info["branches"] = storage_manager.list_branches()
                
                # Get tags
                if hasattr(storage_manager, 'list_tags'):
                    info["tags"] = storage_manager.list_tags()
                
                # Get current branch
                if hasattr(storage_manager, 'session') and hasattr(storage_manager.session, 'branch'):
                    info["current_branch"] = storage_manager.session.branch
                
                # Get recent history
                if hasattr(storage_manager, 'get_history'):
                    try:
                        history = storage_manager.get_history(limit=1)
                        if history:
                            info["last_commit"] = history[0]
                    except:
                        pass
                        
            except Exception as e:
                info["error"] = f"Could not load detailed repository info: {e}"
        
        except Exception as e:
            info["error"] = f"Error inspecting repository: {e}"
        
        return info

    @classmethod
    def connect(
        cls,
        path: Optional[str] = None,
        auto_find: bool = True,
        run_id: Optional[str] = None,
        config: Optional[Union[Dict[str, Any], ParamLakeConfig]] = None,
        **kwargs
    ) -> 'Repo':
        """
        Connect to an existing ParamLake repository.
        
        Args:
            path: Path to the repository (if None, searches current directory)
            auto_find: Whether to automatically search for repository in parent directories
            run_id: Specific run ID to operate on
            config: Configuration object or dictionary
            **kwargs: Additional configuration options
            
        Returns:
            Repo instance connected to the existing repository
            
        Raises:
            RepositoryError: If no repository is found or connection fails
        """
        # Find repository path
        if path is None:
            if auto_find:
                path = cls.find_repository()
                if path is None:
                    raise RepositoryError("No ParamLake repository found in current directory or parent directories")
            else:
                path = "."
        
        # Validate repository exists
        if not cls.is_repository(path):
            raise RepositoryError(f"No ParamLake repository found at: {path}")
        
        # Get repository info to determine connection parameters
        repo_info = cls.get_repository_info(path)
        
        # Create configuration for connection
        connect_config = {
            "output_path": path,
            "storage_type": repo_info.get("storage_type", "icechunk"),
            "create_repo": False,  # Important: don't create, just connect
            "connect_existing": True  # Flag to indicate this is a connection
        }
        
        # Merge with provided config
        if isinstance(config, dict):
            connect_config.update(config)
        elif isinstance(config, ParamLakeConfig):
            connect_config.update(config.to_dict())
        
        connect_config.update(kwargs)
        
        # Create repo instance
        try:
            repo = cls(path=path, run_id=run_id, config=connect_config)
            
            # Set current branch from repository if available
            if repo_info.get("current_branch"):
                repo.current_branch = repo_info["current_branch"]
            
            print(f"✓ Connected to ParamLake repository at: {path}")
            print(f"  Storage type: {repo_info.get('storage_type', 'unknown')}")
            print(f"  Current branch: {repo.current_branch}")
            if repo_info.get("branches"):
                print(f"  Available branches: {', '.join(repo_info['branches'])}")
            
            return repo
            
        except Exception as e:
            raise RepositoryError(f"Failed to connect to repository: {e}")

    @classmethod
    def init_or_connect(
        cls,
        path: str,
        auto_connect: bool = True,
        **kwargs
    ) -> 'Repo':
        """
        Initialize a new repository or connect to existing one.
        
        Args:
            path: Path for the repository
            auto_connect: Whether to connect if repository already exists
            **kwargs: Configuration options for initialization
            
        Returns:
            Repo instance
        """
        if cls.is_repository(path):
            if auto_connect:
                print(f"Repository already exists at {path}, connecting...")
                return cls.connect(path, auto_find=False, **kwargs)
            else:
                raise RepositoryError(f"Repository already exists at: {path}")
        else:
            print(f"Initializing new repository at {path}")
            init_config = kwargs.copy()
            init_config["create_repo"] = True
            return cls(path=path, config=init_config)

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

        # Check if connecting to existing repository
        self.is_existing_repo = self.config.get('connect_existing', False) or self.is_repository(path)
        
        if self.is_existing_repo and not self.config.get('create_repo', False):
            print(f"Opening existing repository at: {path}")

        self.storage_manager: StorageInterface = create_storage_manager(self.config)
        self.current_branch: Optional[str] = 'main' # Default branch
        
        # Try to get actual current branch from storage manager if available
        try:
            if hasattr(self.storage_manager, 'session') and hasattr(self.storage_manager.session, 'branch'):
                self.current_branch = self.storage_manager.session.branch
        except:
            pass

        # Validate connection for existing repositories
        if self.is_existing_repo:
            self._validate_repository_connection()

    def _validate_repository_connection(self) -> None:
        """Validate that we've successfully connected to the repository."""
        try:
            # Test basic operations to ensure connection is working
            if hasattr(self.storage_manager, 'list_branches'):
                branches = self.storage_manager.list_branches()
                if branches and self.current_branch not in branches:
                    # Set to first available branch if current is not valid
                    self.current_branch = branches[0]
            
            # Test that we can read repository status
            self.status()
            
        except Exception as e:
            print(f"Warning: Repository connection validation failed: {e}")

    def track(
        self,
        capture_frequency: int = 1,
        gradients: Union[bool, Dict[str, Any]] = True,
        optimizer_state: bool = True,
        metrics: Union[bool, List[str]] = True,
        activations: bool = False,
        **kwargs
    ) -> Callable:
        """
        Decorator for tracking model training with ParamLake.
        
        Args:
            capture_frequency: How often to capture data (every N epochs)
            gradients: Whether to capture gradients (True/False or dict with options)
            optimizer_state: Whether to capture optimizer state
            metrics: Whether to capture metrics (True/False or list of metric names)
            activations: Whether to capture activations
            **kwargs: Additional configuration options
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                # Import here to avoid circular imports
                from paramlake.decorators.model_decorator import ParamLakeCallback
                
                # Create callback using the existing storage manager
                callback = ParamLakeCallback(
                    storage_manager=self.storage_manager,
                    config=self.config,
                    collect_weights=True,  # Always collect weights for repo tracking
                    collect_gradients=isinstance(gradients, bool) and gradients or isinstance(gradients, dict),
                    collect_activations=activations,
                    collect_optimizer=optimizer_state,
                    track_layers=None,  # Track all layers by default
                    ignore_layers=None,
                )
                
                # Override Model.fit to include our callback (only if TensorFlow is available)
                if HAS_TENSORFLOW:
                    tf = require_tensorflow()
                    original_fit = tf.keras.Model.fit
                    
                    def patched_fit(self_model, *fit_args, **fit_kwargs):
                        # Add our callback to the callbacks list
                        callbacks = fit_kwargs.get('callbacks', [])
                        if callbacks is None:
                            callbacks = []
                        elif not isinstance(callbacks, list):
                            callbacks = [callbacks]
                        
                        # Add our callback if not already there
                        if callback not in callbacks:
                            callbacks.append(callback)
                        
                        fit_kwargs['callbacks'] = callbacks
                        return original_fit(self_model, *fit_args, **fit_kwargs)
                    
                    # Replace the fit method temporarily
                    tf.keras.Model.fit = patched_fit
                
                try:
                    # Call the decorated function
                    result = func(*args, **func_kwargs)
                    
                    # If the function returned a model, capture initial state
                    if HAS_TENSORFLOW:
                        tf = require_tensorflow()
                        if isinstance(result, tf.keras.Model):
                            if callback.weight_collector is not None:
                                callback.weight_collector.capture_model_weights(result, step=0)
                    
                    return result
                
                finally:
                    # Always restore original fit method (only if TensorFlow is available)
                    if HAS_TENSORFLOW:
                        tf = require_tensorflow()
                        tf.keras.Model.fit = original_fit
                    
                    # Ensure final commit if needed
                    try:
                        if hasattr(self.storage_manager, 'commit_if_needed'):
                            self.storage_manager.commit_if_needed(
                                step=getattr(self.storage_manager, 'current_step', 0),
                                force=True,
                                message="Final commit from repo.track()"
                            )
                    except Exception as e:
                        print(f"Warning: Error in final commit: {e}")
                        
            return wrapper
        return decorator

    def _extract_model_parameters(self, model: Any) -> Dict[str, np.ndarray]:
        """
        Extract model parameters in a standardized format using framework-agnostic utilities.
        
        Args:
            model: The model to extract parameters from
            
        Returns:
            Dict mapping parameter names to numpy arrays
        """
        try:
            return extract_model_parameters(model)
        except Exception as e:
            raise ValueError(f"Failed to extract model parameters: {e}")

    def _resolve_reference_to_snapshot(self, reference: str) -> str:
        """Resolve a reference (branch, tag, or snapshot ID) to a snapshot ID."""
        if hasattr(self.storage_manager, 'get_snapshot_id_for_reference'):
            snapshot_id = self.storage_manager.get_snapshot_id_for_reference(reference)
            if snapshot_id:
                return snapshot_id
        # If resolution fails, assume it's already a snapshot ID
        return reference

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

    def switch_branch(self, branch_name: str, create_if_missing: bool = False) -> Dict[str, Any]:
        """Switch to a different branch."""
        if not hasattr(self.storage_manager, 'switch_branch'):
            raise NotImplementedError("Storage manager does not support switching branches.")
        
        result = self.storage_manager.switch_branch(branch_name, create_if_missing)
        if result.get('status') == 'success':
            self.current_branch = branch_name
        return result

    def create_tag(self, tag_name: str, reference: str) -> None:
        """Creates a new tag for a given reference (snapshot_id, branch, or another tag)."""
        if not hasattr(self.storage_manager, 'create_tag'):
            raise NotImplementedError("Storage manager does not support tagging.")
        
        # Resolve reference to snapshot ID
        snapshot_id = self._resolve_reference_to_snapshot(reference)
        self.storage_manager.create_tag(tag_name, snapshot_id)
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
        
        # Resolve reference to snapshot ID
        snapshot_id = self._resolve_reference_to_snapshot(reference)
        parameters = self.storage_manager.load_parameters_from_snapshot(snapshot_id)
        
        if target_model_instance:
            try:
                apply_parameters_to_model(target_model_instance, parameters)
                print(f"✓ Successfully loaded parameters from '{reference}' into model.")
            except Exception as e:
                print(f"Error applying parameters to model: {e}")
                raise
        
        # Update current branch if checking out a branch
        try:
            branches = self.list_branches()
            if reference in branches:
                self.current_branch = reference
        except:
            pass
            
        print(f"Checked out '{reference}'.")

    def diff(self, ref1: str, ref2: Optional[str] = None) -> Dict[str, Any]:
        """Shows differences between two references (commits, branches, tags)."""
        if not hasattr(self.storage_manager, 'diff_snapshots'):
            raise NotImplementedError("Storage manager does not support diff operations.")
        
        # Resolve references to snapshot IDs
        snapshot1 = self._resolve_reference_to_snapshot(ref1)
        
        if ref2 is None:
            # Compare against current working state (use current branch HEAD)
            snapshot2 = self._resolve_reference_to_snapshot(self.current_branch or 'main')
        else:
            snapshot2 = self._resolve_reference_to_snapshot(ref2)
        
        return self.storage_manager.diff_snapshots(snapshot1, snapshot2)

    def merge(self, source_branch: str, strategy: str = 'auto', commit_message: Optional[str] = None) -> Optional[str]:
        """Merges source_branch into the current branch."""
        if not hasattr(self.storage_manager, 'merge_branches'):
            raise NotImplementedError("Storage manager does not support merge operations.")
        
        target_branch = self.current_branch or 'main'
        return self.storage_manager.merge_branches(
            source_branch, 
            target_branch, 
            strategy=strategy,
            commit_message=commit_message
        )

    def rebase(self, branch_name: str, onto_branch: Optional[str] = None, conflict_strategy: str = 'detect') -> str:
        """Rebase a branch onto another branch."""
        if not hasattr(self.storage_manager, 'rebase_branch'):
            raise NotImplementedError("Storage manager does not support rebase operations.")
        
        target_onto = onto_branch or self.current_branch or 'main'
        return self.storage_manager.rebase_branch(branch_name, target_onto, conflict_strategy)

    def reset(self, to_reference: str, branch_name: Optional[str] = None) -> None:
        """Reset a branch to a specific reference."""
        if not hasattr(self.storage_manager, 'reset_branch'):
            raise NotImplementedError("Storage manager does not support reset operations.")
        
        target_branch = branch_name or self.current_branch or 'main'
        self.storage_manager.reset_branch(target_branch, to_reference)

    def push(self, remote_config: Dict[str, Any], branch: Optional[str] = None) -> Dict[str, Any]:
        """Push changes to a remote repository."""
        if not hasattr(self.storage_manager, 'push_to_remote'):
            raise NotImplementedError("Storage manager does not support push operations.")
        
        target_branch = branch or self.current_branch or 'main'
        return self.storage_manager.push_to_remote(remote_config, target_branch)

    def pull(self, remote_config: Dict[str, Any], branch: Optional[str] = None) -> Dict[str, Any]:
        """Pull changes from a remote repository."""
        if not hasattr(self.storage_manager, 'pull_from_remote'):
            raise NotImplementedError("Storage manager does not support pull operations.")
        
        target_branch = branch or self.current_branch or 'main'
        return self.storage_manager.pull_from_remote(remote_config, target_branch)

    def get_conflicts(self, branch1: str, branch2: str) -> List[Dict[str, Any]]:
        """Get conflicts between two branches."""
        if not hasattr(self.storage_manager, 'get_conflicts'):
            return []
        return self.storage_manager.get_conflicts(branch1, branch2)

    def import_model(self, source_path: str, source_format: str, message: Optional[str] = None, branch: Optional[str] = None) -> str:
        """Imports a model from an external format into the repository."""
        if not hasattr(self.storage_manager, 'import_model_from_path'):
            raise NotImplementedError("Storage manager does not support model import.")
        
        target_branch = branch or self.current_branch or 'main'
        return self.storage_manager.import_model_from_path(source_path, source_format, target_branch, message)

    def status(self) -> Dict[str, Any]:
        """Shows the current repository status (current branch, uncommitted changes - conceptual)."""
        status_info = {
            "current_branch": self.current_branch,
            "storage_type": self.config.get('storage_type', 'unknown'),
            "has_git_features": hasattr(self.storage_manager, 'commit_model_state')
        }
        
        print(f"On branch: {self.current_branch}")
        
        # Try to get additional status from storage manager
        if hasattr(self.storage_manager, 'get_storage_info'):
            storage_info = self.storage_manager.get_storage_info()
            status_info.update(storage_info)
        
        # Get recent commits
        try:
            recent_commits = self.log(limit=3)
            status_info["recent_commits"] = len(recent_commits)
        except:
            status_info["recent_commits"] = 0
        
        return status_info

    def checkout_snapshot(self, reference: str, new_branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Checkout to a specific snapshot to continue training from that point.
        
        Args:
            reference: Snapshot ID, branch name, or tag name
            new_branch: If provided, create a new branch from this snapshot
            
        Returns:
            Dictionary with checkout status and new branch info
        """
        if not hasattr(self.storage_manager, 'checkout_snapshot'):
            raise NotImplementedError("Storage manager does not support snapshot checkout.")
        
        result = self.storage_manager.checkout_snapshot(reference, new_branch)
        if result.get('status') == 'success':
            if new_branch:
                self.current_branch = new_branch
            else:
                # Update to detached branch name
                self.current_branch = result.get('branch')
        return result

    def commit_with_rebase(
        self,
        model: Any, 
        message: str, 
        conflict_strategy: str = 'detect'
    ) -> str:
        """
        Commit model state with automatic rebasing for collaborative workflows.
        
        Args:
            model: Model to commit
            message: Commit message
            conflict_strategy: How to handle conflicts ('detect', 'ours', 'theirs')
            
        Returns:
            Snapshot ID of the commit
        """
        if not hasattr(self.storage_manager, 'commit_with_rebase'):
            # Fallback to regular commit + rebase
            snapshot_id = self.commit(model, message)
            return snapshot_id
        
        # First, store the model parameters in the current session
        params_to_commit = self._extract_model_parameters(model)
        
        # Store parameters in current session (this doesn't commit yet)
        # We need to implement a way to store parameters without committing
        # For now, use the regular commit path but with rebase
        return self.storage_manager.commit_with_rebase(message, conflict_strategy)

    def get_analyzer(self, snapshot_id: Optional[str] = None, branch: Optional[str] = None) -> 'IcechunkModelAnalyzer':
        """
        Get an analyzer instance for detailed model analysis.
        
        Args:
            snapshot_id: Specific snapshot to analyze (if None, uses current)
            branch: Branch to analyze (if None, uses current branch)
            
        Returns:
            IcechunkModelAnalyzer instance
        """
        # Import here to avoid circular imports
        from paramlake.storage.icechunk_analyzer import IcechunkModelAnalyzer
        
        if not hasattr(self.storage_manager, 'repo'):
            raise NotImplementedError("Analysis features require Icechunk storage manager.")
        
        target_snapshot = None
        if snapshot_id:
            target_snapshot = snapshot_id
        elif branch:
            target_snapshot = self._resolve_reference_to_snapshot(branch)
        # If neither provided, analyzer will use current HEAD
        
        return IcechunkModelAnalyzer(
            self.storage_manager.repo,
            snapshot_id=target_snapshot,
            lazy_loading=True
        )

    def diff_visual(
        self, 
        ref1: str, 
        ref2: Optional[str] = None, 
        output_format: str = "console",
        include_values: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Create a visual diff between two model versions.
        
        Args:
            ref1: First reference to compare
            ref2: Second reference (defaults to current HEAD)
            output_format: Output format ('console', 'html', 'dict')
            include_values: Whether to include actual tensor values
            
        Returns:
            Formatted diff output
        """
        analyzer = self.get_analyzer()
        return analyzer.diff_snapshots_visual(
            ref1, 
            ref2 or self.current_branch or 'main',
            output_format=output_format,
            include_values=include_values
        )

    def compare_models(
        self,
        ref1: str,
        ref2: str,
        layer_name: str,
        tensor_type: str = "weights",
        stat: str = "norm"
    ) -> Dict[str, Any]:
        """
        Compare statistical properties of models between two references.
        
        Args:
            ref1: First reference
            ref2: Second reference  
            layer_name: Name of layer to compare
            tensor_type: Type of tensor to compare
            stat: Statistic to compare
            
        Returns:
            Comparison results
        """
        analyzer = self.get_analyzer()
        snapshot1 = self._resolve_reference_to_snapshot(ref1)
        return analyzer.compare_snapshots(snapshot1, layer_name, tensor_type, stat=stat)

    def analyze_gradients(self, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze gradient statistics for a specific model version.
        
        Args:
            reference: Reference to analyze (defaults to current)
            
        Returns:
            Gradient analysis results
        """
        target_ref = reference or self.current_branch
        analyzer = self.get_analyzer()
        if reference and reference != self.current_branch:
            analyzer = analyzer.checkout_analyzer(reference)
        
        return analyzer.analyze_gradient_statistics()

    def plot_training_evolution(
        self,
        layer_name: str,
        tensor_name: Optional[str] = None,
        tensor_type: str = "weights",
        stat: str = "norm",
        reference: Optional[str] = None
    ) -> None:
        """
        Plot the evolution of model parameters over training.
        
        Args:
            layer_name: Name of the layer
            tensor_name: Name of the tensor (or None for all)
            tensor_type: Type of tensor
            stat: Statistic to plot
            reference: Reference to analyze
        """
        analyzer = self.get_analyzer()
        if reference and reference != self.current_branch:
            analyzer = analyzer.checkout_analyzer(reference)
        
        analyzer.plot_weight_evolution(
            layer_name=layer_name,
            tensor_name=tensor_name,
            tensor_type=tensor_type,
            stat=stat
        )

    def get_model_summary(self, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model at a specific reference.
        
        Args:
            reference: Reference to analyze (defaults to current)
            
        Returns:
            Model summary including architecture, parameters, and metadata
        """
        analyzer = self.get_analyzer()
        if reference and reference != self.current_branch:
            analyzer = analyzer.checkout_analyzer(reference)
        
        return {
            "metadata": analyzer.get_run_metadata(),
            "layer_names": analyzer.get_layer_names(),
            "training_history": analyzer.get_training_history(),
        }

    def export_training_report(
        self,
        output_path: str,
        reference: Optional[str] = None,
        format: str = "html"
    ) -> None:
        """
        Export a comprehensive training report.
        
        Args:
            output_path: Path to save the report
            reference: Reference to analyze
            format: Report format ('html', 'json')
        """
        analyzer = self.get_analyzer()
        if reference and reference != self.current_branch:
            analyzer = analyzer.checkout_analyzer(reference)
        
        # Generate comprehensive report
        report_data = {
            "summary": self.get_model_summary(reference),
            "gradient_analysis": analyzer.analyze_gradient_statistics(),
            "layer_info": {name: analyzer.get_layer_info(name) for name in analyzer.get_layer_names()},
        }
        
        if format == "html":
            # Generate HTML report
            html_content = self._generate_html_report(report_data)
            with open(output_path, 'w') as f:
                f.write(html_content)
        elif format == "json":
            import json
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from report data."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ParamLake Training Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .stat { margin: 10px 0; }
                .error { color: red; }
                .success { color: green; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>ParamLake Training Report</h1>
        """
        
        # Add summary section
        summary = report_data.get('summary', {})
        metadata = summary.get('metadata', {})
        
        html += f"""
            <div class="section">
                <h2>Model Summary</h2>
                <div class="stat">Framework: {metadata.get('framework', 'Unknown')}</div>
                <div class="stat">Version: {metadata.get('paramlake_version', 'Unknown')}</div>
                <div class="stat">Timestamp: {metadata.get('timestamp', 'Unknown')}</div>
                <div class="stat">Layers: {len(summary.get('layer_names', []))}</div>
            </div>
        """
        
        # Add gradient analysis
        grad_analysis = report_data.get('gradient_analysis', {})
        grad_summary = grad_analysis.get('summary', {})
        
        html += f"""
            <div class="section">
                <h2>Gradient Analysis</h2>
                <div class="stat">Total Layers: {grad_summary.get('total_layers', 0)}</div>
                <div class="stat">Layers with Gradients: {grad_summary.get('layers_with_gradients', 0)}</div>
                <div class="stat">Gradient Coverage: {grad_summary.get('gradient_coverage', 0):.2%}</div>
                <div class="stat">Total Gradient Tensors: {grad_summary.get('total_gradient_tensors', 0)}</div>
            </div>
        """
        
        html += """
            </body>
        </html>
        """
        
        return html 

    def _extract_layer_parameters(self, model: Any, layer_names: List[str]) -> Dict[str, np.ndarray]:
        """Extract parameters from specific layers only."""
        if HAS_TENSORFLOW:
            tf = require_tensorflow()
            if isinstance(model, tf.keras.Model):
                params = {}
                
                # Create a mapping of weight objects to their layer information
                weight_to_layer = {}
                for layer in model.layers:
                    if layer.name in layer_names:  # Only process specified layers
                        for weight in layer.weights:
                            weight_to_layer[id(weight)] = layer.name
                
                for i, weight_var in enumerate(model.weights):
                    # Check if this weight belongs to one of the target layers
                    layer_name = weight_to_layer.get(id(weight_var))
                    if layer_name is not None:  # Only include weights from target layers
                        weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                        
                        # Combine layer name and weight name for uniqueness
                        unique_name = f"{layer_name}/{weight_name}"
                        
                        # Debug logging if verbose mode is enabled
                        if self.config.get("verbose", False):
                            print(f"Extracting layer parameter: '{unique_name}' -> shape: {weight_var.shape}")
                        
                        params[unique_name] = weight_var.numpy()
                return params
        # TODO: Add support for PyTorch (model.state_dict()), etc.
        raise NotImplementedError("Layer parameter extraction not implemented for this model type.")

    def _get_model_layer_names(self, model: Any) -> List[str]:
        """Get all layer names from a model."""
        if HAS_TENSORFLOW:
            tf = require_tensorflow()
            if isinstance(model, tf.keras.Model):
                return [layer.name for layer in model.layers]
        raise NotImplementedError("Layer name extraction not implemented for this model type.")

    def commit_layer(
        self,
        model: Any,
        layer_names: Union[str, List[str]],
        message: str,
        branch: Optional[str] = None,
        author: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Commit only specific layers to the repository.
        
        Args:
            model: Model containing the layers to commit
            layer_names: Name(s) of layers to commit (string or list)
            message: Commit message
            branch: Target branch (defaults to current)
            author: Author name
            additional_metadata: Additional metadata
            
        Returns:
            Snapshot ID
        """
        if not hasattr(self.storage_manager, 'commit_model_state'):
            raise NotImplementedError("The configured storage manager does not support layer commits.")
        
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        # Validate that all specified layers exist
        available_layers = self._get_model_layer_names(model)
        missing_layers = [name for name in layer_names if name not in available_layers]
        if missing_layers:
            raise ValueError(f"Layers not found in model: {missing_layers}")
        
        # Extract only the specified layers' parameters
        layer_params = self._extract_layer_parameters(model, layer_names)
        
        target_branch = branch if branch is not None else self.current_branch
        if target_branch is None:
            target_branch = 'main'

        # Add layer information to metadata
        if additional_metadata is None:
            additional_metadata = {}
        additional_metadata.update({
            'commit_type': 'layer_commit',
            'committed_layers': layer_names,
            'layer_count': len(layer_names)
        })

        snapshot_id = self.storage_manager.commit_model_state(
            model_parameters=layer_params,
            message=message,
            branch_name=target_branch,
            author=author,
            additional_metadata=additional_metadata
        )
        
        layers_str = ', '.join(layer_names)
        print(f"Committed layers [{layers_str}] to branch '{target_branch}' - Snapshot ID: {snapshot_id}")
        return snapshot_id

    def checkout_layer(
        self,
        reference: str,
        layer_names: Union[str, List[str]],
        target_model_instance: Any,
        strategy: str = 'replace'
    ) -> None:
        """
        Checkout specific layers from a reference and apply them to a model.
        
        Args:
            reference: Snapshot ID, branch, or tag to checkout from
            layer_names: Name(s) of layers to checkout (string or list)
            target_model_instance: Model to apply the layers to
            strategy: How to handle the checkout ('replace', 'merge')
        """
        if not hasattr(self.storage_manager, 'load_parameters_from_snapshot'):
            raise NotImplementedError("Storage manager does not support loading parameters from snapshots.")
        
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        # Resolve reference to snapshot ID
        snapshot_id = self._resolve_reference_to_snapshot(reference)
        all_parameters = self.storage_manager.load_parameters_from_snapshot(snapshot_id)
        
        # Filter parameters to only include the specified layers
        layer_parameters = {}
        for param_name, param_data in all_parameters.items():
            # Check if this parameter belongs to one of the target layers
            for layer_name in layer_names:
                if param_name.startswith(f"{layer_name}/"):
                    layer_parameters[param_name] = param_data
                    break
        
        if not layer_parameters:
            print(f"Warning: No parameters found for layers {layer_names} in snapshot {snapshot_id}")
            return
        
        # Apply the layer parameters to the model
        if HAS_TENSORFLOW:
            tf = require_tensorflow()
            if isinstance(target_model_instance, tf.keras.Model):
                self._apply_layer_parameters_to_model(
                    target_model_instance, 
                    layer_parameters, 
                    layer_names,
                    strategy
                )
            else:
                raise NotImplementedError("Layer checkout not implemented for this model type.")
        else:
            raise NotImplementedError("Layer checkout requires TensorFlow. Install with: pip install 'paramlake[tf]'")
        
        layers_str = ', '.join(layer_names)
        print(f"✓ Checked out layers [{layers_str}] from '{reference}' using strategy '{strategy}'")

    def _apply_layer_parameters_to_model(
        self,
        model: Any,
        layer_parameters: Dict[str, np.ndarray],
        target_layer_names: List[str],
        strategy: str
    ) -> None:
        """Apply layer parameters to specific layers in a model."""
        if strategy not in ['replace', 'merge']:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'replace' or 'merge'.")
        
        # Create mapping from weight objects to their layer information
        weight_to_layer = {}
        layer_weights = {}  # Track weights by layer
        
        for layer in model.layers:
            if layer.name in target_layer_names:
                layer_weights[layer.name] = []
                for weight in layer.weights:
                    weight_to_layer[id(weight)] = layer.name
                    layer_weights[layer.name].append(weight)
        
        # Build new weights list
        new_weights = []
        weights_updated = 0
        
        for i, weight_var in enumerate(model.weights):
            layer_name = weight_to_layer.get(id(weight_var))
            
            if layer_name in target_layer_names:
                # This weight belongs to a target layer
                weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                unique_name = f"{layer_name}/{weight_name}"
                sanitized_name = unique_name.replace("/", "_").replace(":", "_")
                
                # Try to find the parameter
                param_data = None
                for key_to_try in [unique_name, sanitized_name, weight_var.name]:
                    if key_to_try in layer_parameters:
                        param_data = layer_parameters[key_to_try]
                        break
                
                if param_data is not None and param_data.shape == weight_var.shape:
                    new_weights.append(param_data)
                    weights_updated += 1
                    if self.config.get("verbose", False):
                        print(f"✓ Updated {unique_name} with shape {param_data.shape}")
                else:
                    if strategy == 'replace':
                        print(f"Warning: Could not find matching parameter for {unique_name}, keeping existing")
                    new_weights.append(weight_var.numpy())
            else:
                # This weight is not in a target layer, keep existing
                new_weights.append(weight_var.numpy())
        
        # Apply the new weights
        try:
            model.set_weights(new_weights)
            print(f"✓ Successfully updated {weights_updated} weights in {len(target_layer_names)} layers")
        except ValueError as e:
            print(f"Error applying layer parameters: {e}")
            raise

    def list_layers(self, reference: Optional[str] = None) -> List[str]:
        """
        List all layers available in a specific reference.
        
        Args:
            reference: Snapshot ID, branch, or tag (defaults to current HEAD)
            
        Returns:
            List of layer names
        """
        if reference is None:
            reference = self.current_branch or 'main'
        
        try:
            # Use the analyzer to get layer information
            analyzer = self.get_analyzer()
            if reference != self.current_branch:
                analyzer = analyzer.checkout_analyzer(reference)
            
            return analyzer.get_layer_names()
        except:
            # Fallback: try to extract from parameters
            snapshot_id = self._resolve_reference_to_snapshot(reference)
            parameters = self.storage_manager.load_parameters_from_snapshot(snapshot_id)
            
            # Extract unique layer names from parameter keys
            layer_names = set()
            for param_name in parameters.keys():
                if '/' in param_name:
                    layer_name = param_name.split('/')[0]
                    layer_names.add(layer_name)
            
            return sorted(list(layer_names))

    def diff_layer(
        self,
        layer_names: Union[str, List[str]],
        ref1: str,
        ref2: Optional[str] = None,
        include_values: bool = False
    ) -> Dict[str, Any]:
        """
        Show differences for specific layers between two references.
        
        Args:
            layer_names: Name(s) of layers to compare (string or list)
            ref1: First reference to compare
            ref2: Second reference (defaults to current HEAD)
            include_values: Whether to include actual tensor values in comparison
            
        Returns:
            Dictionary with layer-specific differences
        """
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        if ref2 is None:
            ref2 = self.current_branch or 'main'
        
        # Get analyzers for both references
        analyzer1 = self.get_analyzer()
        analyzer1 = analyzer1.checkout_analyzer(ref1)
        
        analyzer2 = self.get_analyzer()
        if ref2 != ref1:
            analyzer2 = analyzer2.checkout_analyzer(ref2)
        
        layer_diff = {
            "ref1": ref1,
            "ref2": ref2,
            "layers": {},
            "summary": {
                "layers_compared": len(layer_names),
                "layers_changed": 0,
                "layers_missing": []
            }
        }
        
        for layer_name in layer_names:
            try:
                # Get layer info from both snapshots
                info1 = analyzer1.get_layer_info(layer_name)
                info2 = analyzer2.get_layer_info(layer_name)
                
                # Compare the layer
                layer_comparison = self._compare_layer_info(info1, info2, layer_name, include_values, analyzer1, analyzer2)
                
                if layer_comparison.get('has_changes', False):
                    layer_diff["summary"]["layers_changed"] += 1
                
                layer_diff["layers"][layer_name] = layer_comparison
                
            except Exception as e:
                layer_diff["layers"][layer_name] = {"error": str(e)}
                layer_diff["summary"]["layers_missing"].append(layer_name)
        
        return layer_diff

    def _compare_layer_info(
        self,
        info1: Dict[str, Any],
        info2: Dict[str, Any],
        layer_name: str,
        include_values: bool,
        analyzer1,
        analyzer2
    ) -> Dict[str, Any]:
        """Compare layer information between two snapshots."""
        comparison = {
            "has_changes": False,
            "attribute_changes": {},
            "tensor_changes": {}
        }
        
        # Compare attributes
        attrs1 = {k: v for k, v in info1.items() if k not in ['tensor_types', 'tensors']}
        attrs2 = {k: v for k, v in info2.items() if k not in ['tensor_types', 'tensors']}
        
        for key in set(attrs1.keys()) | set(attrs2.keys()):
            if attrs1.get(key) != attrs2.get(key):
                comparison["attribute_changes"][key] = {
                    "old": attrs1.get(key),
                    "new": attrs2.get(key)
                }
                comparison["has_changes"] = True
        
        # Compare tensors
        tensors1 = info1.get('tensors', {})
        tensors2 = info2.get('tensors', {})
        
        for tensor_type in set(tensors1.keys()) | set(tensors2.keys()):
            type_diff = {}
            
            t1_names = set(tensors1.get(tensor_type, []))
            t2_names = set(tensors2.get(tensor_type, []))
            
            # Check for added/removed tensors
            for name in t1_names - t2_names:
                type_diff[name] = {"status": "removed"}
                comparison["has_changes"] = True
                
            for name in t2_names - t1_names:
                type_diff[name] = {"status": "added"}
                comparison["has_changes"] = True
            
            # Check for modified tensors
            for name in t1_names & t2_names:
                if include_values:
                    try:
                        # Compare actual tensor data
                        data1 = analyzer1.get_tensor_data(layer_name, tensor_type, name)
                        data2 = analyzer2.get_tensor_data(layer_name, tensor_type, name)
                        
                        if data1.shape != data2.shape:
                            type_diff[name] = {
                                "shape_changed": True,
                                "old_shape": data1.shape,
                                "new_shape": data2.shape
                            }
                            comparison["has_changes"] = True
                        elif not np.array_equal(data1, data2):
                            # Calculate difference statistics
                            diff = np.abs(data2 - data1)
                            type_diff[name] = {
                                "values_changed": True,
                                "max_diff": float(np.max(diff)),
                                "mean_diff": float(np.mean(diff)),
                                "norm_diff": float(np.linalg.norm(diff))
                            }
                            comparison["has_changes"] = True
                    except Exception as e:
                        type_diff[name] = {"error": f"Could not compare values: {e}"}
            
            if type_diff:
                comparison["tensor_changes"][tensor_type] = type_diff
        
        return comparison

    def merge_layer(
        self,
        layer_names: Union[str, List[str]],
        source_branch: str,
        strategy: str = 'auto',
        commit_message: Optional[str] = None
    ) -> Optional[str]:
        """
        Merge specific layers from source branch into current branch.
        
        Args:
            layer_names: Name(s) of layers to merge (string or list)
            source_branch: Branch to merge layers from
            strategy: Merge strategy ('auto', 'ours', 'theirs')
            commit_message: Custom commit message
            
        Returns:
            Snapshot ID if successful, None otherwise
        """
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        target_branch = self.current_branch or 'main'
        
        if source_branch == target_branch:
            print("Cannot merge layer from the same branch")
            return None
        
        print(f"Merging layers [{', '.join(layer_names)}] from '{source_branch}' into '{target_branch}'")
        
        try:
            # Get current model state (this would typically come from the calling context)
            # For now, we'll work with the storage manager directly
            
            # Get layer parameters from source branch
            source_snapshot = self._resolve_reference_to_snapshot(source_branch)
            source_params = self.storage_manager.load_parameters_from_snapshot(source_snapshot)
            
            # Filter to only the specified layers
            layer_params = {}
            for param_name, param_data in source_params.items():
                for layer_name in layer_names:
                    if param_name.startswith(f"{layer_name}/"):
                        layer_params[param_name] = param_data
                        break
            
            if not layer_params:
                print(f"No parameters found for layers {layer_names} in source branch")
                return None
            
            # Create commit message
            if not commit_message:
                layers_str = ', '.join(layer_names)
                commit_message = f"Merge layers [{layers_str}] from branch '{source_branch}'"
            
            # Commit the layer parameters
            snapshot_id = self.storage_manager.commit_model_state(
                model_parameters=layer_params,
                message=commit_message,
                branch_name=target_branch,
                additional_metadata={
                    'merge_type': 'layer_merge',
                    'source_branch': source_branch,
                    'merged_layers': layer_names,
                    'strategy': strategy
                }
            )
            
            layers_str = ', '.join(layer_names)
            print(f"✓ Merged layers [{layers_str}] from '{source_branch}' - Snapshot ID: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            print(f"Error merging layers: {e}")
            return None

    def layer_status(self, layer_names: Optional[Union[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Show status of specific layers or all layers.
        
        Args:
            layer_names: Name(s) of layers to check (None for all layers)
            
        Returns:
            Dictionary with layer status information
        """
        try:
            analyzer = self.get_analyzer()
            all_layers = analyzer.get_layer_names()
            
            if layer_names is None:
                target_layers = all_layers
            else:
                if isinstance(layer_names, str):
                    target_layers = [layer_names]
                else:
                    target_layers = layer_names
            
            status = {
                "branch": self.current_branch,
                "total_layers": len(all_layers),
                "checked_layers": len(target_layers),
                "layers": {}
            }
            
            for layer_name in target_layers:
                if layer_name not in all_layers:
                    status["layers"][layer_name] = {"status": "not_found"}
                    continue
                
                try:
                    layer_info = analyzer.get_layer_info(layer_name)
                    
                    layer_status = {
                        "status": "available",
                        "tensor_types": layer_info.get("tensor_types", []),
                        "tensor_count": sum(len(tensors) for tensors in layer_info.get("tensors", {}).values()),
                        "has_weights": "weights" in layer_info.get("tensor_types", []),
                        "has_gradients": "gradients" in layer_info.get("tensor_types", []),
                    }
                    
                    # Add additional metadata if available
                    for key in ["type", "has_gradients"]:
                        if key in layer_info:
                            layer_status[key] = layer_info[key]
                    
                    status["layers"][layer_name] = layer_status
                    
                except Exception as e:
                    status["layers"][layer_name] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return status
            
        except Exception as e:
            return {"error": f"Could not get layer status: {e}"}

    def rebase_layer(
        self,
        layer_names: Union[str, List[str]],
        source_branch: str,
        target_branch: Optional[str] = None,
        conflict_strategy: str = 'detect'
    ) -> str:
        """
        Rebase specific layers from one branch onto another.
        
        Args:
            layer_names: Name(s) of layers to rebase (string or list)
            source_branch: Branch containing the layers to rebase
            target_branch: Branch to rebase onto (defaults to current)
            conflict_strategy: How to handle conflicts ('detect', 'ours', 'theirs')
            
        Returns:
            Snapshot ID of the rebase result
        """
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        if target_branch is None:
            target_branch = self.current_branch or 'main'
        
        print(f"Rebasing layers [{', '.join(layer_names)}] from '{source_branch}' onto '{target_branch}'")
        
        try:
            # Get layer parameters from source branch
            source_snapshot = self._resolve_reference_to_snapshot(source_branch)
            source_params = self.storage_manager.load_parameters_from_snapshot(source_snapshot)
            
            # Get layer parameters from target branch
            target_snapshot = self._resolve_reference_to_snapshot(target_branch)
            target_params = self.storage_manager.load_parameters_from_snapshot(target_snapshot)
            
            # Extract only the specified layers from source
            layer_params = {}
            for param_name, param_data in source_params.items():
                for layer_name in layer_names:
                    if param_name.startswith(f"{layer_name}/"):
                        layer_params[param_name] = param_data
                        break
            
            if not layer_params:
                raise ValueError(f"No parameters found for layers {layer_names} in source branch")
            
            # Handle conflicts if any
            conflicts = []
            for param_name in layer_params:
                if param_name in target_params:
                    if not np.array_equal(layer_params[param_name], target_params[param_name]):
                        conflicts.append(param_name)
            
            if conflicts and conflict_strategy == 'detect':
                raise ValueError(f"Conflicts detected in parameters: {conflicts}")
            elif conflicts and conflict_strategy == 'theirs':
                # Keep target branch version for conflicts
                for param_name in conflicts:
                    layer_params[param_name] = target_params[param_name]
            # For 'ours' strategy, we keep the source version (default behavior)
            
            # Commit the rebased layers
            message = f"Rebase layers [{', '.join(layer_names)}] from '{source_branch}' onto '{target_branch}'"
            
            snapshot_id = self.storage_manager.commit_model_state(
                model_parameters=layer_params,
                message=message,
                branch_name=target_branch,
                additional_metadata={
                    'rebase_type': 'layer_rebase',
                    'source_branch': source_branch,
                    'rebased_layers': layer_names,
                    'conflict_strategy': conflict_strategy,
                    'conflicts_resolved': len(conflicts)
                }
            )
            
            layers_str = ', '.join(layer_names)
            print(f"✓ Rebased layers [{layers_str}] - Snapshot ID: {snapshot_id}")
            if conflicts:
                print(f"✓ Resolved {len(conflicts)} conflicts using strategy '{conflict_strategy}'")
            
            return snapshot_id
            
        except Exception as e:
            print(f"Error rebasing layers: {e}")
            raise

    def reset_layer(
        self,
        layer_names: Union[str, List[str]],
        to_reference: str,
        target_model_instance: Any
    ) -> None:
        """
        Reset specific layers to a previous state.
        
        Args:
            layer_names: Name(s) of layers to reset (string or list)
            to_reference: Reference to reset to (snapshot ID, branch, or tag)
            target_model_instance: Model to apply the reset layers to
        """
        # Normalize layer_names to list
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        print(f"Resetting layers [{', '.join(layer_names)}] to '{to_reference}'")
        
        # This is essentially the same as checkout_layer with replace strategy
        self.checkout_layer(to_reference, layer_names, target_model_instance, strategy='replace')
        
        layers_str = ', '.join(layer_names)
        print(f"✓ Reset layers [{layers_str}] to '{to_reference}'") 