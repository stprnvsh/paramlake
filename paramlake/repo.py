"""
Provides the Repo class for interacting with a ParamLake version-controlled repository.
"""
from typing import Any, Dict, List, Optional, Union, Callable
import functools
import tensorflow as tf # Assuming TensorFlow for now for model type hint
import numpy as np

from paramlake.storage.storage_interface import StorageInterface
from paramlake.storage.factory import create_storage_manager
from paramlake.utils.config import ParamLakeConfig

# Helper to check if Icechunk is available at module level
try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False


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
        
        # Try to get actual current branch from storage manager if available
        try:
            if hasattr(self.storage_manager, 'session') and hasattr(self.storage_manager.session, 'branch'):
                self.current_branch = self.storage_manager.session.branch
        except:
            pass

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
                
                # Override Model.fit to include our callback
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
                    if isinstance(result, tf.keras.Model):
                        if callback.weight_collector is not None:
                            callback.weight_collector.capture_model_weights(result, step=0)
                    
                    return result
                
                finally:
                    # Always restore original fit method
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
        """Helper to extract parameters from a model object."""
        # This needs to be framework-specific. Placeholder for Keras:
        if isinstance(model, tf.keras.Model):
            params = {}
            
            # Create a mapping of weight objects to their layer information
            weight_to_layer = {}
            for layer in model.layers:
                for weight in layer.weights:
                    weight_to_layer[id(weight)] = layer.name
            
            for i, weight_var in enumerate(model.weights):
                # Create unique name using layer name and weight name
                layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
                weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                
                # Combine layer name and weight name for uniqueness
                unique_name = f"{layer_name}/{weight_name}"
                
                # Debug logging if verbose mode is enabled
                if self.config.get("verbose", False):
                    print(f"Extracting parameter: '{unique_name}' -> shape: {weight_var.shape}")
                
                params[unique_name] = weight_var.numpy()
            return params
        # TODO: Add support for PyTorch (model.state_dict()), etc.
        raise NotImplementedError("Model parameter extraction not implemented for this model type.")

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
            if isinstance(target_model_instance, tf.keras.Model):
                if isinstance(parameters, dict):
                    # Create mapping from weight objects to their layer information (same as in extract)
                    weight_to_layer = {}
                    for layer in target_model_instance.layers:
                        for weight in layer.weights:
                            weight_to_layer[id(weight)] = layer.name
                    
                    weights_to_set = []
                    found_weights = {}
                    
                    # First pass: try exact name matching
                    for i, weight_var in enumerate(target_model_instance.weights):
                        # Recreate the unique name using the same logic as extraction
                        layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
                        weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                        unique_name = f"{layer_name}/{weight_name}"
                        
                        # Try to find the parameter with the unique name
                        # Also try the sanitized version for backward compatibility
                        sanitized_name = unique_name.replace("/", "_").replace(":", "_")
                        
                        param_data = None
                        matched_key = None
                        
                        # Try multiple matching strategies
                        for key_to_try in [unique_name, sanitized_name, weight_var.name]:
                            if key_to_try in parameters:
                                param_data = parameters[key_to_try]
                                matched_key = key_to_try
                                break
                        
                        if param_data is not None:
                            # Verify shapes match to prevent mismatch errors
                            if param_data.shape == weight_var.shape:
                                weights_to_set.append(param_data)
                                found_weights[unique_name] = True
                                if self.config.get("verbose", False):
                                    print(f"✓ Matched '{unique_name}' -> '{matched_key}' with shape {param_data.shape}")
                            else:
                                print(f"Warning: Shape mismatch for {unique_name}. Expected {weight_var.shape}, got {param_data.shape}. Using existing weights.")
                                weights_to_set.append(weight_var.numpy())
                        else:
                            # No exact match found, we'll handle this in the fallback
                            weights_to_set.append(None)  # Placeholder
                    
                    # Second pass: positional fallback for unmatched weights
                    has_unmatched = any(w is None for w in weights_to_set)
                    if has_unmatched:
                        print("Some weights not found by name, trying positional matching...")
                        
                        # Sort parameters to match the order they were stored in
                        # We need to recreate the same order as in _extract_model_parameters
                        stored_order = []
                        for i, weight_var in enumerate(target_model_instance.weights):
                            layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
                            weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                            unique_name = f"{layer_name}/{weight_name}"
                            sanitized_name = unique_name.replace("/", "_").replace(":", "_")
                            
                            # Try to find this parameter in the stored parameters
                            if unique_name in parameters:
                                stored_order.append(parameters[unique_name])
                            elif sanitized_name in parameters:
                                stored_order.append(parameters[sanitized_name])
                            elif weight_var.name in parameters:
                                stored_order.append(parameters[weight_var.name])
                            else:
                                stored_order.append(None)
                        
                        for i, weight_var in enumerate(target_model_instance.weights):
                            if weights_to_set[i] is None:  # This weight wasn't matched by name
                                if i < len(stored_order) and stored_order[i] is not None:
                                    param_data = stored_order[i]
                                    if param_data.shape == weight_var.shape:
                                        weights_to_set[i] = param_data
                                        layer_name = weight_to_layer.get(id(weight_var), f"layer_{i}")
                                        weight_name = weight_var.name if weight_var.name else f"weight_{i}"
                                        unique_name = f"{layer_name}/{weight_name}"
                                        found_weights[unique_name] = True
                                        if self.config.get("verbose", False):
                                            print(f"✓ Positionally matched '{unique_name}' at index {i} with shape {param_data.shape}")
                                    else:
                                        print(f"Warning: Positional shape mismatch at index {i}. Expected {weight_var.shape}, got {param_data.shape}. Using existing weights.")
                                        weights_to_set[i] = weight_var.numpy()
                                else:
                                    print(f"Warning: No parameter available at position {i}, using existing weights.")
                                    weights_to_set[i] = weight_var.numpy()
                    
                    if len(weights_to_set) == len(target_model_instance.weights):
                        try:
                            target_model_instance.set_weights(weights_to_set)
                            print(f"✓ Successfully loaded {len(found_weights)} weights from checkpoint")
                        except ValueError as e:
                            print(f"Error setting weights: {e}")
                            print("This may indicate a fundamental architecture mismatch between the saved and current model.")
                            raise
                    else:
                        print("Error: Mismatch in number of weights found in checkpoint for named assignment.")
                    
                elif isinstance(parameters, list):
                    # Direct list assignment - verify shapes first
                    if len(parameters) == len(target_model_instance.weights):
                        shape_mismatch = False
                        for i, (param, weight_var) in enumerate(zip(parameters, target_model_instance.weights)):
                            if param.shape != weight_var.shape:
                                print(f"Shape mismatch at index {i}: expected {weight_var.shape}, got {param.shape}")
                                shape_mismatch = True
                        
                        if not shape_mismatch:
                            target_model_instance.set_weights(parameters)
                        else:
                            print("Cannot set weights due to shape mismatches. Architecture may have changed.")
                    else:
                        print(f"Warning: Weight count mismatch. Model has {len(target_model_instance.weights)}, checkpoint has {len(parameters)}. Cannot set weights.")
            else:
                raise NotImplementedError("Checkout to this model type not implemented.")
            print(f"Loaded parameters from '{reference}' into model.")
        
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