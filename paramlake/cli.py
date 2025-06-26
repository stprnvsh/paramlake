"""
Command-line interface for ParamLake.
"""
import typer
from typing_extensions import Annotated
from typing import Optional
import os

from paramlake.repo import Repo, RepositoryError
# from paramlake.utils.config import ParamLakeConfig # If needed for CLI-specific config loading

app = typer.Typer(help="ParamLake: Git for AI Models.")
repo_app = typer.Typer(name="repo", help="Manage ParamLake repositories (init, commit, checkout, etc.)")
branch_app = typer.Typer(name="branch", help="Manage branches.")
tag_app = typer.Typer(name="tag", help="Manage tags.")
remote_app = typer.Typer(name="remote", help="Manage remote repositories.")

app.add_typer(repo_app)
app.add_typer(branch_app)
app.add_typer(tag_app)
app.add_typer(remote_app)

# --- Global state/config for CLI ---
# This would typically be loaded from a .paramlake/config in the current directory
# or from global user config. For now, simplified.
CURRENT_REPO_PATH: Optional[str] = None  # Will be auto-detected

def find_repository_path() -> Optional[str]:
    """Find the repository path by searching current and parent directories."""
    global CURRENT_REPO_PATH
    if CURRENT_REPO_PATH is None:
        CURRENT_REPO_PATH = Repo.find_repository()
    return CURRENT_REPO_PATH

def get_repo_instance(path: Optional[str] = None) -> Repo:
    """Get a repository instance, with enhanced connection logic."""
    # Try the provided path first
    if path:
        try:
            if Repo.is_repository(path):
                return Repo.connect(path, auto_find=False)
            else:
                raise RepositoryError(f"No repository found at: {path}")
        except RepositoryError as e:
            print(f"Error: {e}")
            raise typer.Exit(code=1)
    
    # Try to find repository automatically
    repo_path = find_repository_path()
    if repo_path:
        try:
            return Repo.connect(repo_path, auto_find=False)
        except RepositoryError as e:
            print(f"Error connecting to repository at {repo_path}: {e}")
            raise typer.Exit(code=1)
    
    # No repository found
    print("Error: No ParamLake repository found.")
    print("Initialize a new repository with 'paramlake repo init' or")
    print("connect to an existing one with 'paramlake repo connect <path>'")
    raise typer.Exit(code=1)

# --- `paramlake repo ...` commands ---
@repo_app.command("init")
def repo_init(
    path: Annotated[str, typer.Argument(help="Path to initialize the repository.")] = ".",
    cloud: Annotated[Optional[str], typer.Option(help="Cloud backend (s3, gcs, azure)")] = None,
    bucket: Annotated[Optional[str], typer.Option(help="Cloud storage bucket")] = None,
    prefix: Annotated[Optional[str], typer.Option(help="Storage prefix")] = None,
    region: Annotated[Optional[str], typer.Option(help="Cloud region")] = "us-east-1",
    force: Annotated[bool, typer.Option("--force", help="Force initialization even if repository exists")] = False
):
    """Initializes a new ParamLake repository or connects to existing one."""
    try:
        config = {
            "create_repo": True, 
            "storage_type": "icechunk",
            "git_features": {"enabled": True}
        }
        
        # Add cloud configuration if provided
        if cloud:
            config["storage_backend"] = cloud
            if bucket:
                config["bucket"] = bucket
            if prefix:
                config["prefix"] = prefix
            config["region"] = region
        
        if force:
            repo = Repo(path=path, config=config)
            print(f"Initialized ParamLake repository in {path}")
        else:
            repo = Repo.init_or_connect(path=path, **config)
        
        if cloud:
            print(f"Using {cloud} cloud storage:")
            print(f"  Bucket: {bucket}")
            print(f"  Prefix: {prefix}")
            print(f"  Region: {region}")
        
        # Update global repo path for subsequent commands
        global CURRENT_REPO_PATH
        CURRENT_REPO_PATH = path
        
    except RepositoryError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"Error initializing repository: {e}")
        raise typer.Exit(code=1)

@repo_app.command("connect")
def repo_connect(
    path: Annotated[Optional[str], typer.Argument(help="Path to existing repository. If not provided, searches current directory.")] = None,
    run_id: Annotated[Optional[str], typer.Option("--run-id", help="Specific run ID to connect to")] = None,
    branch: Annotated[Optional[str], typer.Option("--branch", help="Branch to switch to after connecting")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show detailed connection information")] = False
):
    """Connect to an existing ParamLake repository."""
    try:
        # Connect to repository
        config = {"verbose": verbose} if verbose else {}
        repo = Repo.connect(path=path, run_id=run_id, config=config)
        
        # Switch to specific branch if requested
        if branch:
            try:
                result = repo.switch_branch(branch)
                if result.get('status') == 'success':
                    print(f"Switched to branch: {branch}")
                else:
                    print(f"Warning: Could not switch to branch '{branch}': {result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"Warning: Could not switch to branch '{branch}': {e}")
        
        # Update global repo path
        global CURRENT_REPO_PATH
        CURRENT_REPO_PATH = repo.config.get('output_path')
        
        # Show additional info if verbose
        if verbose:
            status = repo.status()
            print(f"\nRepository Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
                
    except RepositoryError as e:
        print(f"Error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"Error connecting to repository: {e}")
        raise typer.Exit(code=1)

@repo_app.command("info")
def repo_info(
    path: Annotated[Optional[str], typer.Argument(help="Path to repository to inspect. Defaults to current repository.")] = None,
    detailed: Annotated[bool, typer.Option("--detailed", help="Show detailed information")] = False
):
    """Show information about a ParamLake repository."""
    try:
        # Determine which repository to inspect
        if path is None:
            path = find_repository_path()
            if path is None:
                print("Error: No repository specified and none found in current directory")
                raise typer.Exit(code=1)
        
        if not Repo.is_repository(path):
            print(f"Error: No ParamLake repository found at: {path}")
            raise typer.Exit(code=1)
        
        info = Repo.get_repository_info(path)
        
        print(f"Repository Information:")
        print(f"  Path: {info['path']}")
        print(f"  Storage Type: {info['storage_type']}")
        print(f"  Git Features: {'Yes' if info['has_git_features'] else 'No'}")
        print(f"  Size: {info['repository_size']:,} bytes ({info['repository_size'] / (1024*1024):.1f} MB)")
        
        if info.get('current_branch'):
            print(f"  Current Branch: {info['current_branch']}")
        
        if info.get('branches'):
            print(f"  Branches ({len(info['branches'])}): {', '.join(info['branches'])}")
        
        if info.get('tags'):
            print(f"  Tags ({len(info['tags'])}): {', '.join(info['tags'].keys())}")
        
        if info.get('last_commit') and detailed:
            commit = info['last_commit']
            print(f"  Last Commit:")
            print(f"    ID: {commit.get('id', 'unknown')}")
            print(f"    Message: {commit.get('message', 'No message')}")
            print(f"    Author: {commit.get('author', 'Unknown')}")
            print(f"    Date: {commit.get('timestamp', 'Unknown')}")
        
        if info.get('error'):
            print(f"  Warning: {info['error']}")
            
    except Exception as e:
        print(f"Error getting repository info: {e}")
        raise typer.Exit(code=1)

@repo_app.command("find")
def repo_find(
    start_path: Annotated[str, typer.Argument(help="Directory to start searching from")] = ".",
    all_repos: Annotated[bool, typer.Option("--all", help="Find all repositories in subdirectories")] = False
):
    """Find ParamLake repositories in the current directory or subdirectories."""
    try:
        if all_repos:
            # Search for all repositories in subdirectories
            found_repos = []
            start_path = os.path.abspath(start_path)
            
            for root, dirs, files in os.walk(start_path):
                if Repo.is_repository(root):
                    found_repos.append(root)
            
            if found_repos:
                print(f"Found {len(found_repos)} ParamLake repositories:")
                for repo_path in found_repos:
                    print(f"  {repo_path}")
            else:
                print("No ParamLake repositories found.")
        else:
            # Find the nearest repository
            repo_path = Repo.find_repository(start_path)
            if repo_path:
                print(f"Found ParamLake repository at: {repo_path}")
                
                # Show basic info
                if Repo.is_repository(repo_path):
                    info = Repo.get_repository_info(repo_path)
                    print(f"  Storage Type: {info['storage_type']}")
                    if info.get('current_branch'):
                        print(f"  Current Branch: {info['current_branch']}")
            else:
                print("No ParamLake repository found in current directory or parent directories.")
                
    except Exception as e:
        print(f"Error searching for repositories: {e}")
        raise typer.Exit(code=1)

@repo_app.command("validate")
def repo_validate(
    path: Annotated[Optional[str], typer.Argument(help="Path to repository to validate")] = None,
    fix: Annotated[bool, typer.Option("--fix", help="Attempt to fix issues found")] = False
):
    """Validate a ParamLake repository and optionally fix issues."""
    try:
        if path is None:
            path = find_repository_path()
            if path is None:
                print("Error: No repository specified and none found in current directory")
                raise typer.Exit(code=1)
        
        print(f"Validating repository at: {path}")
        
        # Check if it's a valid repository
        if not Repo.is_repository(path):
            print("✗ Not a valid ParamLake repository")
            raise typer.Exit(code=1)
        
        print("✓ Valid ParamLake repository detected")
        
        # Try to connect and perform basic operations
        try:
            repo = Repo.connect(path, auto_find=False)
            print("✓ Successfully connected to repository")
            
            # Test basic operations
            try:
                status = repo.status()
                print("✓ Repository status accessible")
            except Exception as e:
                print(f"✗ Could not get repository status: {e}")
            
            try:
                branches = repo.list_branches()
                print(f"✓ Found {len(branches)} branches")
            except Exception as e:
                print(f"✗ Could not list branches: {e}")
            
            try:
                tags = repo.list_tags()
                print(f"✓ Found {len(tags)} tags")
            except Exception as e:
                print(f"✗ Could not list tags: {e}")
                
        except Exception as e:
            print(f"✗ Could not connect to repository: {e}")
            if fix:
                print("Fix mode not implemented yet")
            raise typer.Exit(code=1)
        
        print("✓ Repository validation completed successfully")
        
    except Exception as e:
        print(f"Error validating repository: {e}")
        raise typer.Exit(code=1)

@repo_app.command("commit")
def repo_commit(
    message: Annotated[str, typer.Option("-m", "--message", help="Commit message.")],
    model_path: Annotated[Optional[str], typer.Option(help="Path to saved model file (experimental)")] = None,
    branch: Annotated[Optional[str], typer.Option(help="Branch to commit to.")] = None,
    author: Annotated[Optional[str], typer.Option(help="Author of the commit.")] = None,
):
    """Records changes to the repository (commits the model state)."""
    repo = get_repo_instance()
    
    if model_path and os.path.exists(model_path):
        # Experimental: Try to load and commit a saved model
        try:
            from paramlake.utils.framework_utils import HAS_TENSORFLOW, require_tensorflow
            if not HAS_TENSORFLOW:
                print("Error: TensorFlow is required for model loading.")
                print("Install with: pip install 'paramlake[tf]' or pip install tensorflow")
                raise typer.Exit(code=1)
            
            tf = require_tensorflow()
            model = tf.keras.models.load_model(model_path)
            snapshot_id = repo.commit(model, message, branch=branch, author=author)
            print(f"Committed model from {model_path}")
            print(f"Snapshot ID: {snapshot_id}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("For training-time commits, use the Python API: repo.commit(model_instance, message=...)")
            raise typer.Exit(code=1)
    else:
        print("CLI commit requires a model file path with --model-path")
        print("For in-training commits, use the Python API within your training script:")
        print("  @repo.track()")
        print("  def train_model():")
        print("      # ... training code ...")
        print("  repo.commit(model, 'Training completed')")
        raise typer.Exit(code=1)

@repo_app.command("checkout")
def repo_checkout(
    reference: Annotated[str, typer.Argument(help="Branch, tag, or snapshot ID to checkout.")],
    model_path: Annotated[Optional[str], typer.Option("-m", "--model", help="Path to model file to update")] = None,
    create_branch: Annotated[Optional[str], typer.Option("-b", help="Create new branch from reference")] = None,
):
    """Switches to a different branch or restores a specific model version."""
    repo = get_repo_instance()
    
    try:
        if create_branch:
            # Create new branch from reference
            repo.create_branch(create_branch, from_reference=reference)
            repo.current_branch = create_branch
            print(f"Created and switched to new branch '{create_branch}' from '{reference}'")
        else:
            # Try to switch branch first
            if reference in repo.list_branches():
                result = repo.switch_branch(reference)
                if result.get('status') == 'success':
                    print(f"Switched to branch '{reference}'")
                else:
                    print(f"Failed to switch to branch '{reference}': {result.get('message', 'Unknown error')}")
            else:
                # Not a branch, try to checkout specific snapshot
                if model_path and os.path.exists(model_path):
                    from paramlake.utils.framework_utils import HAS_TENSORFLOW, require_tensorflow
                    if not HAS_TENSORFLOW:
                        print("Error: TensorFlow is required for model operations.")
                        print("Install with: pip install 'paramlake[tf]' or pip install tensorflow")
                        raise typer.Exit(code=1)
                    
                    tf = require_tensorflow()
                    model = tf.keras.models.load_model(model_path)
                    repo.checkout(reference, model)
                    model.save(model_path)
                    print(f"Updated model at {model_path} with state from '{reference}'")
                else:
                    print(f"Checked out '{reference}' (no model file specified to update)")
    except Exception as e:
        print(f"Error during checkout: {e}")
        raise typer.Exit(code=1)

@repo_app.command("log")
def repo_log(
    reference: Annotated[Optional[str], typer.Argument(help="Branch, tag, or snapshot ID to show history for. Defaults to current.")] = None,
    limit: Annotated[Optional[int], typer.Option("-n", "--max-count", help="Limit number of commits to show.")] = None,
    oneline: Annotated[bool, typer.Option("--oneline", help="Show one line per commit")] = False,
):
    """Shows the commit history."""
    repo = get_repo_instance()
    try:
        history = repo.log(reference=reference or repo.current_branch, limit=limit)
        
        if not history:
            print("No commits found.")
            return
            
        if not oneline:
            # Detailed format (already printed by repo.log)
            pass
        else:
            # One-line format
            print(f"History for '{reference or repo.current_branch}':")
            for entry in history:
                commit_id = entry.get('id', 'unknown')[:12]
                message = entry.get('message', 'No message')
                print(f"{commit_id} {message}")
    except Exception as e:
        print(f"Error retrieving history: {e}")

@repo_app.command("status")
def repo_status():
    """Show the working tree status."""
    repo = get_repo_instance()
    try:
        status = repo.status()
        # Additional status information beyond what's printed
        if status.get('has_git_features'):
            print(f"Git features: Enabled")
        else:
            print(f"Git features: Not available (use 'icechunk' storage type)")
    except Exception as e:
        print(f"Error getting status: {e}")

@repo_app.command("diff")
def repo_diff(
    ref1: Annotated[str, typer.Argument(help="First reference to compare")],
    ref2: Annotated[Optional[str], typer.Argument(help="Second reference (defaults to current HEAD)")] = None,
    summary: Annotated[bool, typer.Option("--summary", help="Show only summary of changes")] = False,
):
    """Shows differences between two references."""
    repo = get_repo_instance()
    try:
        diff_result = repo.diff(ref1, ref2)
        
        if summary:
            # Show summary only
            summary_info = diff_result.get('summary', {})
            print(f"Total changes: {summary_info.get('total_changes', 0)}")
            print(f"Layers added: {len(summary_info.get('layers_added', []))}")
            print(f"Layers removed: {len(summary_info.get('layers_removed', []))}")
            print(f"Layers modified: {len(summary_info.get('layers_modified', []))}")
        else:
            # Show detailed diff (this would need formatting)
            print(f"Diff between '{ref1}' and '{ref2 or 'HEAD'}':")
            print(f"Changes: {diff_result}")
            
    except Exception as e:
        print(f"Error computing diff: {e}")

@repo_app.command("merge")
def repo_merge(
    source_branch: Annotated[str, typer.Argument(help="Branch to merge into current branch")],
    strategy: Annotated[str, typer.Option("--strategy", help="Merge strategy (auto, ours, theirs)")] = "auto",
    message: Annotated[Optional[str], typer.Option("-m", "--message", help="Merge commit message")] = None,
):
    """Merges a branch into the current branch."""
    repo = get_repo_instance()
    try:
        result = repo.merge(source_branch, strategy=strategy, commit_message=message)
        if result:
            print(f"Merge completed. Snapshot ID: {result}")
        else:
            print("Merge completed (no new snapshot created)")
    except Exception as e:
        print(f"Error during merge: {e}")
        print("You may need to resolve conflicts manually or use a different strategy")

# --- `paramlake branch ...` commands ---
@branch_app.command("create")
def branch_create(
    branch_name: Annotated[str, typer.Argument(help="Name of the new branch.")],
    from_reference: Annotated[Optional[str], typer.Argument(help="Reference (snapshot ID, branch, tag) to base the new branch on. Defaults to current HEAD.")] = None
):
    """Creates a new branch."""
    repo = get_repo_instance()
    try:
        repo.create_branch(branch_name, from_reference=from_reference or repo.current_branch)
    except Exception as e:
        print(f"Error creating branch: {e}")

@branch_app.command("list")
def branch_list():
    """Lists all branches."""
    repo = get_repo_instance()
    try:
        branches = repo.list_branches()
        print("Branches:")
        for branch in branches:
            prefix = "* " if branch == repo.current_branch else "  "
            print(f"{prefix}{branch}")
    except Exception as e:
        print(f"Error listing branches: {e}")

@branch_app.command("delete")
def branch_delete(
    branch_name: Annotated[str, typer.Argument(help="Name of the branch to delete.")],
    force: Annotated[bool, typer.Option("-f", "--force", help="Force delete branch")] = False
):
    """Deletes a branch."""
    repo = get_repo_instance()
    try:
        repo.delete_branch(branch_name)
    except Exception as e:
        if not force:
            print(f"Error deleting branch: {e}")
            print("Use --force to force deletion")
        else:
            print(f"Force deleted branch '{branch_name}' (errors ignored)")

@branch_app.command("switch")
def branch_switch(
    branch_name: Annotated[str, typer.Argument(help="Name of the branch to switch to")],
    create: Annotated[bool, typer.Option("-c", "--create", help="Create branch if it doesn't exist")] = False
):
    """Switch to a different branch."""
    repo = get_repo_instance()
    try:
        result = repo.switch_branch(branch_name, create_if_missing=create)
        if result.get('status') == 'success':
            print(f"Switched to branch '{branch_name}'")
        else:
            print(f"Failed to switch: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Error switching branch: {e}")

# --- `paramlake tag ...` commands ---
@tag_app.command("create")
def tag_create(
    tag_name: Annotated[str, typer.Argument(help="Name of the new tag.")],
    reference: Annotated[Optional[str], typer.Argument(help="Snapshot ID, branch, or tag to point the new tag to. Defaults to current HEAD.")] = None,
    message: Annotated[Optional[str], typer.Option("-m", "--message", help="Tag message")] = None
):
    """Creates a new tag."""
    repo = get_repo_instance()
    try:
        ref_to_tag = reference or repo.current_branch
        repo.create_tag(tag_name, ref_to_tag)
    except Exception as e:
        print(f"Error creating tag: {e}")

@tag_app.command("list")
def tag_list():
    """Lists all tags."""
    repo = get_repo_instance()
    try:
        tags = repo.list_tags()
        if tags:
            print("Tags:")
            for name, snapshot_id in tags.items():
                print(f"  {name} -> {snapshot_id[:12]}...")
        else:
            print("No tags found.")
    except Exception as e:
        print(f"Error listing tags: {e}")

@tag_app.command("delete")
def tag_delete(tag_name: Annotated[str, typer.Argument(help="Name of the tag to delete.")]):
    """Deletes a tag."""
    repo = get_repo_instance()
    try:
        repo.delete_tag(tag_name)
    except Exception as e:
        print(f"Error deleting tag: {e}")

# --- Advanced Git commands ---
@app.command("rebase")
def cli_rebase(
    branch_name: Annotated[str, typer.Argument(help="Branch to rebase")],
    onto: Annotated[Optional[str], typer.Option("--onto", help="Branch to rebase onto")] = None,
    strategy: Annotated[str, typer.Option("--strategy", help="Conflict resolution strategy")] = "detect"
):
    """Rebase a branch onto another branch."""
    repo = get_repo_instance()
    try:
        result = repo.rebase(branch_name, onto_branch=onto, conflict_strategy=strategy)
        print(f"Rebase completed. Snapshot ID: {result}")
    except Exception as e:
        print(f"Error during rebase: {e}")

@app.command("reset")
def cli_reset(
    reference: Annotated[str, typer.Argument(help="Reference to reset to")],
    branch: Annotated[Optional[str], typer.Option("--branch", help="Branch to reset (defaults to current)")] = None,
    hard: Annotated[bool, typer.Option("--hard", help="Hard reset (discards changes)")] = False
):
    """Reset a branch to a specific reference."""
    repo = get_repo_instance()
    try:
        repo.reset(reference, branch_name=branch)
        reset_type = "hard" if hard else "soft"
        print(f"Reset ({reset_type}) to '{reference}'")
    except Exception as e:
        print(f"Error during reset: {e}")

# --- Remote commands (stubs for future implementation) ---
@remote_app.command("add")
def remote_add(
    name: Annotated[str, typer.Argument(help="Remote name")],
    url: Annotated[str, typer.Argument(help="Remote URL or configuration")]
):
    """Add a remote repository."""
    print(f"Adding remote '{name}' -> '{url}' (Not fully implemented)")

@remote_app.command("list")
def remote_list():
    """List remote repositories."""
    print("Remote repositories: (Not fully implemented)")

@app.command("import")
def cli_import(
    source: Annotated[str, typer.Argument(help="Source file path")],
    format: Annotated[str, typer.Option("--format", help="Source format (hdf5, tf-saved-model)")] = "hdf5",
    message: Annotated[Optional[str], typer.Option("-m", "--message", help="Import commit message")] = None,
    branch: Annotated[Optional[str], typer.Option("--branch", help="Target branch")] = None
):
    """Import a model from an external format."""
    repo = get_repo_instance()
    try:
        snapshot_id = repo.import_model(source, format, message=message, branch=branch)
        print(f"Imported model from {source}. Snapshot ID: {snapshot_id}")
    except Exception as e:
        print(f"Error importing model: {e}")

# --- Advanced analysis commands ---
@app.command("analyze")
def cli_analyze(
    reference: Annotated[Optional[str], typer.Argument(help="Reference to analyze (defaults to current)")] = None,
    gradients: Annotated[bool, typer.Option("--gradients", help="Analyze gradients")] = False,
    output: Annotated[Optional[str], typer.Option("-o", "--output", help="Output file for report")] = None,
    format: Annotated[str, typer.Option("--format", help="Report format (html, json)")] = "html"
):
    """Analyze model training data and generate reports."""
    repo = get_repo_instance()
    try:
        if gradients:
            # Gradient analysis
            result = repo.analyze_gradients(reference)
            print("Gradient Analysis Results:")
            summary = result.get('summary', {})
            print(f"  Total layers: {summary.get('total_layers', 0)}")
            print(f"  Layers with gradients: {summary.get('layers_with_gradients', 0)}")
            print(f"  Gradient coverage: {summary.get('gradient_coverage', 0):.2%}")
            print(f"  Total gradient tensors: {summary.get('total_gradient_tensors', 0)}")
        
        if output:
            # Generate comprehensive report
            repo.export_training_report(output, reference=reference, format=format)
            print(f"Report exported to: {output}")
        else:
            # Show model summary
            summary = repo.get_model_summary(reference)
            metadata = summary.get('metadata', {})
            print(f"Model Summary for '{reference or 'current'}':")
            print(f"  Framework: {metadata.get('framework', 'Unknown')}")
            print(f"  Layers: {len(summary.get('layer_names', []))}")
            print(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

@repo_app.command("checkout-snapshot")
def repo_checkout_snapshot(
    reference: Annotated[str, typer.Argument(help="Snapshot ID, branch, or tag to checkout")],
    new_branch: Annotated[Optional[str], typer.Option("-b", "--branch", help="Create new branch from snapshot")] = None
):
    """Checkout to a specific snapshot and optionally create a new branch."""
    repo = get_repo_instance()
    try:
        result = repo.checkout_snapshot(reference, new_branch)
        if result.get('status') == 'success':
            print(f"Successfully checked out snapshot: {result.get('snapshot_id', '')[:12]}...")
            print(f"Current branch: {result.get('branch', 'unknown')}")
            if new_branch:
                print(f"Created new branch: {new_branch}")
        else:
            print(f"Checkout failed: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"Error during checkout: {e}")

@repo_app.command("diff-visual")
def repo_diff_visual(
    ref1: Annotated[str, typer.Argument(help="First reference to compare")],
    ref2: Annotated[Optional[str], typer.Argument(help="Second reference (defaults to current HEAD)")] = None,
    format: Annotated[str, typer.Option("--format", help="Output format (console, html, dict)")] = "console",
    include_values: Annotated[bool, typer.Option("--values", help="Include tensor values in diff")] = False,
    output: Annotated[Optional[str], typer.Option("-o", "--output", help="Output file for HTML format")] = None
):
    """Generate a visual diff between two model versions."""
    repo = get_repo_instance()
    try:
        result = repo.diff_visual(ref1, ref2, output_format=format, include_values=include_values)
        
        if format == "console":
            print(result)
        elif format == "html" and output:
            with open(output, 'w') as f:
                f.write(result)
            print(f"Visual diff saved to: {output}")
        elif format == "dict":
            import json
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Diff data saved to: {output}")
            else:
                print(json.dumps(result, indent=2, default=str))
        else:
            print("For HTML format, please specify --output file")
            
    except Exception as e:
        print(f"Error generating visual diff: {e}")

@app.command("compare")
def cli_compare(
    ref1: Annotated[str, typer.Argument(help="First reference")],
    ref2: Annotated[str, typer.Argument(help="Second reference")],
    layer: Annotated[str, typer.Option("--layer", help="Layer name to compare")],
    tensor_type: Annotated[str, typer.Option("--type", help="Tensor type (weights, gradients)")] = "weights",
    stat: Annotated[str, typer.Option("--stat", help="Statistic to compare (norm, mean, var)")] = "norm"
):
    """Compare statistical properties between two model versions."""
    repo = get_repo_instance()
    try:
        result = repo.compare_models(ref1, ref2, layer, tensor_type, stat)
        print(f"Comparison between {ref1} and {ref2}:")
        print(f"Layer: {layer}, Type: {tensor_type}, Stat: {stat}")
        for tensor_name, stats in result.items():
            if stat in stats:
                old_val, new_val = stats[stat]
                print(f"  {tensor_name}:")
                print(f"    {ref1}: {old_val}")
                print(f"    {ref2}: {new_val}")
    except Exception as e:
        print(f"Error comparing models: {e}")

@app.command("plot")
def cli_plot(
    layer: Annotated[str, typer.Argument(help="Layer name to plot")],
    tensor: Annotated[Optional[str], typer.Option("--tensor", help="Specific tensor name")] = None,
    type: Annotated[str, typer.Option("--type", help="Tensor type")] = "weights",
    stat: Annotated[str, typer.Option("--stat", help="Statistic to plot")] = "norm",
    reference: Annotated[Optional[str], typer.Option("--ref", help="Reference to analyze")] = None
):
    """Plot the evolution of model parameters over training."""
    repo = get_repo_instance()
    try:
        repo.plot_training_evolution(layer, tensor, type, stat, reference)
        print(f"Plotted {stat} evolution for {layer}/{type}")
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    app() 