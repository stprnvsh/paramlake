"""
Command-line interface for ParamLake.
"""
import typer
from typing_extensions import Annotated
from typing import Optional

from paramlake.repo import Repo
# from paramlake.utils.config import ParamLakeConfig # If needed for CLI-specific config loading

app = typer.Typer(help="ParamLake: Git for AI Models.")
repo_app = typer.Typer(name="repo", help="Manage ParamLake repositories (init, commit, checkout, etc.)")
branch_app = typer.Typer(name="branch", help="Manage branches.")
tag_app = typer.Typer(name="tag", help="Manage tags.")

app.add_typer(repo_app)
app.add_typer(branch_app)
app.add_typer(tag_app)

# --- Global state/config for CLI ---
# This would typically be loaded from a .paramlake/config in the current directory
# or from global user config. For now, simplified.
CURRENT_REPO_PATH: Optional[str] = "." # Assume current directory is a repo or contains one.

def get_repo_instance(path: Optional[str] = None) -> Repo:
    effective_path = path if path else CURRENT_REPO_PATH
    if not effective_path:
        print("Error: Repository path not specified or found. Initialize with 'paramlake repo init <path>' or run from within a repo.")
        raise typer.Exit(code=1)
    try:
        # Basic config for CLI usage - can be expanded
        # The Repo class should ideally discover run_id if path points to an existing repo
        # or allow it to be specified.
        repo_config = {"output_path": effective_path, "storage_type": "icechunk"} 
        return Repo(path=effective_path, config=repo_config)
    except Exception as e:
        print(f"Error opening repository at '{effective_path}': {e}")
        raise typer.Exit(code=1)

# --- `paramlake repo ...` commands ---
@repo_app.command("init")
def repo_init(path: Annotated[str, typer.Argument(help="Path to initialize the repository.")] = "."):
    """Initializes a new ParamLake repository."""
    try:
        # Repo __init__ with create_repo=True or similar logic needed in Repo/StorageManager
        # For now, assume Repo creation happens if path doesn't exist or is empty for IceChunk.
        Repo(path=path, config={"create_repo": True, "storage_type": "icechunk"}) # create_repo needs to be handled by config
        print(f"Initialized empty ParamLake repository in {path}")
        # TODO: Create .paramlake/config to store repo path for subsequent commands if path != "."
    except Exception as e:
        print(f"Error initializing repository: {e}")
        raise typer.Exit(code=1)

@repo_app.command("commit")
def repo_commit(
    message: Annotated[str, typer.Option("-m", "--message", help="Commit message.")],
    model_path: Annotated[str, typer.Option(help="Path to the model file/directory to commit (framework specific).")] = "", # Placeholder
    branch: Annotated[Optional[str], typer.Option(help="Branch to commit to.")] = None,
    author: Annotated[Optional[str], typer.Option(help="Author of the commit.")] = None,
):
    """Records changes to the repository (commits the model state)."""
    repo = get_repo_instance()
    if not model_path:
        print("Error: --model-path must be specified for committing via CLI (e.g., path to a saved Keras model).")
        print("For in-training commits, use the ParamLake Python API within your training script.")
        raise typer.Exit(code=1)
    
    # TODO: Implement model loading from model_path based on some convention or framework flag
    # This is a placeholder - in reality, you'd load the model here.
    # For this stub, we can't actually load a model without knowing its type and framework.
    print(f"Placeholder: Would load model from {model_path}")
    # mock_model_params = {"layer1/weights": np.random.rand(10,10).astype(np.float32)}
    # For now, cannot proceed without a model object.
    print("CLI commit of arbitrary model files is complex and needs framework-specific loading.")
    print("Please use the Python API: repo.commit(model_instance, message=...)")
    # repo.commit(mock_model_params, message, branch=branch, author=author)
    # print(f"Committed to branch '{branch or repo.current_branch}'.")
    raise typer.Exit(code=1)

@repo_app.command("checkout")
def repo_checkout(
    reference: Annotated[str, typer.Argument(help="Branch, tag, or snapshot ID to checkout.")],
    output_path: Annotated[Optional[str], typer.Option("-o", "--output", help="Path to export the checked-out model state (optional).")] = None,
):
    """Switches to a different branch or restores a specific model version."""
    repo = get_repo_instance()
    # repo.checkout(reference) # This API needs target_model_instance
    # For CLI, checkout usually means setting the current HEAD and optionally exporting.
    # We need to store the current reference (e.g. in .paramlake/HEAD)
    print(f"Placeholder: Setting current reference to '{reference}'.")
    if output_path:
        # repo.export_model(reference, output_path, target_format="some_format") # Needs export_model impl.
        print(f"Placeholder: Would export model from '{reference}' to '{output_path}'. (Export not implemented)")
    print(f"Switched to '{reference}'. Future commands will operate on this reference.")

@repo_app.command("log")
def repo_log(
    reference: Annotated[Optional[str], typer.Argument(help="Branch, tag, or snapshot ID to show history for. Defaults to current.")] = None,
    limit: Annotated[Optional[int], typer.Option("-n", "--max-count", help="Limit number of commits to show.")] = None,
):
    """Shows the commit history."""
    repo = get_repo_instance()
    repo.log(reference=reference or repo.current_branch, limit=limit)

@repo_app.command("status")
def repo_status():
    """Show the working tree status."""
    repo = get_repo_instance()
    repo.status()

# --- `paramlake branch ...` commands ---
@branch_app.command("create")
def branch_create(
    branch_name: Annotated[str, typer.Argument(help="Name of the new branch.")],
    from_reference: Annotated[Optional[str], typer.Argument(help="Reference (snapshot ID, branch, tag) to base the new branch on. Defaults to current HEAD.")] = None
):
    """Creates a new branch."""
    repo = get_repo_instance()
    repo.create_branch(branch_name, from_reference=from_reference or repo.current_branch)

@branch_app.command("list")
def branch_list():
    """Lists all branches."""
    repo = get_repo_instance()
    branches = repo.list_branches()
    print("Branches:")
    for branch in branches:
        prefix = "* " if branch == repo.current_branch else "  "
        print(f"{prefix}{branch}")

@branch_app.command("delete")
def branch_delete(branch_name: Annotated[str, typer.Argument(help="Name of the branch to delete.")]):
    """Deletes a branch."""
    repo = get_repo_instance()
    repo.delete_branch(branch_name)

# --- `paramlake tag ...` commands ---
@tag_app.command("create")
def tag_create(
    tag_name: Annotated[str, typer.Argument(help="Name of the new tag.")],
    reference: Annotated[Optional[str], typer.Argument(help="Snapshot ID, branch, or tag to point the new tag to. Defaults to current HEAD.")] = None
):
    """Creates a new tag."""
    repo = get_repo_instance()
    # Repo API needs update: create_tag should resolve reference internally if it can be branch/tag name
    # For now, assume reference must be a resolved snapshot_id for the storage_manager.create_tag
    # This is a simplification for the CLI stub.
    ref_to_tag = reference or repo.current_branch # This isn't snapshot_id directly yet
    print(f"Placeholder: Tagging functionality needs reference to be resolved to snapshot ID by Repo class.") 
    print(f"Simulating: Tag '{tag_name}' would be created for reference '{ref_to_tag}'.")
    # repo.create_tag(tag_name, ref_to_tag) # This call will fail if ref_to_tag is not snapshot_id

@tag_app.command("list")
def tag_list():
    """Lists all tags."""
    repo = get_repo_instance()
    tags = repo.list_tags()
    print("Tags:")
    for name, snapshot_id in tags.items():
        print(f"  {name} -> {snapshot_id}")

@tag_app.command("delete")
def tag_delete(tag_name: Annotated[str, typer.Argument(help="Name of the tag to delete.")]):
    """Deletes a tag."""
    repo = get_repo_instance()
    repo.delete_tag(tag_name)

# Placeholder for future commands (diff, merge, import, export)
@app.command("diff", hidden=True)
def cli_diff(ref1: str, ref2: Optional[str] = None):
    print("Diff functionality is not yet implemented in the CLI.")

@app.command("merge", hidden=True)
def cli_merge(source_branch: str):
    print("Merge functionality is not yet implemented in the CLI.")

@app.command("import", hidden=True)
def cli_import(source: str, format: str):
    print("Import functionality is not yet implemented in the CLI.")

@app.command("export", hidden=True)
def cli_export(reference: str, output: str, format: str):
    print("Export functionality is not yet implemented in the CLI.")


if __name__ == "__main__":
    app() 