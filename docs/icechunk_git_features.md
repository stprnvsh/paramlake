# Icechunk Git-like Features in ParamLake

This document describes the Git-like features implemented in ParamLake when using Icechunk as the storage backend.

## Overview

ParamLake with Icechunk now supports Git-like version control operations for ML model training, enabling:
- Branch-based experimentation
- Model state commits
- Snapshot diffing and comparison
- Tag management
- History tracking
- Merge operations (simplified)

## Implemented Features

### 1. Branch Operations

#### Create Branch
```python
storage_manager.create_branch("experiment-1", from_reference="main")
```

#### Switch Branch
```python
result = storage_manager.switch_branch("experiment-1", create_if_missing=True)
# Returns: {"status": "success", "branch": "experiment-1", "snapshot": "abc123...", "message": "Switched to branch 'experiment-1'"}
```

#### List Branches
```python
branches = storage_manager.list_branches()
# Returns: ["main", "experiment-1", "experiment-2"]
```

#### Delete Branch
```python
storage_manager.delete_branch("experiment-1")
```

### 2. Commit Operations

#### Commit Model State
```python
snapshot_id = storage_manager.commit_model_state(
    model_parameters={"layer1_weights": np.array(...), "layer2_weights": np.array(...)},
    message="Updated learning rate to 0.01",
    branch_name="experiment-1",
    author="john.doe@example.com",
    additional_metadata={"learning_rate": 0.01, "batch_size": 32}
)
```

### 3. Tag Management

#### Create Tag
```python
storage_manager.create_tag("v1.0", reference="main", message="First stable version")
```

#### List Tags
```python
tags = storage_manager.list_tags()
# Returns: {"v1.0": "snapshot_id_123", "best-accuracy": "snapshot_id_456"}
```

#### Delete Tag
```python
storage_manager.delete_tag("v1.0")
```

### 4. History and Diff

#### Get History
```python
history = storage_manager.get_history(reference="main", limit=10)
# Returns list of commits with id, message, timestamp, author, and tags
```

#### Diff Snapshots
```python
diff = storage_manager.diff_snapshots("main", "experiment-1")
# Returns detailed diff with metadata changes, layer changes, and summary
```

### 5. Checkout Operations

#### Checkout Snapshot
```python
result = storage_manager.checkout_snapshot(
    reference="snapshot_abc123",
    new_branch="from-checkpoint"
)
# Allows continuing training from any historical point
```

### 6. Merge Operations

#### Merge Branches
```python
# Simple merge strategies
snapshot_id = storage_manager.merge_branches(
    source_branch="experiment-1",
    target_branch="main",
    strategy="theirs",  # or "ours", "manual"
    commit_message="Merged experiment-1 into main"
)
```

### 7. Import/Export

#### Import Model
```python
snapshot_id = storage_manager.import_model_from_path(
    source_path="model.h5",
    source_format="hdf5",
    branch_name="imported",
    commit_message="Imported pre-trained model"
)
```

## Analyzer Features

The `IcechunkModelAnalyzer` also supports Git-like operations:

### Visual Diff
```python
analyzer = IcechunkModelAnalyzer(repo_config)
diff_output = analyzer.diff_snapshots_visual(
    reference1="main",
    reference2="experiment-1", 
    output_format="console",  # or "html", "dict"
    include_values=True
)
```

### Checkout Different Snapshot
```python
# Create analyzer for specific snapshot
analyzer_v1 = analyzer.checkout_analyzer("v1.0")
```

## Configuration

Enable Git-like features in your ParamLake configuration:

```python
config = {
    "storage_type": "icechunk",
    "git_features": {
        "enabled": True,
        "default_branch": "main",
        "tag_snapshots": True,
        "commit_frequency": 10,  # Auto-commit every 10 epochs
        "remotes": {
            "origin": {
                "url": "s3://my-bucket/ml-models",
                "type": "s3"
            }
        }
    }
}
```

## Use Cases

### 1. Experiment Tracking
```python
# Create branch for experiment
storage_manager.create_branch("lr-experiment", from_reference="main")
storage_manager.switch_branch("lr-experiment")

# Train model...

# Commit results
storage_manager.commit_changes("Experiment with LR=0.001")
```

### 2. Model Comparison
```python
# Compare two experiments
diff = storage_manager.diff_snapshots("lr-experiment", "dropout-experiment")
print(f"Total changes: {diff['summary']['total_changes']}")
```

### 3. Rollback to Previous State
```python
# Find best performing model
history = storage_manager.get_history("main", limit=50)
best_snapshot = find_best_accuracy(history)

# Create branch from best model
storage_manager.checkout_snapshot(best_snapshot, new_branch="from-best")
```

### 4. Collaborative Training
```python
# Tag important checkpoints
storage_manager.create_tag("baseline", "main")

# Create feature branch
storage_manager.create_branch("feature/new-architecture")

# After testing, merge back
storage_manager.merge_branches("feature/new-architecture", "main")
```

## Actual Limitations

1. **Session Management**: Sessions become read-only after commits, requiring new sessions for continued work (handled automatically)

2. **Merge Model**: Uses rebase-based merging rather than traditional Git three-way merges, but with sophisticated conflict resolution

3. **Remote Protocol**: No separate push/pull protocol needed - Icechunk repositories are inherently distributed when stored in cloud storage

4. **Immutable Tags**: Once deleted, tags cannot be recreated with the same name (Icechunk design choice)

## Icechunk Advantages Over Traditional Git

1. **Superior Conflict Resolution**: Chunk-level conflict detection and resolution with multiple strategies:
   - `ConflictDetector()` for automatic conflict detection
   - `BasicConflictSolver()` with 'ours', 'theirs', or custom strategies

2. **Built-in Distribution**: No need for separate local/remote repositories - all operations work directly in cloud storage

3. **Atomic Operations**: Strong consistency guarantees from cloud storage backends

4. **Native Array Versioning**: Purpose-built for multidimensional data, not retrofitted from text files

5. **Advanced Rebase**: Native rebase support with conflict resolution:
   ```python
   # Rebase with conflict detection
   storage_manager.rebase_branch("feature-branch", "main", conflict_strategy='detect')
   
   # Rebase with automatic resolution
   storage_manager.rebase_branch("feature-branch", "main", conflict_strategy='ours')
   ```

6. **Collaborative Commits**: Built-in support for collaborative workflows:
   ```python
   # Commit with automatic rebase
   snapshot_id = storage_manager.commit_with_rebase("Update model", conflict_strategy='detect')
   ```

## Enhanced Features Implemented

### Advanced Conflict Resolution
```python
# Detect conflicts between branches
conflicts = storage_manager.get_conflicts("feature-branch", "main")
for conflict in conflicts:
    print(f"Conflict in {conflict['path']}: chunks {conflict['conflicted_chunks']}")

# Merge with different strategies
storage_manager.merge_branches("feature", "main", strategy='auto')  # Detect conflicts
storage_manager.merge_branches("feature", "main", strategy='ours')   # Keep target
storage_manager.merge_branches("feature", "main", strategy='theirs') # Use source
```

### Proper Rebase Operations
```python
# Full rebase functionality
storage_manager.rebase_branch("feature-branch", "main", conflict_strategy='detect')
```

### Enhanced Branch Management
```python
# Reset branch to any reference
storage_manager.reset_branch("feature-branch", "v1.0-tag")

# Switch branches with session management
result = storage_manager.switch_branch("experiment-1", create_if_missing=True)
```

## Future Enhancements

1. **Enhanced Diff Visualization**: Rich web UI for comparing model states
2. **Hooks System**: Pre/post-commit hooks for validation and automation  
3. **Multi-Repository Workflows**: Coordinating across multiple model repositories
4. **Advanced Conflict UI**: Interactive conflict resolution tools
5. **Performance Optimizations**: Further optimize for large model operations

## Example Workflow

```python
from paramlake import create_storage_manager

# Initialize with Git features
config = {
    "storage_type": "icechunk",
    "bucket": "ml-experiments",
    "git_features": {"enabled": True}
}

storage = create_storage_manager(config)

# Main development
storage.switch_branch("main")
# ... train baseline model ...
storage.commit_changes("Baseline model")
storage.create_tag("baseline", "main")

# Experiment 1: Higher learning rate with proper conflict handling
storage.create_branch("high-lr", from_reference="baseline")
storage.switch_branch("high-lr")
# ... train with high LR ...

# Use collaborative commit with automatic rebase
try:
    snapshot_id = storage.commit_with_rebase("High LR experiment", conflict_strategy='detect')
    print(f"Successfully committed: {snapshot_id}")
except ValueError as e:
    print(f"Conflicts detected: {e}")
    # Handle conflicts or use different strategy
    snapshot_id = storage.commit_with_rebase("High LR experiment", conflict_strategy='ours')

# Experiment 2: Different architecture with conflict detection
storage.create_branch("new-arch", from_reference="baseline")
storage.switch_branch("new-arch")
# ... train with new architecture ...

# Check for conflicts before merging
conflicts = storage.get_conflicts("new-arch", "main")
if conflicts:
    print(f"Found {len(conflicts)} conflicts:")
    for conflict in conflicts:
        print(f"  - {conflict['path']}: {len(conflict['conflicted_chunks'])} chunks")
    
    # Resolve conflicts using strategy
    storage.merge_branches("new-arch", "main", strategy='auto')
else:
    storage.merge_branches("new-arch", "main", strategy='auto')

# Advanced rebasing example
try:
    # Rebase experimental branch onto latest main
    storage.rebase_branch("experimental", "main", conflict_strategy='detect')
except ValueError as e:
    print(f"Rebase conflicts: {e}")
    # Use different strategy or resolve manually
    storage.rebase_branch("experimental", "main", conflict_strategy='theirs')

# Compare results with enhanced diff
diff = storage.diff_snapshots("high-lr", "new-arch")
analyzer = IcechunkModelAnalyzer(config)
visual_diff = analyzer.diff_snapshots_visual("high-lr", "new-arch", output_format="console")
print(visual_diff)

# Reset branch if needed
storage.reset_branch("experimental", "baseline")

# Merge best experiment with conflict resolution
try:
    snapshot_id = storage.merge_branches("new-arch", "main", strategy='auto')
    storage.create_tag("v2.0", "main")
    print(f"Successfully merged and tagged v2.0: {snapshot_id}")
except ValueError as e:
    print(f"Merge conflicts: {e}")
    # Use manual resolution or different strategy
``` 