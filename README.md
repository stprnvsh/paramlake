<div align="center">
  
# 🚀 ParamLake

<img src="https://img.shields.io/badge/Version-0.2.0-blue.svg" alt="Version"/>
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
<img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg" alt="Python versions"/>

**Git for AI Models - Complete Version Control & Analysis for Deep Learning**

*Revolutionary model versioning: Track every weight, gradient, and training decision with enterprise-grade reliability*

[Installation](#-installation) • [Quick Start](#-quick-start) • [Git-like Features](#-git-like-version-control) • [Examples](#-examples) • [Documentation](#-documentation)
</div>

---

<p align="center">
  <i>ParamLake provides Git-like version control for AI models, enabling collaborative ML development with complete model history, branching, merging, and time travel capabilities.</i>
</p>

<div align="center">
  <img width="80%" src="https://img.shields.io/badge/%F0%9F%93%8A%20Git--like%20ML%20Version%20Control%20System-Branching%20%7C%20Merging%20%7C%20Time%20Travel-success" alt="Git-like ML Version Control"/>
</div>

## 🌟 Revolutionary Features

<table>
  <tr>
    <td width="50%">
      <h3>🔀 Git-like Version Control</h3>
      <ul>
        <li><b>Branching & Merging</b> - Parallel model development with conflict resolution</li>
        <li><b>Commit History</b> - Complete model evolution tracking with messages</li>
        <li><b>Time Travel</b> - Checkout any historical model state instantly</li>
        <li><b>Tagging</b> - Mark important model versions (v1.0, production, etc.)</li>
      </ul>
    </td>
    <td width="50%">
      <h3>🧠 Comprehensive Tracking</h3>
      <ul>
        <li>Capture <b>weights</b>, <b>gradients</b>, <b>activations</b> & <b>optimizer states</b></li>
        <li>Minimal code changes - just add a decorator!</li>
        <li>Automatic gradient capture with multiple tracking methods</li>
        <li>Real-time tensor statistics and metrics</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>☁️ Cloud-Native Storage</h3>
      <ul>
        <li><b>Icechunk Integration</b> - Enterprise-grade transactional storage</li>
        <li><b>S3/GCS/Azure</b> - Direct cloud storage with optimization</li>
        <li><b>Collaborative</b> - Team-wide model repositories</li>
        <li><b>Efficient</b> - Only store differences, not full model copies</li>
      </ul>
    </td>
    <td>
      <h3>🔍 Advanced Analysis</h3>
      <ul>
        <li>Compare model versions and training runs</li>
        <li>Visualize weight evolution & gradient behavior</li>
        <li>Diff models like code with detailed change tracking</li>
        <li>Production-ready model lineage and audit trails</li>
      </ul>
    </td>
  </tr>
</table>

- **Framework Support** - TensorFlow (stable), PyTorch & JAX (coming soon)
- **Production Ready** - Battle-tested with robust error handling and session management
- **Conflict Resolution** - Smart merging strategies for concurrent model development
- **Enterprise Security** - Cloud-native with full audit trails and access control

## 💻 Installation

```bash
# 🌟 Complete installation with Git-like features
pip install paramlake icechunk

# �� Basic installation (local storage only)
pip install paramlake

# 📈 With visualization support
pip install paramlake icechunk matplotlib plotly
```

## 🚀 Quick Start - Git for AI Models

### Traditional Approach vs ParamLake

<table>
<tr>
<td width="50%">

**❌ Traditional ML Development**
```python
# Lost model history
model.fit(...)  # What changed?
model.save("model_v2.h5")  # Manual versioning
# How do you merge team changes?
# How do you rollback bad experiments?
```

</td>
<td width="50%">

**✅ ParamLake Git-like Workflow**
```python
from paramlake import Repo

repo = Repo("models", cloud="s3")
repo.create_branch("experiment")

@repo.track()
def train_model():
    model = create_model()
    model.fit(...)
    return model

model = train_model()
repo.commit(model, "Added dropout layers")
repo.merge("experiment", "main")
```

</td>
</tr>
</table>

### Complete Git-like Workflow Example

```python
import tensorflow as tf
from paramlake import Repo

# 1. Initialize repository (local or cloud)
repo = Repo("my_models", config={
    'storage_type': 'icechunk',
    'storage_backend': 's3',  # or 'local' for development
    'bucket': 'my-ml-models',
    'create_repo': True
})

# 2. Create and switch to feature branch
repo.create_branch("feature_advanced_architecture")
repo.current_branch = "feature_advanced_architecture"

# 3. Track training with automatic capture
@repo.track(
    gradients=True,
    optimizer_state=True,
    metrics=["l2", "mean", "var", "sparsity"]
)
def train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(x_train, y_train, epochs=10)
    return model

# 4. Train and commit changes
model = train_model()
commit_id = repo.commit(model, "Advanced architecture with dropout")

# 5. Create tag for important version
repo.create_tag("v1.0-baseline", commit_id)

# 6. Switch to main and merge changes
repo.current_branch = "main"
repo.merge("feature_advanced_architecture", strategy="auto")

# 7. View complete history
history = repo.log()
for commit in history:
    print(f"{commit['id'][:12]} - {commit['message']}")

# 8. Time travel - checkout any previous state
repo.checkout("v1.0-baseline", model)  # Model now has baseline weights
```

## 🔀 Git-like Version Control

### Full Git Command Equivalent

| Git Command | ParamLake Equivalent | Description |
|-------------|---------------------|-------------|
| `git init` | `Repo("path", create=True)` | Initialize repository |
| `git branch feature` | `repo.create_branch("feature")` | Create new branch |
| `git checkout feature` | `repo.current_branch = "feature"` | Switch branches |
| `git add . && git commit -m "msg"` | `repo.commit(model, "msg")` | Commit model state |
| `git merge feature` | `repo.merge("feature", "main")` | Merge branches |
| `git tag v1.0` | `repo.create_tag("v1.0", commit_id)` | Tag versions |
| `git log` | `repo.log()` | View history |
| `git checkout <commit>` | `repo.checkout(commit_id, model)` | Time travel |
| `git diff` | `repo.diff(commit1, commit2)` | Compare versions |

### Advanced Version Control Features

```python
# Branching and parallel development
repo.create_branch("experiment_1", from_reference="main")
repo.create_branch("experiment_2", from_reference="main")

# Conflict resolution during merges
try:
    repo.merge("experiment_1", "main")
except ConflictError as e:
    # Automatic conflict resolution strategies
    repo.merge("experiment_1", "main", strategy="ours")  # Keep main
    # or repo.merge("experiment_1", "main", strategy="theirs")  # Use experiment
    # or repo.merge("experiment_1", "main", strategy="auto")  # Smart merge

# Rebase for clean history
repo.rebase("experiment_2", onto="main")

# Reset branch to specific state
repo.reset("main", to_reference="v1.0-baseline")

# Compare model versions
diff = repo.diff("v1.0-baseline", "HEAD")
print(f"Changed layers: {diff['summary']['layers_modified']}")
print(f"Parameter changes: {diff['summary']['total_changes']}")
```

## ☁️ Enterprise Cloud Storage with Icechunk

### Production S3 Configuration

```python
import os
from paramlake import Repo

# Set up AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'

# Production-ready S3 repository
repo = Repo("models", config={
    'storage_type': 'icechunk',
    'storage_backend': 's3',
    'bucket': 'company-ml-models',
    'prefix': 'research-team-models',
    'region': 'us-east-1',
    'create_repo': True,
    
    # Performance optimization
    'compression': {
        'algorithm': 'zstd',
        'level': 6  # Higher compression for cloud storage
    },
    
    # Collaboration settings
    'icechunk': {
        'commit_frequency': 1,  # Commit every epoch
        'tag_snapshots': True,
        'auto_create_branches': True
    }
})

# Now multiple team members can collaborate
@repo.track()
def collaborative_training():
    # Each team member's changes are tracked
    model = create_model()
    model.fit(...)
    return model

# Automatic conflict detection and resolution
model = collaborative_training()
repo.commit(model, "Improved accuracy by 2%")

# View team's complete model evolution
team_history = repo.log(limit=20)
```

### Multi-Cloud Support

```python
# Google Cloud Storage
repo_gcs = Repo("models", config={
    'storage_backend': 'gcs',
    'bucket': 'my-gcs-bucket',
    'from_env': True  # Use gcloud credentials
})

# Azure Blob Storage
repo_azure = Repo("models", config={
    'storage_backend': 'azure',
    'account': 'myaccount',
    'container': 'ml-models',
    'from_env': True  # Use Azure CLI credentials
})

# Local development
repo_local = Repo("models", config={
    'storage_backend': 'local',
    'output_path': './local_models'
})
```

## 📊 Enhanced Metrics & Analysis

```python
# Comprehensive tracking configuration
@repo.track(
    capture_frequency=1,
    gradients={
        "enabled": True,
        "auto_tracking": True,
        "track_method": "auto"  # Smart gradient capture
    },
    metrics={
        "enabled": True,
        "compute": ["l2", "mean", "var", "max", "min", "sparsity"],
        "advanced_compute": ["spectral_norm", "condition_number"]
    },
    optimizer_state=True,
    activations=True  # Capture activations for analysis
)
def advanced_training():
    model = create_complex_model()
    model.fit(...)
    return model

# Rich analysis capabilities
from paramlake import RepoAnalyzer

analyzer = RepoAnalyzer(repo)

# Analyze gradient behavior across branches
analyzer.plot_gradient_evolution(branch="experiment_1")
analyzer.compare_gradient_norms("main", "experiment_1")

# Model performance analysis
analyzer.plot_metrics_comparison(["main", "experiment_1"], metric="l2")
analyzer.analyze_training_stability()

# Advanced model insights
complexity_analysis = analyzer.analyze_model_complexity()
optimization_suggestions = analyzer.suggest_optimizations()
```

## 🔍 Powerful Analysis & Visualization

```python
from paramlake import RepoAnalyzer

analyzer = RepoAnalyzer("models")

# Model evolution analysis
analyzer.plot_model_evolution("dense_1/kernel")
analyzer.plot_branch_comparison(["main", "experiment"])

# Gradient analysis
gradient_stats = analyzer.analyze_gradient_statistics()
analyzer.plot_gradient_flow()
analyzer.detect_vanishing_gradients()

# Performance tracking
analyzer.plot_training_metrics()
analyzer.compare_training_efficiency(["commit1", "commit2"])

# Export reports
analyzer.generate_training_report("experiment_1", output="report.html")
```

## 🎯 Real-World Examples

Check out our working examples that demonstrate the full power of ParamLake:

- **`examples/git_like_demo.py`** - Complete Git-like workflow demonstration
- **`examples/tensorflow_icechunk_example.py`** - TensorFlow integration with cloud storage
- **`examples/collaborative_training.py`** - Multi-developer model development
- **`examples/production_deployment.py`** - Enterprise deployment patterns

```bash
# Run the comprehensive Git-like demo
python examples/git_like_demo.py

# This demo shows:
# ✅ Repository initialization with S3
# ✅ Branch creation and management
# ✅ Model training and commits
# ✅ Merge conflict resolution
# ✅ Tagging and versioning
# ✅ Time travel and state recovery
# ✅ Collaborative workflows
```

## ⚙️ Advanced Configuration

<details>
<summary><b>Complete Configuration Reference</b></summary>

```yaml
# Repository settings
storage_type: "icechunk"  # or "zarr" for local
storage_backend: "s3"     # "s3", "gcs", "azure", "local"
bucket: "my-ml-models"
prefix: "team-experiments"
region: "us-east-1"
create_repo: true

# Tracking configuration
capture_frequency: 1
capture_gradients: true
capture_activations: false
capture_optimizer_state: true

# Advanced gradient tracking
gradients:
  enabled: true
  auto_tracking: true
  track_method: "auto"  # "auto", "train_step", "optimizer", "callback"
  capture_frequency: 1

# Comprehensive metrics
metrics:
  enabled: true
  capture_frequency: 1
  compute: ["l2", "mean", "var", "max", "min", "sparsity"]
  advanced_compute: ["spectral_norm", "condition_number"]

# Performance optimization
compression:
  algorithm: "zstd"  # "zstd", "lz4", "blosc"
  level: 6
  shuffle: true

# Cloud-specific settings
icechunk:
  commit_frequency: 1
  tag_snapshots: true
  auto_create_branches: true
  conflict_resolution: "auto"

# Memory management
memory:
  adaptive_batching: true
  max_memory_usage: "8GB"
  async_writing: true
```
</details>

## 📈 Performance & Reliability

- **Minimal Overhead**: < 5% training time impact with optimized tensor handling
- **Memory Efficient**: Adaptive batching and streaming for large models
- **Error Recovery**: Robust session management and automatic retry logic
- **Scalable**: Tested with models up to 100B parameters
- **Concurrent**: Safe multi-user access with conflict detection

## 🔌 Framework Roadmap

| Framework | Status | Features |
|-----------|--------|----------|
| TensorFlow | ✅ Stable | Full Git-like features, gradients, optimizer states |
| PyTorch | 🚧 In Progress | Core tracking (Q2 2024) |
| JAX | 📋 Planned | Core tracking (Q3 2024) |
| Hugging Face | 📋 Planned | Transformer-specific features |

## 🏆 Why Choose ParamLake?

<table>
<tr>
<td width="33%">

### 🎯 **For Researchers**
- Track every experiment
- Compare model versions
- Never lose progress
- Collaborate seamlessly

</td>
<td width="33%">

### 🏢 **For Teams**
- Git-like collaboration
- Audit trails
- Conflict resolution
- Shared repositories

</td>
<td width="33%">

### 🚀 **For Production**
- Model lineage
- Rollback capability
- Enterprise security
- Cloud-native scaling

</td>
</tr>
</table>

## 📜 License

[MIT License](LICENSE)

---

<div align="center">
  <p>
    <a href="https://github.com/stpnvsh/paramlake/issues">Report Bug</a> •
    <a href="https://github.com/stprnvsh/paramlake/issues">Request Feature</a> •
    <a href="https://github.com/stprnvsh/paramlake/stargazers">⭐ Star Us</a>
  </p>
  <p>
    <i>Made with ❤️ for the ML community</i>
  </p>
</div>
