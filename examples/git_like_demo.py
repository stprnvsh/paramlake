#!/usr/bin/env python3
"""
ParamLake Git-like Version Control Demo

This example demonstrates the Git-like version control features of ParamLake
using Icechunk as the backend. It shows how to:

1. Initialize a repository
2. Create and manage branches 
3. Commit model states
4. Merge branches with conflict resolution
5. Rebase branches
6. Create and manage tags
7. View commit history
8. Switch between branches and snapshots
9. Perform diffs between versions

Requirements:
- Python 3.11
- TensorFlow
- Icechunk 
- ParamLake

Run with: python git_like_demo.py
"""

import os
import tempfile
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import ParamLake components
from paramlake import Repo
from paramlake.utils.config import ParamLakeConfig


def create_simple_model():
    """Create a simple neural network model for demonstration."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def generate_sample_data(num_samples=1000):
    """Generate sample data for training."""
    X = np.random.randn(num_samples, 10)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int)
    return X, y


def train_model_briefly(model, X, y, epochs=2):
    """Train model for a few epochs."""
    print(f"Training model for {epochs} epochs...")
    history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return history


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_model_summary(model, prefix=""):
    """Print a summary of the model's current state."""
    total_params = sum([np.prod(w.shape) for w in model.get_weights()])
    weights_sum = sum([np.sum(w) for w in model.get_weights()])
    print(f"{prefix}Model state: {total_params} parameters, weights sum: {weights_sum:.6f}")


def main():
    """Main demonstration function."""
    print("ParamLake Git-like Version Control Demo")
    print("Using Python", tf.__version__, "and TensorFlow", tf.__version__)
    
    # Create temporary directory for this demo
    temp_dir = tempfile.mkdtemp(prefix="paramlake_git_demo_")
    print(f"Demo repository location: {temp_dir}")
    
    try:
        # ============================================================
        print_section_header("1. Repository Initialization with Icechunk")
        # ============================================================
        
        print("Icechunk Setup Options:")
        print("  Local Storage: Uses local filesystem for development/testing")
        print("  S3 Storage: Uses AWS S3 for production cloud-native storage")
        print("  GCS Storage: Uses Google Cloud Storage")
        print("  Azure Storage: Uses Azure Blob Storage")
        print()
        
        # Choose storage backend - Change USE_S3 to True to demo S3 setup
        USE_S3 = True  # Set to True to use S3 instead of local storage
        
        if USE_S3:
            print("🌐 Configuring Icechunk with S3 backend...")
            
            # Set AWS credentials as environment variables for Icechunk
            os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
            os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
            os.environ['AWS_REGION'] = 'us-east-1'
            
            # S3 Configuration for Icechunk
            # Using the provided bucket from ARN: arn:aws:s3:::paramlake
            # Add timestamp to prefix to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            config = ParamLakeConfig({
                'storage_type': 'icechunk',
                'storage_backend': 's3',
                'create_repo': True,
                'verbose': True,  # Enable verbose logging
                
                # S3-specific configuration using provided credentials
                'bucket': 'paramlake',  # From ARN: arn:aws:s3:::paramlake
                'prefix': f'git-demo-experiments-{timestamp}',  # Unique S3 prefix for this demo run
                'region': 'us-east-1',  # AWS region
                'endpoint_url': 'https://s3.amazonaws.com',  # Standard S3 endpoint
                
                # Icechunk-specific settings optimized for S3
                'icechunk': {
                    'commit_frequency': 1,  # Commit after each training session
                    'auto_create_branches': True,
                    'compression': {
                        'algorithm': 'zstd',  # Better compression for cloud storage
                        'level': 6  # Higher compression for bandwidth savings
                    }
                },
                
                # Advanced S3 performance settings
                'storage_settings': {
                    'concurrency': {
                        'max_concurrent_requests': 20,
                        'ideal_request_size': 5000000  # 5MB chunks for S3 optimization
                    },
                    'storage_class': 'STANDARD',  # Standard S3 storage class
                },
                
                # Enhanced caching for cloud storage
                'caching': {
                    'chunk_cache_size': 200000000,  # 200MB cache for S3
                    'metadata_cache_size': 20000000   # 20MB metadata cache
                }
            })
            
            print("  ✓ S3 bucket: paramlake")
            print(f"  ✓ Prefix: git-demo-experiments-{timestamp}") 
            print("  ✓ Region: us-east-1")
            print("  ✓ Compression: zstd level 6")
            print("  ✓ Storage class: STANDARD")
            print("  ✓ AWS credentials configured")
            print("  📡 Cloud-native Git-like version control enabled!")
            
        else:
            print("💻 Configuring Icechunk with local filesystem backend...")
            # Local filesystem configuration for Icechunk
            config = ParamLakeConfig({
                'output_path': temp_dir,
                'storage_type': 'icechunk',
                'storage_backend': 'local',
                'create_repo': True,
                'verbose': True,  # Enable verbose logging to debug shape issues
                
                # Icechunk-specific settings
                'icechunk': {
                    'commit_frequency': 1,  # Commit after each training session
                    'auto_create_branches': True,
                    'verbose': True  # Enable verbose logging for Icechunk operations
                },
                
                # Local storage optimization
                'compression': {
                    'algorithm': 'zstd',  # Fast compression for local development
                    'level': 3  # Balanced speed/size
                },
                
                # Local caching settings
                'caching': {
                    'chunk_cache_size': 100000000,  # 100MB cache
                    'metadata_cache_size': 10000000   # 10MB metadata cache
                }
            })
            
            print(f"  ✓ Local path: {temp_dir}")
            print("  ✓ Compression: zstd level 3")
            print("  ✓ Chunk cache: 100MB")
            print("  ✓ Ready for local development")
        
        print("\nIcechunk Backend Architecture:")
        print("  • Repository: Git-like versioned storage")
        print("  • Sessions: Writable/readonly access to branches")
        print("  • Snapshots: Immutable model state commits")
        print("  • Branches: Parallel development streams")
        print("  • Tags: Named version references")
        print("  • Zarr Integration: Efficient array storage")
        
        # Initialize repository
        repo = Repo(temp_dir, config=config)
        print(f"\n✓ Initialized ParamLake repository with Icechunk backend")
        print(f"✓ Current branch: {repo.current_branch}")
        print(f"✓ Storage backend: {config.get('storage_backend', 'local')}")
        
        # Display repository configuration details
        if hasattr(repo.storage_manager, 'repo'):
            icechunk_repo = repo.storage_manager.repo
            print("✓ Git-like features enabled:")
            print("    - Branching and merging")
            print("    - Commit history and ancestry")  
            print("    - Tag management")
            print("    - Conflict detection and resolution")
            print("    - State recovery and time travel")
        
        print("\nHow Icechunk Works:")
        print("  1. Each commit creates an immutable snapshot")
        print("  2. Branches track different development paths")  
        print("  3. Merges combine changes from multiple branches")
        print("  4. Storage is optimized for cloud-native access")
        print("  5. Zarr arrays provide efficient ML data formats")
        
        if USE_S3:
            print("\n🌐 S3 Cloud Storage Benefits:")
            print("  • Scalable: Handles models of any size")
            print("  • Collaborative: Share models across teams")
            print("  • Durable: 99.999999999% (11 9's) data durability")
            print("  • Versioned: Complete model evolution history")
            print("  • Efficient: Only stores differences between versions")
        
        # ============================================================
        print_section_header("2. Initial Model Training and First Commit")
        # ============================================================
        
        # Create and train initial model
        model_v1 = create_simple_model()
        X, y = generate_sample_data()
        
        print("Initial model:")
        print_model_summary(model_v1, "  ")
        
        # Train the model
        train_model_briefly(model_v1, X, y, epochs=3)
        print("After training:")
        print_model_summary(model_v1, "  ")
        
        # Make first commit
        commit_1 = repo.commit(
            model_v1, 
            "Initial model training - baseline version",
            author="demo_user@example.com"
        )
        print(f"✓ First commit: {commit_1}")
        
        # ============================================================
        print_section_header("3. Branch Creation and Development")
        # ============================================================
        
        # Create a feature branch for experimental training
        try:
            repo.create_branch("feature_advanced_training")
            print("✓ Created branch: feature_advanced_training")
        except ValueError as e:
            if "update conflict" in str(e):
                print("Branch feature_advanced_training already exists, continuing...")
            else:
                raise
        
        # List all branches
        branches = repo.list_branches()
        print(f"Available branches: {branches}")
        
        # Switch to feature branch
        repo.current_branch = "feature_advanced_training"
        print(f"✓ Switched to branch: {repo.current_branch}")
        
        # Continue training on feature branch
        train_model_briefly(model_v1, X, y, epochs=5)
        print("After additional training on feature branch:")
        print_model_summary(model_v1, "  ")
        
        # Commit changes on feature branch
        commit_2 = repo.commit(
            model_v1,
            "Advanced training - increased epochs and refinements",
            author="demo_user@example.com"
        )
        print(f"✓ Feature branch commit: {commit_2}")
        
        # ============================================================
        print_section_header("4. Parallel Development on Main Branch")
        # ============================================================
        
        # Switch back to main for parallel development
        repo.current_branch = "main"
        print(f"✓ Switched back to: {repo.current_branch}")
        
        # Load the original state and make different changes
        repo.checkout(commit_1, model_v1)
        print("✓ Checked out original commit on main branch")
        print_model_summary(model_v1, "  Current main branch state: ")
        
        # Make different modifications on main
        # Change the model architecture slightly by adjusting learning rate
        model_v1.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        train_model_briefly(model_v1, X, y, epochs=4)
        print("After different training approach on main:")
        print_model_summary(model_v1, "  ")
        
        # Commit parallel changes
        commit_3 = repo.commit(
            model_v1,
            "Main branch: optimized learning rate and training",
            author="demo_user@example.com"
        )
        print(f"✓ Main branch commit: {commit_3}")
        
        # ============================================================
        print_section_header("5. Tagging Important Versions")
        # ============================================================
        
        # Create tags for important milestones
        repo.create_tag("v1.0-baseline", commit_1)
        repo.create_tag("v1.1-optimized", commit_3)
        print("✓ Created tags: v1.0-baseline, v1.1-optimized")
        
        # List all tags
        tags = repo.list_tags()
        print("Available tags:")
        for tag, snapshot in tags.items():
            print(f"  {tag} -> {snapshot}")
        
        # ============================================================
        print_section_header("6. Viewing Commit History")
        # ============================================================
        
        # Show history for main branch
        print("Commit history for main branch:")
        main_history = repo.log("main", limit=10)
        
        # Show history for feature branch
        print("\nCommit history for feature branch:")
        feature_history = repo.log("feature_advanced_training", limit=10)
        
        # ============================================================
        print_section_header("7. Branch Merging with Conflict Resolution")
        # ============================================================
        
        # Attempt to merge feature branch into main
        print("Attempting to merge feature_advanced_training into main...")
        
        try:
            # Check for conflicts first
            conflicts = repo.storage_manager.get_conflicts("main", "feature_advanced_training")
            if conflicts:
                print(f"⚠️  Detected {len(conflicts)} potential conflicts:")
                for conflict in conflicts[:3]:  # Show first 3 conflicts
                    print(f"  - {conflict.get('layer', 'Unknown layer')}: {conflict.get('type', 'Parameter conflict')}")
                
                # Perform merge with auto-resolution strategy
                print("Performing merge with automatic conflict resolution...")
                merge_result = repo.storage_manager.merge_branches(
                    "feature_advanced_training", 
                    "main",
                    strategy="auto",
                    commit_message="Merge feature_advanced_training: Auto-resolved conflicts"
                )
                print(f"✓ Merge completed: {merge_result}")
            else:
                print("No conflicts detected - performing clean merge...")
                merge_result = repo.storage_manager.merge_branches(
                    "feature_advanced_training", 
                    "main",
                    strategy="fast-forward",
                    commit_message="Merge feature_advanced_training: Fast-forward merge"
                )
                print(f"✓ Clean merge completed: {merge_result}")
                
        except Exception as e:
            print(f"Merge failed: {e}")
            print("Attempting rebase strategy instead...")
            
            # Try rebase as alternative
            try:
                rebase_result = repo.storage_manager.rebase_branch(
                    "feature_advanced_training",
                    "main",
                    conflict_strategy="auto"
                )
                print(f"✓ Rebase completed: {rebase_result}")
            except Exception as e2:
                print(f"Rebase also failed: {e2}")
                print("Manual intervention would be required in a real scenario")
        
        # ============================================================
        print_section_header("8. Advanced Git-like Operations")
        # ============================================================
        
        # Create another development branch
        try:
            repo.create_branch("experimental_new_architecture", commit_1)
            print("✓ Created experimental branch from baseline")
        except ValueError as e:
            if "update conflict" in str(e):
                print("Branch experimental_new_architecture already exists, continuing...")
            else:
                raise
        
        # Switch to experimental branch
        repo.storage_manager.switch_branch("experimental_new_architecture")
        print("✓ Switched to experimental branch")
        
        # Create a significantly different model architecture
        model_experimental = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model_experimental.compile(
            optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        train_model_briefly(model_experimental, X, y, epochs=3)
        print("Experimental model architecture:")
        print_model_summary(model_experimental, "  ")
        
        # Commit experimental changes
        commit_exp = repo.commit(
            model_experimental,
            "Experimental: New architecture with dropout layers",
            author="experimental_dev@example.com"
        )
        print(f"✓ Experimental commit: {commit_exp}")
        
        # ============================================================
        print_section_header("9. Diff and Comparison Operations")
        # ============================================================
        
        # Compare different versions
        print("Performing diff between baseline and current experimental version...")
        try:
            diff_result = repo.storage_manager.diff_snapshots(commit_1, commit_exp)
            print("Diff summary:")
            if 'summary' in diff_result:
                summary = diff_result['summary']
                print(f"  - Layers changed: {summary.get('changed_layers', 0)}")
                print(f"  - Parameters added: {summary.get('added_parameters', 0)}")
                print(f"  - Parameters removed: {summary.get('removed_parameters', 0)}")
                print(f"  - Total parameter changes: {summary.get('total_changes', 0)}")
            
            if 'layer_changes' in diff_result:
                print(f"  - Detailed changes available for {len(diff_result['layer_changes'])} layers")
                
        except Exception as e:
            print(f"Diff operation not fully implemented: {e}")
        
        # ============================================================
        print_section_header("10. Repository Status and Cleanup")
        # ============================================================
        
        # Show final repository status
        print("Final repository status:")
        repo.status()
        
        # List all branches and their latest commits
        print("\nFinal branch summary:")
        all_branches = repo.list_branches()
        for branch in all_branches:
            try:
                history = repo.log(branch, limit=1)
                if history:
                    latest = history[0]
                    print(f"  {branch}: {latest.get('id', 'Unknown')[:12]} - {latest.get('message', 'No message')}")
                else:
                    print(f"  {branch}: No commits")
            except:
                print(f"  {branch}: Unable to retrieve history")
        
        # Show all tags
        print(f"\nTags created: {list(repo.list_tags().keys())}")
        
        # ============================================================
        print_section_header("11. Demonstrating Checkout and State Recovery")
        # ============================================================
        
        # Create a fresh model and load different states
        recovery_model = create_simple_model()
        
        print("Demonstrating state recovery:")
        
        # Load baseline version
        print("Loading baseline version (v1.0)...")
        repo.checkout("v1.0-baseline", recovery_model)
        print_model_summary(recovery_model, "  Baseline state: ")
        
        # Load optimized version  
        print("Loading optimized version (v1.1)...")
        repo.checkout("v1.1-optimized", recovery_model)
        print_model_summary(recovery_model, "  Optimized state: ")
        
        # Load experimental version
        print("Loading experimental version...")
        try:
            repo.checkout(commit_exp, recovery_model)
            print_model_summary(recovery_model, "  Experimental state: ")
        except Exception as e:
            print(f"  Note: Experimental version has different architecture: {e}")
        
        print("\n✓ Git-like version control demonstration completed successfully!")
        print(f"Repository preserved at: {temp_dir}")
        print("\nThis demo showcased:")
        print("  • Repository initialization with Icechunk")
        print("  • Branch creation and management") 
        print("  • Model state commits with messages")
        print("  • Parallel development workflows")
        print("  • Merge and rebase operations")
        print("  • Tagging and versioning")
        print("  • Commit history tracking")
        print("  • State recovery and checkout")
        print("  • Diff and comparison operations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup (optional - comment out to preserve the repo for inspection)
        try:
            if repo and hasattr(repo, 'storage_manager'):
                repo.storage_manager.close()
        except:
            pass
            
        # Uncomment the next line to automatically clean up the demo repository
        # shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nDemo repository location: {temp_dir}")
        print("(Repository preserved for inspection)")


if __name__ == "__main__":
    main() 