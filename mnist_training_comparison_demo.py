#!/usr/bin/env python3
"""
ParamLake MNIST Training Comparison Demo - Using @paramlake Decorator

This example demonstrates the power of the @paramlake decorator for comparing 
different training strategies on MNIST dataset with automatic parameter tracking.

Example: Compare different optimizers (Adam vs SGD) with automatic commits,
weight/gradient/optimizer tracking, and cross-branch analysis.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import ParamLake components
from paramlake import paramlake, Repo
from paramlake.utils.config import ParamLakeConfig


def create_cnn_model():
    """Create a CNN model for MNIST classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool1'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='pool2'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dropout(0.5, name='dropout'),
        tf.keras.layers.Dense(64, activation='relu', name='dense1'),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    
    return model


def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset."""
    print("📥 Loading MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Use subset for demo speed
    x_train = x_train[:10000]  # 10k samples
    y_train = y_train[:10000]
    x_test = x_test[:2000]     # 2k test samples
    y_test = y_test[:2000]
    
    print(f"✓ Training data: {x_train.shape}, Labels: {y_train.shape}")
    print(f"✓ Test data: {x_test.shape}, Labels: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def create_paramlake_config(strategy_name, total_epochs):
    """Create ParamLake configuration for a training strategy."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure for S3 storage
    os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
    os.environ['AWS_REGION'] = 'us-east-1'
    
    config = ParamLakeConfig({
        'storage_type': 'icechunk',
        'storage_backend': 's3',
        'create_repo': True,
        'verbose': False,  # Reduce output noise
        'capture_frequency': 5,  # Capture data every 5 epochs, not every epoch
        
        'bucket': 'paramlake',
        'prefix': f'mnist-{strategy_name.lower()}-{timestamp}',
        'region': 'us-east-1',
        'endpoint_url': 'https://s3.amazonaws.com',
        
        # Enable all automatic tracking
        'capture_weights': True,
        'capture_gradients': False,  # Disable gradients to reduce collection load
        'capture_optimizer_state': False,  # Disable optimizer state to reduce collection load
        'capture_activations': False,  # Skip for performance
        
        # Fix compression configuration - disable entirely to avoid codec issues
        'compression': {
            'algorithm': 'none',  # Disable compression entirely for compatibility
            'level': 0,
            'shuffle': False
        },
        
        # Automatic commits every 5 epochs
        'icechunk': {
            'commit_frequency': 10,  # Auto-commit every 10 epochs instead of 5
            'tag_snapshots': True,  # Auto-tag important snapshots
            'verbose': False  # Reduce verbosity to minimize output
        },
        
        # Git-like features
        'git_features': {
            'enabled': True,
            'default_branch': strategy_name.lower(),  # Each strategy gets its own branch
            'auto_commit': True,
            'auto_tag': True
        },
        
        # Enhanced metrics collection
        'metrics': {
            'enabled': True,
            'capture_frequency': 5,  # Every 5 epochs instead of every epoch
            'compute': ["l2", "mean"]  # Reduced metrics to avoid overwhelming storage
        }
    })
    
    return config


# Strategy 1: Adam Optimizer Training
@paramlake(
    config=create_paramlake_config("Adam", 10),
    run_id="adam_strategy"
)
def train_adam_strategy(train_data, val_data, test_data):
    """Train model with Adam optimizer using @paramlake decorator."""
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    
    print("\n🚀 Training Adam Strategy (Automatic ParamLake Tracking)")
    
    # Create and compile model
    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model compiled with Adam optimizer")
    print("✓ ParamLake will automatically track:")
    print("  - Weights every 5 epochs")
    print("  - Training metrics") 
    print("  - Auto-commits every 10 epochs")
    
    # Train model - ParamLake automatically captures everything!
    history = model.fit(
        x_train, y_train,
        epochs=10,  # Reduced from 30 to 10 for faster execution
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Adam Strategy Results:")
    print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    
    return model, history, test_acc


# Strategy 2: SGD with Momentum Training  
@paramlake(
    config=create_paramlake_config("SGD", 10),
    run_id="sgd_strategy"
)
def train_sgd_strategy(train_data, val_data, test_data):
    """Train model with SGD+Momentum using @paramlake decorator."""
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    
    print("\n🚀 Training SGD+Momentum Strategy (Automatic ParamLake Tracking)")
    
    # Create and compile model
    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model compiled with SGD+Momentum optimizer")
    print("✓ ParamLake will automatically track:")
    print("  - Weights every 5 epochs")
    print("  - Training metrics")
    print("  - Auto-commits every 10 epochs")
    
    # Train model - ParamLake automatically captures everything!
    history = model.fit(
        x_train, y_train,
        epochs=10,  # Reduced from 30 to 10 for faster execution
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Test evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ SGD+Momentum Strategy Results:")
    print(f"   Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    
    return model, history, test_acc


def analyze_repository_state():
    """Actually analyze real repository states and show commits/branches."""
    print("\n" + "="*80)
    print("🗂️  REAL REPOSITORY ANALYSIS")
    print("="*80)
    
    try:
        import icechunk
        
        # Configure for S3 access
        os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
        os.environ['AWS_REGION'] = 'us-east-1'
        
        # List all paramlake repositories in S3
        print("\n📋 SCANNING S3 FOR PARAMLAKE REPOSITORIES")
        print("-" * 50)
        
        try:
            import boto3
            s3 = boto3.client('s3')
            paginator = s3.get_paginator('list_objects_v2')
            
            repositories = []
            for page in paginator.paginate(Bucket='paramlake', Prefix='mnist-'):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.json') or 'icechunk' in key:
                            repo_prefix = key.split('/')[0]
                            if repo_prefix not in repositories:
                                repositories.append(repo_prefix)
            
            print(f"Found {len(repositories)} repositories:")
            for repo in repositories:
                print(f"  📁 {repo}")
                
        except Exception as e:
            print(f"Error scanning S3: {e}")
            
        # Try to connect to recent repositories and show their state
        print("\n📊 REPOSITORY DETAILS")
        print("-" * 50)
        
        for repo_prefix in repositories[-2:]:  # Check last 2 repositories
            print(f"\n🔍 Analyzing: {repo_prefix}")
            try:
                # Create icechunk storage and repository
                storage = icechunk.s3_storage(
                    bucket="paramlake", 
                    prefix=repo_prefix,
                    from_env=True
                )
                
                repo = icechunk.Repository.open(storage)
                
                # List branches
                print(f"  📂 Branches:")
                try:
                    branches = list(repo.list_branches())
                    for branch in branches:
                        print(f"    • {branch}")
                except Exception as e:
                    print(f"    Error listing branches: {e}")
                
                # Get commit history
                print(f"  📝 Recent Commits:")
                try:
                    # Get the latest snapshot
                    session = repo.readonly_session("main")
                    snapshot_id = session.snapshot_id
                    
                    # Get ancestry/history
                    history = repo.ancestry(snapshot_id=snapshot_id)
                    commit_count = 0
                    for ancestor in history:
                        if commit_count < 5:  # Show last 5 commits
                            print(f"    🆔 {ancestor.id}")
                            print(f"       📅 {ancestor.written_at}")
                            print(f"       💬 {ancestor.message}")
                            commit_count += 1
                        else:
                            break
                            
                except Exception as e:
                    print(f"    Error getting commits: {e}")
                    
                # Show repository stats
                print(f"  📊 Repository Stats:")
                try:
                    session = repo.readonly_session("main")
                    store = session.store
                    import zarr
                    
                    # Try to open the zarr group
                    try:
                        group = zarr.open_group(store, mode="r")
                        print(f"    📁 Groups: {list(group.keys())}")
                        
                        # Check for weights data
                        if 'weights' in group:
                            weights_group = group['weights']
                            print(f"    ⚖️  Weight snapshots: {len(list(weights_group.keys()))}")
                            
                        # Check for gradients data  
                        if 'gradients' in group:
                            gradients_group = group['gradients']
                            print(f"    📈 Gradient snapshots: {len(list(gradients_group.keys()))}")
                            
                        # Check for metrics
                        if 'metrics' in group:
                            metrics_group = group['metrics']
                            print(f"    📊 Metrics captured: {len(list(metrics_group.keys()))}")
                            
                    except Exception as e:
                        print(f"    No data groups found or error: {e}")
                        
                except Exception as e:
                    print(f"    Error accessing repository stats: {e}")
                    
            except Exception as e:
                print(f"  ❌ Error connecting to {repo_prefix}: {e}")
                
    except Exception as e:
        print(f"❌ Error in repository analysis: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ParamLake MNIST Demo - Real Repository Analysis")
    print("=" * 80)
    
    try:
        # Load data
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
        
        # Create validation split
        val_split = 0.1
        val_size = int(len(x_train) * val_split)
        x_val = x_train[:val_size]
        y_val = y_train[:val_size]
        x_train = x_train[val_size:]
        y_train = y_train[val_size:]
        
        print(f"✓ Final training set: {x_train.shape}")
        print(f"✓ Validation set: {x_val.shape}")
        
        # ============================================================
        print("\n" + "="*70)
        print("1. Training with @paramlake Decorator - Adam Strategy")
        print("="*70)
        
        # Train Adam strategy - everything automatic!
        adam_model, adam_history, adam_test_acc = train_adam_strategy(
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
        )
        
        # ============================================================
        print("\n" + "="*70)  
        print("2. Training with @paramlake Decorator - SGD Strategy")
        print("="*70)
        
        # Train SGD strategy - everything automatic!
        sgd_model, sgd_history, sgd_test_acc = train_sgd_strategy(
            (x_train, y_train), (x_val, y_val), (x_test, y_test)
        )
        
        # ============================================================
        print("\n" + "="*70)
        print("3. Results Comparison")  
        print("="*70)
        
        print(f"\n🏆 Final Results:")
        print(f"  Adam Strategy:      Test Accuracy = {adam_test_acc:.4f}")
        print(f"  SGD+Momentum:       Test Accuracy = {sgd_test_acc:.4f}")
        
        winner = "Adam" if adam_test_acc > sgd_test_acc else "SGD+Momentum"
        diff = abs(adam_test_acc - sgd_test_acc)
        print(f"  🥇 Winner: {winner} (+{diff:.4f} accuracy advantage)")
        
        # ============================================================ 
        # REAL: Repository Analysis
        analyze_repository_state()
        
        # ============================================================
        print("\n" + "="*70)
        print("✅ DEMO COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 