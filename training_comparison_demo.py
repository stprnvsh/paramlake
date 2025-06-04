#!/usr/bin/env python3
"""
ParamLake Training Comparison Demo - Advanced Version

This example demonstrates comparing the same model trained with different
strategies on different branches with periodic commits and cross-branch diffs.

Example: Compare aggressive vs conservative learning rates over 100 epochs
with commits every 25 epochs to track training evolution.
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from datetime import datetime

# Import ParamLake components
from paramlake import Repo
from paramlake.utils.config import ParamLakeConfig


def create_simple_model():
    """Create a simple neural network model for demonstration."""
    # Use explicit input layer to ensure consistent naming
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(32, activation='relu', name='dense_2'), 
        tf.keras.layers.Dense(1, activation='sigmoid', name='dense_3')
    ])
    
    return model


def generate_sample_data(num_samples=1000):
    """Generate sample data for training."""
    np.random.seed(42)  # For reproducible results
    X = np.random.randn(num_samples, 10)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int)
    return X, y


def print_model_summary(model, title="Model Summary"):
    """Print a summary of the model's current state."""
    total_params = sum([np.prod(w.shape) for w in model.get_weights()])
    weights_sum = sum([np.sum(w) for w in model.get_weights()])
    weights_norm = np.sqrt(sum([np.sum(w**2) for w in model.get_weights()]))
    print(f"\n{title}:")
    print(f"  Total parameters: {total_params}")
    print(f"  Weights sum: {weights_sum:.6f}")
    print(f"  Weights L2 norm: {weights_norm:.6f}")


def train_model_with_commits(model, X, y, repo, branch_name, learning_rate=0.001, 
                           total_epochs=100, commit_frequency=25, batch_size=32, 
                           strategy_name=""):
    """Train model with periodic commits to track progress."""
    print(f"\n🚀 Training {strategy_name} strategy for {total_epochs} epochs...")
    print(f"📝 Committing every {commit_frequency} epochs on branch '{branch_name}'")
    
    # Switch to the appropriate branch
    repo.current_branch = branch_name
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    commit_ids = []
    training_history = {'loss': [], 'accuracy': []}
    
    # Train in chunks and commit periodically
    for epoch_start in range(0, total_epochs, commit_frequency):
        epoch_end = min(epoch_start + commit_frequency, total_epochs)
        epochs_in_chunk = epoch_end - epoch_start
        
        print(f"  📈 Training epochs {epoch_start + 1}-{epoch_end}...")
        
        # Train for this chunk
        history = model.fit(X, y, epochs=epochs_in_chunk, batch_size=batch_size, 
                          verbose=0, initial_epoch=0)
        
        # Track overall history
        training_history['loss'].extend(history.history['loss'])
        training_history['accuracy'].extend(history.history['accuracy'])
        
        # Get current performance
        current_loss = history.history['loss'][-1]
        current_acc = history.history['accuracy'][-1]
        
        print(f"    ✓ Epoch {epoch_end}: Loss={current_loss:.4f}, Accuracy={current_acc:.4f}")
        
        # Commit the model state
        commit_message = f"{strategy_name} training - epoch {epoch_end}/{total_epochs}"
        commit_id = repo.commit(model, commit_message)
        commit_ids.append((epoch_end, commit_id))
        
        print(f"    💾 Committed: {commit_id}")
    
    final_loss = training_history['loss'][-1]
    final_acc = training_history['accuracy'][-1]
    
    print(f"✅ {strategy_name} training completed!")
    print(f"   Final: Loss={final_loss:.4f}, Accuracy={final_acc:.4f}")
    
    return commit_ids, training_history, final_loss, final_acc


def compare_model_weights(params1, params2, title="Weight Comparison"):
    """Compare weights between two parameter sets."""
    print(f"\n{title}:")
    
    total_diff_norm = 0
    total_params = 0
    layer_diffs = {}
    
    for name in params1.keys():
        if name in params2:
            w1, w2 = params1[name], params2[name]
            if w1.shape == w2.shape:
                diff = w1 - w2
                diff_norm = np.linalg.norm(diff)
                rel_diff = diff_norm / (np.linalg.norm(w1) + 1e-8)
                
                layer_diffs[name] = {
                    'abs_diff': diff_norm,
                    'rel_diff': rel_diff,
                    'max_diff': np.max(np.abs(diff))
                }
                
                print(f"  {name}:")
                print(f"    Absolute difference norm: {diff_norm:.6f}")
                print(f"    Relative difference: {rel_diff:.6f}")
                print(f"    Max absolute difference: {np.max(np.abs(diff)):.6f}")
                
                total_diff_norm += diff_norm**2
                total_params += w1.size
            else:
                print(f"  {name}: Shape mismatch - {w1.shape} vs {w2.shape}")
        else:
            print(f"  {name}: Missing in second model")
    
    overall_diff = np.sqrt(total_diff_norm)
    print(f"\n  Overall difference norm: {overall_diff:.6f}")
    print(f"  Average per-parameter difference: {overall_diff/np.sqrt(total_params):.8f}")
    
    return overall_diff, layer_diffs


def compare_commits_across_branches(repo, aggressive_commits, conservative_commits):
    """Compare corresponding commits across branches."""
    print("\n" + "="*60)
    print("📊 CROSS-BRANCH COMMIT COMPARISON")
    print("="*60)
    
    comparison_results = []
    
    for (agg_epoch, agg_commit), (cons_epoch, cons_commit) in zip(aggressive_commits, conservative_commits):
        if agg_epoch == cons_epoch:  # Should always be true given our setup
            print(f"\n🔄 Comparing Epoch {agg_epoch} across strategies:")
            print(f"   Aggressive: {agg_commit}")
            print(f"   Conservative: {cons_commit}")
            
            # Load parameters from both commits
            agg_params = repo.storage_manager.load_parameters_from_snapshot(agg_commit)
            cons_params = repo.storage_manager.load_parameters_from_snapshot(cons_commit)
            
            # Compare them
            overall_diff, layer_diffs = compare_model_weights(
                agg_params, cons_params, 
                f"Aggressive vs Conservative at Epoch {agg_epoch}"
            )
            
            comparison_results.append({
                'epoch': agg_epoch,
                'aggressive_commit': agg_commit,
                'conservative_commit': cons_commit,
                'overall_diff': overall_diff,
                'layer_diffs': layer_diffs
            })
    
    return comparison_results


def analyze_training_evolution(comparison_results):
    """Analyze how the difference between strategies evolves over training."""
    print("\n" + "="*60)
    print("📈 TRAINING EVOLUTION ANALYSIS")
    print("="*60)
    
    epochs = [r['epoch'] for r in comparison_results]
    diffs = [r['overall_diff'] for r in comparison_results]
    
    print(f"\nDivergence between strategies over time:")
    for epoch, diff in zip(epochs, diffs):
        print(f"  Epoch {epoch:3d}: Overall difference = {diff:.6f}")
    
    # Calculate rate of divergence
    if len(diffs) > 1:
        print(f"\nDivergence trends:")
        for i in range(1, len(diffs)):
            change = diffs[i] - diffs[i-1]
            epoch_diff = epochs[i] - epochs[i-1]
            rate = change / epoch_diff
            
            if change > 0:
                trend = "📈 Diverging"
            elif change < 0:
                trend = "📉 Converging"
            else:
                trend = "➡️  Stable"
                
            print(f"  Epochs {epochs[i-1]}-{epochs[i]}: {trend} (rate: {rate:.8f}/epoch)")
    
    # Find the layer with the most divergence
    if comparison_results:
        print(f"\nLayer-wise divergence at final checkpoint (Epoch {epochs[-1]}):")
        final_layer_diffs = comparison_results[-1]['layer_diffs']
        sorted_layers = sorted(final_layer_diffs.items(), 
                             key=lambda x: x[1]['abs_diff'], reverse=True)
        
        for i, (layer_name, layer_diff) in enumerate(sorted_layers[:3]):
            print(f"  {i+1}. {layer_name}: {layer_diff['abs_diff']:.6f} "
                  f"(rel: {layer_diff['rel_diff']:.4f})")
    
    return diffs


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("ParamLake Advanced Training Comparison Demo")
    print("Multi-Epoch Training with Periodic Commits & Cross-Branch Diffs")
    print("=" * 70)
    
    # Setup
    temp_dir = tempfile.mkdtemp(prefix="paramlake_advanced_")
    print(f"Repository location: {temp_dir}")
    
    # Configure for S3 storage
    os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
    os.environ['AWS_REGION'] = 'us-east-1'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = ParamLakeConfig({
        'storage_type': 'icechunk',
        'storage_backend': 's3',
        'create_repo': True,
        'verbose': False,  # Reduce noise for cleaner output
        
        'bucket': 'paramlake',
        'prefix': f'advanced-comparison-{timestamp}',
        'region': 'us-east-1',
        'endpoint_url': 'https://s3.amazonaws.com',
        
        'icechunk': {
            'commit_frequency': 1,
            'verbose': False
        }
    })
    
    # Training configuration
    TOTAL_EPOCHS = 100
    COMMIT_FREQUENCY = 25
    
    try:
        # Initialize repository
        repo = Repo(temp_dir, config=config)
        print("✓ Initialized ParamLake repository")
        
        # Generate consistent training data
        X, y = generate_sample_data()
        print("✓ Generated training data")
        
        # ============================================================
        print("\n" + "="*60)
        print("1. Create Baseline Model")
        print("="*60)
        
        # Create and train initial baseline model
        baseline_model = create_simple_model()
        print_model_summary(baseline_model, "Initial Model (Random Weights)")
        
        # Train for a few epochs to get a reasonable starting point
        baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        baseline_model.fit(X, y, epochs=5, verbose=0)
        
        # Commit baseline
        baseline_commit = repo.commit(baseline_model, "Baseline model - initial training")
        print(f"✓ Baseline commit: {baseline_commit}")
        
        # ============================================================
        print("\n" + "="*60)
        print("2. Aggressive Training Strategy (Branch: aggressive)")
        print("="*60)
        
        # Create aggressive training branch
        repo.create_branch("aggressive", baseline_commit)
        
        # Create model copy for aggressive training
        aggressive_model = create_simple_model()
        repo.checkout(baseline_commit, aggressive_model)
        print("✓ Loaded baseline weights into aggressive model")
        
        # Train aggressively with periodic commits
        aggressive_commits, agg_history, agg_loss, agg_acc = train_model_with_commits(
            aggressive_model, X, y, repo, "aggressive",
            learning_rate=0.1,  # High learning rate
            total_epochs=TOTAL_EPOCHS,
            commit_frequency=COMMIT_FREQUENCY,
            batch_size=64,  # Large batches
            strategy_name="Aggressive"
        )
        
        # ============================================================
        print("\n" + "="*60)
        print("3. Conservative Training Strategy (Branch: conservative)")
        print("="*60)
        
        # Create conservative training branch
        repo.create_branch("conservative", baseline_commit)
        
        # Create model copy for conservative training
        conservative_model = create_simple_model()
        repo.checkout(baseline_commit, conservative_model)
        print("✓ Loaded baseline weights into conservative model")
        
        # Train conservatively with periodic commits
        conservative_commits, cons_history, cons_loss, cons_acc = train_model_with_commits(
            conservative_model, X, y, repo, "conservative",
            learning_rate=0.001,  # Low learning rate
            total_epochs=TOTAL_EPOCHS,
            commit_frequency=COMMIT_FREQUENCY,
            batch_size=16,  # Small batches
            strategy_name="Conservative"
        )
        
        # ============================================================
        print("\n" + "="*60)
        print("4. Cross-Branch Commit Analysis")
        print("="*60)
        
        # Compare corresponding commits across branches
        comparison_results = compare_commits_across_branches(
            repo, aggressive_commits, conservative_commits
        )
        
        # Analyze training evolution
        divergence_progression = analyze_training_evolution(comparison_results)
        
        # ============================================================
        print("\n" + "="*60)
        print("5. Final Results Summary")
        print("="*60)
        
        print(f"\n📋 Training Configuration:")
        print(f"  Total epochs: {TOTAL_EPOCHS}")
        print(f"  Commit frequency: Every {COMMIT_FREQUENCY} epochs")
        print(f"  Commits per branch: {len(aggressive_commits)}")
        
        print(f"\n🎯 Final Performance:")
        print(f"  Aggressive:   Loss={agg_loss:.4f}, Accuracy={agg_acc:.4f}")
        print(f"  Conservative: Loss={cons_loss:.4f}, Accuracy={cons_acc:.4f}")
        
        print(f"\n📈 Strategy Divergence:")
        print(f"  Initial divergence (Epoch {COMMIT_FREQUENCY}): {divergence_progression[0]:.6f}")
        print(f"  Final divergence (Epoch {TOTAL_EPOCHS}): {divergence_progression[-1]:.6f}")
        print(f"  Total divergence growth: {divergence_progression[-1] - divergence_progression[0]:.6f}")
        
        # Determine which strategy was more effective
        if agg_acc > cons_acc:
            better_strategy = "Aggressive"
            worse_strategy = "Conservative"
            acc_diff = agg_acc - cons_acc
        else:
            better_strategy = "Conservative"
            worse_strategy = "Aggressive"
            acc_diff = cons_acc - agg_acc
            
        print(f"\n🏆 Winner: {better_strategy} strategy")
        print(f"   Accuracy advantage: {acc_diff:.4f} ({acc_diff*100:.2f}%)")
        
        print(f"\n📂 Repository Structure:")
        print(f"  Baseline:     {baseline_commit}")
        print(f"  Aggressive:   {len(aggressive_commits)} commits")
        print(f"  Conservative: {len(conservative_commits)} commits")
        
        print(f"\n✅ Advanced training comparison completed successfully!")
        print(f"📁 Repository preserved at: {temp_dir}")
        
        # Show git-like command examples
        print(f"\n💡 Git-like Operations Demonstrated:")
        print(f"  ✓ Branch creation and switching")
        print(f"  ✓ Periodic commits during training")
        print(f"  ✓ Cross-branch parameter comparison")
        print(f"  ✓ Training evolution tracking")
        print(f"  ✓ Checkpoint-to-checkpoint diffs")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            if 'repo' in locals():
                repo.storage_manager.close()
        except:
            pass


if __name__ == "__main__":
    main() 