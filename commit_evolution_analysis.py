#!/usr/bin/env python3
"""
Commit Evolution Analysis - Compare Weight Changes Across Training History

This script analyzes how weights evolved during training by comparing weights
across different commits within each branch (Adam and SGD), showing the
training dynamics and optimization paths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import icechunk
import pandas as pd
from datetime import datetime
import matplotlib.patches as mpatches

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
os.environ['AWS_REGION'] = 'us-east-1'

def get_commit_history(repo, branch_name):
    """Get commit history for a specific branch."""
    
    print(f"📝 Getting commit history for branch: {branch_name}")
    
    try:
        # Get the latest commit on the branch
        session = repo.readonly_session(branch_name)
        snapshot_id = session.snapshot_id
        
        # Get ancestry/history
        history = repo.ancestry(snapshot_id=snapshot_id)
        
        commits = []
        for ancestor in history:
            commit_info = {
                'id': ancestor.id,
                'message': ancestor.message,
                'timestamp': ancestor.written_at,
                'branch': branch_name
            }
            commits.append(commit_info)
        
        print(f"✅ Found {len(commits)} commits in {branch_name}")
        for i, commit in enumerate(commits[:5]):  # Show first 5
            print(f"  {i+1}. {commit['id'][:12]} - {commit['message']}")
        
        return commits
    
    except Exception as e:
        print(f"❌ Error getting commit history for {branch_name}: {e}")
        return []

def load_weights_from_commit(repo, commit_id):
    """Load weights from a specific commit."""
    
    try:
        # Create a readonly session for the specific commit snapshot
        session = repo.readonly_session(snapshot_id=commit_id)
        store = session.store
        group = zarr.open_group(store, mode="r")
        
        if 'parameters' not in group:
            return {}
        
        params_group = group['parameters']
        weights = {}
        
        for param_name in params_group.keys():
            param_array = params_group[param_name]
            weights[param_name] = param_array[:]
        
        return weights
        
    except Exception as e:
        print(f"⚠️ Could not load weights from commit {commit_id[:12]}: {e}")
        return {}

def analyze_weight_evolution(repo, commits, branch_name):
    """Analyze how weights evolved across commits in a branch."""
    
    print(f"\n🔍 Analyzing weight evolution for {branch_name} branch...")
    
    evolution_data = {
        'commits': [],
        'weights_history': [],
        'statistics_history': [],
        'differences_history': []
    }
    
    previous_weights = None
    
    # Process commits in chronological order (reverse the list)
    sorted_commits = sorted(commits, key=lambda x: x['timestamp'])
    
    for i, commit in enumerate(sorted_commits):
        print(f"  📊 Processing commit {i+1}/{len(sorted_commits)}: {commit['id'][:12]}")
        
        # Load weights from this commit
        weights = load_weights_from_commit(repo, commit['id'])
        
        if not weights:
            continue
        
        # Compute statistics for this commit
        stats = compute_commit_statistics(weights, commit)
        
        # Compute differences from previous commit
        if previous_weights is not None:
            differences = compute_commit_differences(previous_weights, weights)
        else:
            differences = None
        
        evolution_data['commits'].append(commit)
        evolution_data['weights_history'].append(weights)
        evolution_data['statistics_history'].append(stats)
        evolution_data['differences_history'].append(differences)
        
        previous_weights = weights
    
    return evolution_data

def compute_commit_statistics(weights, commit):
    """Compute comprehensive statistics for weights at a specific commit."""
    
    stats = {
        'commit_id': commit['id'],
        'timestamp': commit['timestamp'],
        'message': commit['message'],
        'layer_stats': {},
        'overall_stats': {}
    }
    
    all_weights = []
    total_params = 0
    
    for layer_name, weight_array in weights.items():
        layer_stats = {
            'mean': np.mean(weight_array),
            'std': np.std(weight_array),
            'min': np.min(weight_array),
            'max': np.max(weight_array),
            'l2_norm': np.linalg.norm(weight_array),
            'sparsity': np.sum(np.abs(weight_array) < 1e-6) / weight_array.size,
            'num_params': weight_array.size
        }
        
        stats['layer_stats'][layer_name] = layer_stats
        total_params += layer_stats['num_params']
        all_weights.extend(weight_array.flatten())
    
    # Overall statistics
    all_weights = np.array(all_weights)
    stats['overall_stats'] = {
        'total_params': total_params,
        'mean': np.mean(all_weights),
        'std': np.std(all_weights),
        'min': np.min(all_weights),
        'max': np.max(all_weights),
        'l2_norm': np.linalg.norm(all_weights),
        'sparsity': np.sum(np.abs(all_weights) < 1e-6) / len(all_weights)
    }
    
    return stats

def compute_commit_differences(prev_weights, curr_weights):
    """Compute differences between consecutive commits."""
    
    differences = {}
    
    common_keys = set(prev_weights.keys()) & set(curr_weights.keys())
    
    for layer_name in common_keys:
        prev_w = prev_weights[layer_name]
        curr_w = curr_weights[layer_name]
        
        if prev_w.shape != curr_w.shape:
            continue
        
        diff = curr_w - prev_w
        abs_diff = np.abs(diff)
        
        layer_diff = {
            'raw_diff': diff,
            'abs_diff': abs_diff,
            'mean_abs_diff': np.mean(abs_diff),
            'max_abs_diff': np.max(abs_diff),
            'l2_diff': np.linalg.norm(diff),
            'relative_change': np.linalg.norm(diff) / (np.linalg.norm(prev_w) + 1e-8)
        }
        
        differences[layer_name] = layer_diff
    
    return differences

def create_evolution_plots(adam_evolution, sgd_evolution):
    """Create comprehensive plots showing weight evolution over time."""
    
    print("📊 Creating weight evolution visualizations...")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Extract data for plotting
    adam_stats = adam_evolution['statistics_history']
    sgd_stats = sgd_evolution['statistics_history']
    adam_diffs = [d for d in adam_evolution['differences_history'] if d is not None]
    sgd_diffs = [d for d in sgd_evolution['differences_history'] if d is not None]
    
    # 1. Overall Weight Statistics Evolution
    ax1 = plt.subplot(4, 3, 1)
    
    # Extract overall L2 norms over time
    adam_l2_norms = [stat['overall_stats']['l2_norm'] for stat in adam_stats]
    sgd_l2_norms = [stat['overall_stats']['l2_norm'] for stat in sgd_stats]
    
    ax1.plot(range(len(adam_l2_norms)), adam_l2_norms, 'o-', label='Adam', linewidth=2, markersize=6)
    ax1.plot(range(len(sgd_l2_norms)), sgd_l2_norms, 's-', label='SGD', linewidth=2, markersize=6)
    ax1.set_title('Overall Weight Magnitude Evolution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Commit Number')
    ax1.set_ylabel('L2 Norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Weight Standard Deviation Evolution
    ax2 = plt.subplot(4, 3, 2)
    
    adam_stds = [stat['overall_stats']['std'] for stat in adam_stats]
    sgd_stds = [stat['overall_stats']['std'] for stat in sgd_stats]
    
    ax2.plot(range(len(adam_stds)), adam_stds, 'o-', label='Adam', linewidth=2, markersize=6)
    ax2.plot(range(len(sgd_stds)), sgd_stds, 's-', label='SGD', linewidth=2, markersize=6)
    ax2.set_title('Weight Variance Evolution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Commit Number')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Weight Range Evolution
    ax3 = plt.subplot(4, 3, 3)
    
    adam_ranges = [(stat['overall_stats']['max'] - stat['overall_stats']['min']) for stat in adam_stats]
    sgd_ranges = [(stat['overall_stats']['max'] - stat['overall_stats']['min']) for stat in sgd_stats]
    
    ax3.plot(range(len(adam_ranges)), adam_ranges, 'o-', label='Adam', linewidth=2, markersize=6)
    ax3.plot(range(len(sgd_ranges)), sgd_ranges, 's-', label='SGD', linewidth=2, markersize=6)
    ax3.set_title('Weight Range Evolution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Commit Number')
    ax3.set_ylabel('Max - Min')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Layer-specific L2 Norm Evolution (select key layers)
    ax4 = plt.subplot(4, 3, 4)
    
    key_layers = ['conv1_kernel', 'conv2_kernel', 'dense1_kernel', 'output_kernel']
    colors = ['red', 'blue', 'green', 'purple']
    
    for layer, color in zip(key_layers, colors):
        if layer in adam_stats[0]['layer_stats']:
            adam_layer_norms = [stat['layer_stats'][layer]['l2_norm'] for stat in adam_stats]
            ax4.plot(range(len(adam_layer_norms)), adam_layer_norms, 'o-', 
                    color=color, alpha=0.7, label=f'Adam {layer}', linewidth=2)
    
    ax4.set_title('Layer-specific Weight Evolution (Adam)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Commit Number')
    ax4.set_ylabel('L2 Norm')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Layer-specific L2 Norm Evolution for SGD
    ax5 = plt.subplot(4, 3, 5)
    
    for layer, color in zip(key_layers, colors):
        if layer in sgd_stats[0]['layer_stats']:
            sgd_layer_norms = [stat['layer_stats'][layer]['l2_norm'] for stat in sgd_stats]
            ax5.plot(range(len(sgd_layer_norms)), sgd_layer_norms, 's-', 
                    color=color, alpha=0.7, label=f'SGD {layer}', linewidth=2)
    
    ax5.set_title('Layer-specific Weight Evolution (SGD)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Commit Number')
    ax5.set_ylabel('L2 Norm')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.grid(True, alpha=0.3)
    
    # 6. Weight Change Magnitude Between Commits
    ax6 = plt.subplot(4, 3, 6)
    
    if adam_diffs and sgd_diffs:
        # Calculate total change magnitude for each commit-to-commit transition
        adam_changes = []
        sgd_changes = []
        
        for diff_data in adam_diffs:
            total_change = sum([diff_data[layer]['l2_diff'] for layer in diff_data.keys()])
            adam_changes.append(total_change)
        
        for diff_data in sgd_diffs:
            total_change = sum([diff_data[layer]['l2_diff'] for layer in diff_data.keys()])
            sgd_changes.append(total_change)
        
        ax6.plot(range(1, len(adam_changes)+1), adam_changes, 'o-', label='Adam', linewidth=2, markersize=6)
        ax6.plot(range(1, len(sgd_changes)+1), sgd_changes, 's-', label='SGD', linewidth=2, markersize=6)
    
    ax6.set_title('Weight Change Magnitude Between Commits', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Commit Transition')
    ax6.set_ylabel('Total L2 Change')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Weight Change
    ax7 = plt.subplot(4, 3, 7)
    
    if adam_diffs and sgd_diffs:
        adam_cumulative = np.cumsum([0] + adam_changes)
        sgd_cumulative = np.cumsum([0] + sgd_changes)
        
        ax7.plot(range(len(adam_cumulative)), adam_cumulative, 'o-', label='Adam', linewidth=2, markersize=6)
        ax7.plot(range(len(sgd_cumulative)), sgd_cumulative, 's-', label='SGD', linewidth=2, markersize=6)
    
    ax7.set_title('Cumulative Weight Change During Training', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Commit Number')
    ax7.set_ylabel('Cumulative L2 Change')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Weight Distribution Evolution (histogram comparison)
    ax8 = plt.subplot(4, 3, 8)
    
    # Compare initial vs final weight distributions
    if len(adam_stats) >= 2:
        initial_weights = []
        final_weights = []
        
        # Get all weights from first and last commits
        initial_weights_dict = adam_evolution['weights_history'][0]
        final_weights_dict = adam_evolution['weights_history'][-1]
        
        for layer_weights in initial_weights_dict.values():
            initial_weights.extend(layer_weights.flatten())
        for layer_weights in final_weights_dict.values():
            final_weights.extend(layer_weights.flatten())
        
        ax8.hist(initial_weights, bins=50, alpha=0.6, density=True, label='Adam Initial')
        ax8.hist(final_weights, bins=50, alpha=0.6, density=True, label='Adam Final')
    
    ax8.set_title('Weight Distribution: Initial vs Final (Adam)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Weight Value')
    ax8.set_ylabel('Density')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Learning Rate Effect Visualization
    ax9 = plt.subplot(4, 3, 9)
    
    # Plot relative changes (how much weights changed relative to their magnitude)
    if adam_diffs and sgd_diffs:
        adam_rel_changes = []
        sgd_rel_changes = []
        
        for diff_data in adam_diffs:
            avg_rel_change = np.mean([diff_data[layer]['relative_change'] for layer in diff_data.keys()])
            adam_rel_changes.append(avg_rel_change)
        
        for diff_data in sgd_diffs:
            avg_rel_change = np.mean([diff_data[layer]['relative_change'] for layer in diff_data.keys()])
            sgd_rel_changes.append(avg_rel_change)
        
        ax9.plot(range(1, len(adam_rel_changes)+1), adam_rel_changes, 'o-', label='Adam', linewidth=2, markersize=6)
        ax9.plot(range(1, len(sgd_rel_changes)+1), sgd_rel_changes, 's-', label='SGD', linewidth=2, markersize=6)
    
    ax9.set_title('Relative Weight Change Rate', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Commit Transition')
    ax9.set_ylabel('Relative Change')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Training Stability Analysis
    ax10 = plt.subplot(4, 3, 10)
    
    # Show variance in weight changes (training stability)
    if len(adam_changes) > 1 and len(sgd_changes) > 1:
        adam_stability = np.std(adam_changes)
        sgd_stability = np.std(sgd_changes)
        
        strategies = ['Adam', 'SGD']
        stabilities = [adam_stability, sgd_stability]
        colors = ['blue', 'orange']
        
        bars = ax10.bar(strategies, stabilities, color=colors, alpha=0.7)
        ax10.set_title('Training Stability (Lower = More Stable)', fontsize=12, fontweight='bold')
        ax10.set_ylabel('Std Dev of Weight Changes')
        
        # Add value labels on bars
        for bar, val in zip(bars, stabilities):
            ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stabilities)*0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax10.grid(True, alpha=0.3)
    
    # 11. Convergence Analysis
    ax11 = plt.subplot(4, 3, 11)
    
    # Show how the rate of change decreases over time (convergence)
    if adam_changes and sgd_changes:
        # Smooth the changes with a rolling average
        adam_smooth = pd.Series(adam_changes).rolling(window=min(3, len(adam_changes)), center=True).mean()
        sgd_smooth = pd.Series(sgd_changes).rolling(window=min(3, len(sgd_changes)), center=True).mean()
        
        ax11.plot(range(1, len(adam_smooth)+1), adam_smooth, 'o-', label='Adam (smoothed)', linewidth=2, markersize=4)
        ax11.plot(range(1, len(sgd_smooth)+1), sgd_smooth, 's-', label='SGD (smoothed)', linewidth=2, markersize=4)
    
    ax11.set_title('Convergence Pattern (Smoothed)', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Commit Transition')
    ax11.set_ylabel('Weight Change (Smoothed)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Summary Statistics Table
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    # Create summary table
    summary_text = f"""
COMMIT EVOLUTION SUMMARY

Adam Branch:
• Total Commits: {len(adam_stats)}
• Initial L2 Norm: {adam_stats[0]['overall_stats']['l2_norm']:.2f}
• Final L2 Norm: {adam_stats[-1]['overall_stats']['l2_norm']:.2f}
• Total Change: {adam_cumulative[-1] if adam_diffs else 'N/A'}

SGD Branch:
• Total Commits: {len(sgd_stats)}
• Initial L2 Norm: {sgd_stats[0]['overall_stats']['l2_norm']:.2f}
• Final L2 Norm: {sgd_stats[-1]['overall_stats']['l2_norm']:.2f}
• Total Change: {sgd_cumulative[-1] if sgd_diffs else 'N/A'}

Training Dynamics:
• Adam: {'More Dynamic' if adam_stability > sgd_stability else 'More Stable'} 
• SGD: {'More Dynamic' if sgd_stability > adam_stability else 'More Stable'}

Convergence:
• Adam Final Rate: {adam_changes[-1] if adam_changes else 'N/A'}
• SGD Final Rate: {sgd_changes[-1] if sgd_changes else 'N/A'}
    """
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('commit_evolution_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved commit evolution analysis to 'commit_evolution_analysis.png'")
    
    return fig

def print_evolution_summary(adam_evolution, sgd_evolution):
    """Print detailed summary of weight evolution analysis."""
    
    print("\n" + "="*80)
    print("📈 COMMIT EVOLUTION ANALYSIS SUMMARY")
    print("="*80)
    
    adam_stats = adam_evolution['statistics_history']
    sgd_stats = sgd_evolution['statistics_history']
    adam_diffs = [d for d in adam_evolution['differences_history'] if d is not None]
    sgd_diffs = [d for d in sgd_evolution['differences_history'] if d is not None]
    
    print(f"\n📊 COMMIT ANALYSIS:")
    print(f"{'Branch':<10} {'Commits':<8} {'Initial L2':<12} {'Final L2':<12} {'Change':<12}")
    print("-" * 60)
    
    adam_initial = adam_stats[0]['overall_stats']['l2_norm']
    adam_final = adam_stats[-1]['overall_stats']['l2_norm']
    adam_change = adam_final - adam_initial
    
    sgd_initial = sgd_stats[0]['overall_stats']['l2_norm']
    sgd_final = sgd_stats[-1]['overall_stats']['l2_norm']
    sgd_change = sgd_final - sgd_initial
    
    print(f"{'Adam':<10} {len(adam_stats):<8} {adam_initial:<12.3f} {adam_final:<12.3f} {adam_change:<+12.3f}")
    print(f"{'SGD':<10} {len(sgd_stats):<8} {sgd_initial:<12.3f} {sgd_final:<12.3f} {sgd_change:<+12.3f}")
    
    print(f"\n🔄 TRAINING DYNAMICS:")
    if adam_diffs and sgd_diffs:
        adam_changes = [sum([diff_data[layer]['l2_diff'] for layer in diff_data.keys()]) for diff_data in adam_diffs]
        sgd_changes = [sum([diff_data[layer]['l2_diff'] for layer in diff_data.keys()]) for diff_data in sgd_diffs]
        
        adam_avg_change = np.mean(adam_changes)
        sgd_avg_change = np.mean(sgd_changes)
        adam_stability = np.std(adam_changes)
        sgd_stability = np.std(sgd_changes)
        
        print(f"  Adam - Avg Change: {adam_avg_change:.3f}, Stability (std): {adam_stability:.3f}")
        print(f"  SGD  - Avg Change: {sgd_avg_change:.3f}, Stability (std): {sgd_stability:.3f}")
        
        if adam_avg_change > sgd_avg_change:
            print(f"  🏃 Adam made larger weight updates on average")
        else:
            print(f"  🚶 SGD made larger weight updates on average")
        
        if adam_stability > sgd_stability:
            print(f"  📈 Adam showed more variable training dynamics")
        else:
            print(f"  📈 SGD showed more variable training dynamics")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print(f"  • Adam network {'grew' if adam_change > 0 else 'shrank'} by {abs(adam_change):.3f} L2 units")
    print(f"  • SGD network {'grew' if sgd_change > 0 else 'shrank'} by {abs(sgd_change):.3f} L2 units")
    
    # Analyze layer-specific evolution
    print(f"\n🧠 LAYER-SPECIFIC EVOLUTION:")
    print(f"{'Layer':<15} {'Adam Change':<12} {'SGD Change':<12} {'Difference':<12}")
    print("-" * 51)
    
    common_layers = set(adam_stats[0]['layer_stats'].keys()) & set(sgd_stats[0]['layer_stats'].keys())
    for layer in sorted(common_layers):
        adam_layer_initial = adam_stats[0]['layer_stats'][layer]['l2_norm']
        adam_layer_final = adam_stats[-1]['layer_stats'][layer]['l2_norm']
        adam_layer_change = adam_layer_final - adam_layer_initial
        
        sgd_layer_initial = sgd_stats[0]['layer_stats'][layer]['l2_norm']
        sgd_layer_final = sgd_stats[-1]['layer_stats'][layer]['l2_norm']
        sgd_layer_change = sgd_layer_final - sgd_layer_initial
        
        diff = adam_layer_change - sgd_layer_change
        
        print(f"{layer:<15} {adam_layer_change:<+12.3f} {sgd_layer_change:<+12.3f} {diff:<+12.3f}")

def main():
    """Main commit evolution analysis function."""
    
    print("="*80)
    print("📈 PARAMLAKE COMMIT EVOLUTION ANALYSIS")
    print("Tracking Weight Changes Across Training History")
    print("="*80)
    
    try:
        # Find and connect to the repository
        import boto3
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        
        repositories = []
        for page in paginator.paginate(Bucket='paramlake', Prefix='mnist-'):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if 'icechunk' in key or key.endswith('.json'):
                        repo_prefix = key.split('/')[0]
                        if repo_prefix not in repositories:
                            repositories.append(repo_prefix)
        
        if not repositories:
            print("❌ No repositories found")
            return
        
        repo_prefix = sorted(repositories)[-1]
        print(f"📁 Using repository: {repo_prefix}")
        
        # Connect to repository
        storage = icechunk.s3_storage(
            bucket="paramlake", 
            prefix=repo_prefix,
            from_env=True
        )
        
        repo = icechunk.Repository.open(storage)
        print(f"✅ Successfully connected to repository")
        
        # Get commit history for both branches
        adam_commits = get_commit_history(repo, 'adam')
        sgd_commits = get_commit_history(repo, 'sgd')
        
        if not adam_commits or not sgd_commits:
            print("❌ Could not get commit history for one or both branches")
            return
        
        # Analyze evolution for both branches
        adam_evolution = analyze_weight_evolution(repo, adam_commits, 'Adam')
        sgd_evolution = analyze_weight_evolution(repo, sgd_commits, 'SGD')
        
        if not adam_evolution['weights_history'] or not sgd_evolution['weights_history']:
            print("❌ Could not analyze weight evolution")
            return
        
        # Create visualizations
        create_evolution_plots(adam_evolution, sgd_evolution)
        
        # Print detailed summary
        print_evolution_summary(adam_evolution, sgd_evolution)
        
        print(f"\n✅ COMMIT EVOLUTION ANALYSIS COMPLETE!")
        print(f"📁 Generated visualization: commit_evolution_analysis.png")
        
    except Exception as e:
        print(f"❌ Error in commit evolution analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 