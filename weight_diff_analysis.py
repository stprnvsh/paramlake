#!/usr/bin/env python3
"""
Weight Difference Analysis - Compare Adam vs SGD Training Strategies

This script loads weights from both Adam and SGD branches in the icechunk repository
and creates comprehensive visualizations of the differences between the two training strategies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
import icechunk
import pandas as pd
from matplotlib.patches import Rectangle

# Set AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_aws_access_key_id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_aws_secret_access_key'
os.environ['AWS_REGION'] = 'us-east-1'

def load_weights_from_branch(repo, branch_name):
    """Load all weights from a specific branch."""
    
    print(f"📥 Loading weights from branch: {branch_name}")
    
    try:
        session = repo.readonly_session(branch_name)
        store = session.store
        group = zarr.open_group(store, mode="r")
        
        if 'parameters' not in group:
            print(f"❌ No parameters found in branch {branch_name}")
            return {}
        
        params_group = group['parameters']
        weights = {}
        
        for param_name in params_group.keys():
            param_array = params_group[param_name]
            weights[param_name] = param_array[:]
            print(f"  ✓ Loaded {param_name}: {weights[param_name].shape}")
        
        print(f"✅ Successfully loaded {len(weights)} weight arrays from {branch_name}")
        return weights
        
    except Exception as e:
        print(f"❌ Error loading weights from {branch_name}: {e}")
        return {}

def compute_weight_statistics(weights_dict, name):
    """Compute comprehensive statistics for a set of weights."""
    
    stats = {
        'name': name,
        'total_params': 0,
        'layer_stats': {},
        'overall_stats': {}
    }
    
    all_weights = []
    
    for layer_name, weight_array in weights_dict.items():
        layer_stats = {
            'shape': weight_array.shape,
            'num_params': weight_array.size,
            'mean': np.mean(weight_array),
            'std': np.std(weight_array),
            'min': np.min(weight_array),
            'max': np.max(weight_array),
            'l2_norm': np.linalg.norm(weight_array),
            'sparsity': np.sum(np.abs(weight_array) < 1e-6) / weight_array.size
        }
        
        stats['layer_stats'][layer_name] = layer_stats
        stats['total_params'] += layer_stats['num_params']
        all_weights.extend(weight_array.flatten())
    
    # Overall statistics
    all_weights = np.array(all_weights)
    stats['overall_stats'] = {
        'mean': np.mean(all_weights),
        'std': np.std(all_weights),
        'min': np.min(all_weights),
        'max': np.max(all_weights),
        'l2_norm': np.linalg.norm(all_weights),
        'sparsity': np.sum(np.abs(all_weights) < 1e-6) / len(all_weights)
    }
    
    return stats

def compute_weight_differences(adam_weights, sgd_weights):
    """Compute detailed differences between Adam and SGD weights."""
    
    print("🔍 Computing weight differences...")
    
    differences = {}
    
    # Ensure both dictionaries have the same keys
    common_keys = set(adam_weights.keys()) & set(sgd_weights.keys())
    if not common_keys:
        print("❌ No common weight arrays found between branches")
        return {}
    
    print(f"📊 Comparing {len(common_keys)} common weight arrays")
    
    for layer_name in common_keys:
        adam_w = adam_weights[layer_name]
        sgd_w = sgd_weights[layer_name]
        
        if adam_w.shape != sgd_w.shape:
            print(f"⚠️ Shape mismatch for {layer_name}: {adam_w.shape} vs {sgd_w.shape}")
            continue
        
        # Compute various difference metrics
        diff = adam_w - sgd_w
        abs_diff = np.abs(diff)
        rel_diff = abs_diff / (np.abs(adam_w) + 1e-8)  # Relative difference
        
        layer_diff = {
            'raw_diff': diff,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff,
            'mean_abs_diff': np.mean(abs_diff),
            'max_abs_diff': np.max(abs_diff),
            'std_diff': np.std(diff),
            'l2_diff': np.linalg.norm(diff),
            'cosine_similarity': np.dot(adam_w.flatten(), sgd_w.flatten()) / 
                               (np.linalg.norm(adam_w) * np.linalg.norm(sgd_w)),
            'correlation': np.corrcoef(adam_w.flatten(), sgd_w.flatten())[0, 1]
        }
        
        differences[layer_name] = layer_diff
        
        print(f"  📈 {layer_name}:")
        print(f"    Mean absolute diff: {layer_diff['mean_abs_diff']:.6f}")
        print(f"    Max absolute diff:  {layer_diff['max_abs_diff']:.6f}")
        print(f"    L2 norm diff:       {layer_diff['l2_diff']:.6f}")
        print(f"    Cosine similarity:  {layer_diff['cosine_similarity']:.6f}")
    
    return differences

def create_weight_comparison_plots(adam_weights, sgd_weights, differences):
    """Create comprehensive visualization of weight differences."""
    
    print("📊 Creating weight comparison visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate the number of subplots needed
    common_keys = list(differences.keys())
    n_layers = len(common_keys)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 5 * n_layers))
    
    for i, layer_name in enumerate(common_keys):
        adam_w = adam_weights[layer_name]
        sgd_w = sgd_weights[layer_name]
        diff_data = differences[layer_name]
        
        # Create a row of 4 subplots for each layer
        base_idx = i * 4 + 1
        
        # 1. Weight distribution comparison
        ax1 = plt.subplot(n_layers, 4, base_idx)
        adam_flat = adam_w.flatten()
        sgd_flat = sgd_w.flatten()
        
        ax1.hist(adam_flat, bins=50, alpha=0.7, label='Adam', density=True)
        ax1.hist(sgd_flat, bins=50, alpha=0.7, label='SGD', density=True)
        ax1.set_title(f'{layer_name}\nWeight Distributions')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute difference heatmap (for 2D weights)
        ax2 = plt.subplot(n_layers, 4, base_idx + 1)
        abs_diff = diff_data['abs_diff']
        
        if len(abs_diff.shape) >= 2:
            # For conv kernels, show first channel or average across channels
            if len(abs_diff.shape) == 4:  # Conv kernel (h, w, in_ch, out_ch)
                diff_2d = np.mean(abs_diff, axis=(2, 3))  # Average across channels
            elif len(abs_diff.shape) == 2:  # Dense layer (in, out)
                diff_2d = abs_diff
            else:
                diff_2d = abs_diff.reshape(-1, abs_diff.shape[-1])
            
            im = ax2.imshow(diff_2d, cmap='Reds', aspect='auto')
            plt.colorbar(im, ax=ax2)
        else:
            # For 1D arrays (biases), show as bar plot
            ax2.bar(range(len(abs_diff)), abs_diff)
        
        ax2.set_title(f'{layer_name}\nAbsolute Differences')
        
        # 3. Scatter plot: Adam vs SGD weights
        ax3 = plt.subplot(n_layers, 4, base_idx + 2)
        
        # Sample points for large arrays to avoid overcrowding
        n_sample = min(5000, len(adam_flat))
        idx = np.random.choice(len(adam_flat), n_sample, replace=False)
        
        ax3.scatter(adam_flat[idx], sgd_flat[idx], alpha=0.6, s=1)
        
        # Add diagonal line
        min_val = min(np.min(adam_flat), np.min(sgd_flat))
        max_val = max(np.max(adam_flat), np.max(sgd_flat))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax3.set_xlabel('Adam Weights')
        ax3.set_ylabel('SGD Weights')
        ax3.set_title(f'{layer_name}\nAdam vs SGD Correlation')
        ax3.grid(True, alpha=0.3)
        
        # Add correlation text
        corr = diff_data['correlation']
        ax3.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                transform=ax3.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. Difference statistics
        ax4 = plt.subplot(n_layers, 4, base_idx + 3)
        ax4.axis('off')
        
        # Create text summary of statistics
        stats_text = f"""
{layer_name} Statistics:

Shape: {adam_w.shape}
Total Parameters: {adam_w.size:,}

Difference Metrics:
• Mean Abs Diff: {diff_data['mean_abs_diff']:.6f}
• Max Abs Diff: {diff_data['max_abs_diff']:.6f}
• L2 Norm Diff: {diff_data['l2_diff']:.6f}
• Std Dev Diff: {diff_data['std_diff']:.6f}

Similarity Metrics:
• Cosine Similarity: {diff_data['cosine_similarity']:.6f}
• Correlation: {diff_data['correlation']:.6f}

Adam Stats:
• Mean: {np.mean(adam_w):.6f}
• Std: {np.std(adam_w):.6f}
• L2 Norm: {np.linalg.norm(adam_w):.6f}

SGD Stats:
• Mean: {np.mean(sgd_w):.6f}
• Std: {np.std(sgd_w):.6f}
• L2 Norm: {np.linalg.norm(sgd_w):.6f}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('weight_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved detailed weight comparison to 'weight_comparison_analysis.png'")
    
    return fig

def create_summary_plots(adam_stats, sgd_stats, differences):
    """Create summary plots comparing overall statistics."""
    
    print("📊 Creating summary comparison plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract layer names and metrics
    layer_names = list(differences.keys())
    
    # 1. L2 Norm Differences by Layer
    ax = axes[0, 0]
    l2_diffs = [differences[layer]['l2_diff'] for layer in layer_names]
    bars = ax.bar(range(len(layer_names)), l2_diffs, color='skyblue', alpha=0.8)
    ax.set_title('L2 Norm Differences by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm Difference')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, l2_diffs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(l2_diffs)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Cosine Similarity by Layer
    ax = axes[0, 1]
    cosine_sims = [differences[layer]['cosine_similarity'] for layer in layer_names]
    bars = ax.bar(range(len(layer_names)), cosine_sims, color='lightcoral', alpha=0.8)
    ax.set_title('Cosine Similarity by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(cosine_sims) * 0.95, 1.0])
    
    # Add value labels
    for bar, val in zip(bars, cosine_sims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1-min(cosine_sims))*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Mean Absolute Differences
    ax = axes[0, 2]
    mean_diffs = [differences[layer]['mean_abs_diff'] for layer in layer_names]
    bars = ax.bar(range(len(layer_names)), mean_diffs, color='lightgreen', alpha=0.8)
    ax.set_title('Mean Absolute Differences by Layer', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 4. Weight Magnitude Comparison
    ax = axes[1, 0]
    adam_l2_norms = [np.linalg.norm(adam_stats['layer_stats'][layer]['shape']) for layer in layer_names 
                     if layer in adam_stats['layer_stats']]
    sgd_l2_norms = [np.linalg.norm(sgd_stats['layer_stats'][layer]['shape']) for layer in layer_names 
                    if layer in sgd_stats['layer_stats']]
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    ax.bar(x - width/2, [adam_stats['layer_stats'][layer]['l2_norm'] for layer in layer_names], 
           width, label='Adam', alpha=0.8, color='blue')
    ax.bar(x + width/2, [sgd_stats['layer_stats'][layer]['l2_norm'] for layer in layer_names], 
           width, label='SGD', alpha=0.8, color='orange')
    
    ax.set_title('Weight Magnitudes (L2 Norm) Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('L2 Norm')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Overall Statistics Comparison
    ax = axes[1, 1]
    metrics = ['mean', 'std', 'sparsity']
    adam_vals = [adam_stats['overall_stats'][m] for m in metrics]
    sgd_vals = [sgd_stats['overall_stats'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    ax.bar(x - width/2, adam_vals, width, label='Adam', alpha=0.8, color='blue')
    ax.bar(x + width/2, sgd_vals, width, label='SGD', alpha=0.8, color='orange')
    
    ax.set_title('Overall Weight Statistics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Difference Distribution
    ax = axes[1, 2]
    all_diffs = []
    for layer in layer_names:
        all_diffs.extend(differences[layer]['raw_diff'].flatten())
    
    ax.hist(all_diffs, bins=100, alpha=0.7, color='purple', density=True)
    ax.set_title('Distribution of All Weight Differences', fontsize=14, fontweight='bold')
    ax.set_xlabel('Weight Difference (Adam - SGD)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.axvline(np.mean(all_diffs), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(all_diffs):.6f}')
    ax.axvline(np.median(all_diffs), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(all_diffs):.6f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('weight_diff_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Saved summary comparison to 'weight_diff_summary.png'")
    
    return fig

def print_detailed_analysis(adam_stats, sgd_stats, differences):
    """Print detailed numerical analysis of the differences."""
    
    print("\n" + "="*80)
    print("📊 DETAILED WEIGHT DIFFERENCE ANALYSIS")
    print("="*80)
    
    print(f"\n🔍 OVERALL STATISTICS COMPARISON:")
    print(f"{'Metric':<20} {'Adam':<15} {'SGD':<15} {'Difference':<15}")
    print("-" * 65)
    
    for metric in ['mean', 'std', 'min', 'max', 'l2_norm', 'sparsity']:
        adam_val = adam_stats['overall_stats'][metric]
        sgd_val = sgd_stats['overall_stats'][metric]
        diff = adam_val - sgd_val
        print(f"{metric:<20} {adam_val:<15.6f} {sgd_val:<15.6f} {diff:<15.6f}")
    
    print(f"\n📋 LAYER-BY-LAYER ANALYSIS:")
    print(f"{'Layer':<15} {'L2 Diff':<12} {'Mean Diff':<12} {'Max Diff':<12} {'Cosine Sim':<12}")
    print("-" * 63)
    
    for layer in differences.keys():
        diff_data = differences[layer]
        print(f"{layer:<15} {diff_data['l2_diff']:<12.6f} {diff_data['mean_abs_diff']:<12.6f} "
              f"{diff_data['max_abs_diff']:<12.6f} {diff_data['cosine_similarity']:<12.6f}")
    
    # Find most and least different layers
    l2_diffs = {layer: differences[layer]['l2_diff'] for layer in differences.keys()}
    most_diff_layer = max(l2_diffs, key=l2_diffs.get)
    least_diff_layer = min(l2_diffs, key=l2_diffs.get)
    
    print(f"\n🏆 KEY FINDINGS:")
    print(f"  • Most different layer: {most_diff_layer} (L2 diff: {l2_diffs[most_diff_layer]:.6f})")
    print(f"  • Least different layer: {least_diff_layer} (L2 diff: {l2_diffs[least_diff_layer]:.6f})")
    print(f"  • Total parameters compared: {adam_stats['total_params']:,}")
    
    # Calculate overall similarity
    avg_cosine_sim = np.mean([differences[layer]['cosine_similarity'] for layer in differences.keys()])
    print(f"  • Average cosine similarity: {avg_cosine_sim:.6f}")
    
    if avg_cosine_sim > 0.9:
        print(f"  • ✅ Networks are highly similar despite different training")
    elif avg_cosine_sim > 0.7:
        print(f"  • ⚠️ Networks show moderate similarity")
    else:
        print(f"  • ❌ Networks are quite different")

def main():
    """Main analysis function."""
    
    print("="*80)
    print("🔍 PARAMLAKE WEIGHT DIFFERENCE ANALYSIS")
    print("Comparing Adam vs SGD Training Strategies")
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
        
        # Load weights from both branches
        adam_weights = load_weights_from_branch(repo, 'adam')
        sgd_weights = load_weights_from_branch(repo, 'sgd')
        
        if not adam_weights or not sgd_weights:
            print("❌ Failed to load weights from one or both branches")
            return
        
        # Compute statistics
        print("\n📊 Computing weight statistics...")
        adam_stats = compute_weight_statistics(adam_weights, 'Adam')
        sgd_stats = compute_weight_statistics(sgd_weights, 'SGD')
        
        # Compute differences
        differences = compute_weight_differences(adam_weights, sgd_weights)
        
        if not differences:
            print("❌ No weight differences could be computed")
            return
        
        # Create visualizations
        create_weight_comparison_plots(adam_weights, sgd_weights, differences)
        create_summary_plots(adam_stats, sgd_stats, differences)
        
        # Print detailed analysis
        print_detailed_analysis(adam_stats, sgd_stats, differences)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 Generated visualizations:")
        print(f"  • weight_comparison_analysis.png - Detailed layer-by-layer comparison")
        print(f"  • weight_diff_summary.png - Summary statistics and distributions")
        
    except Exception as e:
        print(f"❌ Error in weight difference analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 