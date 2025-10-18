"""
Simple visualization script for batch size experiment results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(filename='batch_experiment_results.json'):
    """Load results from JSON file."""
    with open(filename, 'r') as f:
        results = json.load(f)
    return {int(k): v for k, v in results.items()}


def plot_results(results):
    """Create visualization of batch size experiment results."""
    batch_sizes = sorted(results.keys())
    
    # Aggregate metrics
    metrics = {
        'Communication Accuracy': [],
        'Eve Success Rate': [],
        'Secrecy Score': [],
        'Stability Score': []
    }
    
    errors = {
        'Communication Accuracy': [],
        'Eve Success Rate': [],
        'Secrecy Score': [],
        'Stability Score': []
    }
    
    for bs in batch_sizes:
        runs = results[bs]
        metrics['Communication Accuracy'].append(np.mean([r['communication_accuracy'] for r in runs]))
        metrics['Eve Success Rate'].append(np.mean([r['eve_success_rate'] for r in runs]))
        metrics['Secrecy Score'].append(np.mean([r['secrecy_score'] for r in runs]))
        metrics['Stability Score'].append(np.mean([r['stability_score'] for r in runs]))
        
        errors['Communication Accuracy'].append(np.std([r['communication_accuracy'] for r in runs]))
        errors['Eve Success Rate'].append(np.std([r['eve_success_rate'] for r in runs]))
        errors['Secrecy Score'].append(np.std([r['secrecy_score'] for r in runs]))
        errors['Stability Score'].append(np.std([r['stability_score'] for r in runs]))
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Batch Size Experiment Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Communication Accuracy
    ax = axes[0, 0]
    ax.errorbar(batch_sizes, metrics['Communication Accuracy'], 
                yerr=errors['Communication Accuracy'], 
                marker='o', capsize=5, linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Communication Accuracy')
    ax.set_title('Bob\'s Decryption Success')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Plot 2: Eve Success Rate
    ax = axes[0, 1]
    ax.errorbar(batch_sizes, metrics['Eve Success Rate'], 
                yerr=errors['Eve Success Rate'], 
                marker='s', capsize=5, linewidth=2, markersize=8, color='red')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random Guessing')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Eve Success Rate')
    ax.set_title('Adversary Attack Effectiveness')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    ax.legend()
    
    # Plot 3: Secrecy Score
    ax = axes[1, 0]
    ax.errorbar(batch_sizes, metrics['Secrecy Score'], 
                yerr=errors['Secrecy Score'], 
                marker='^', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Secrecy Score')
    ax.set_title('Security Gap (Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Stability Score
    ax = axes[1, 1]
    ax.errorbar(batch_sizes, metrics['Stability Score'], 
                yerr=errors['Stability Score'], 
                marker='D', capsize=5, linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Stability Score')
    ax.set_title('Training Stability (Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_experiment_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: batch_experiment_results.png")
    plt.show()


def plot_tradeoffs(results):
    """Plot trade-off analysis."""
    batch_sizes = sorted(results.keys())
    
    comm_acc = [np.mean([r['communication_accuracy'] for r in results[bs]]) for bs in batch_sizes]
    secrecy = [np.mean([r['secrecy_score'] for r in results[bs]]) for bs in batch_sizes]
    stability = [np.mean([r['stability_score'] for r in results[bs]]) for bs in batch_sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Trade-off Analysis', fontsize=16, fontweight='bold')
    
    # Communication vs Security
    ax = axes[0]
    ax.plot(comm_acc, secrecy, 'o-', linewidth=2, markersize=10)
    for i, bs in enumerate(batch_sizes):
        ax.annotate(f'{bs}', (comm_acc[i], secrecy[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Communication Accuracy')
    ax.set_ylabel('Secrecy Score')
    ax.set_title('Accuracy vs Security')
    ax.grid(True, alpha=0.3)
    
    # Stability vs Security
    ax = axes[1]
    ax.plot(stability, secrecy, 's-', linewidth=2, markersize=10, color='green')
    for i, bs in enumerate(batch_sizes):
        ax.annotate(f'{bs}', (stability[i], secrecy[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Stability Score')
    ax.set_ylabel('Secrecy Score')
    ax.set_title('Stability vs Security')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    print("Plot saved as: tradeoff_analysis.png")
    plt.show()


def print_summary(results):
    """Print text summary of results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    batch_sizes = sorted(results.keys())
    
    print(f"\n{'Batch Size':<12} {'Comm Acc':<12} {'Eve Rate':<12} {'Secrecy':<12} {'Stability':<12}")
    print("-"*80)
    
    for bs in batch_sizes:
        runs = results[bs]
        comm_acc = np.mean([r['communication_accuracy'] for r in runs])
        eve_rate = np.mean([r['eve_success_rate'] for r in runs])
        secrecy = np.mean([r['secrecy_score'] for r in runs])
        stability = np.mean([r['stability_score'] for r in runs])
        
        print(f"{bs:<12} {comm_acc:<12.2%} {eve_rate:<12.2%} {secrecy:<12.2f} {stability:<12.3f}")


if __name__ == "__main__":
    # Load results
    results = load_results()
    
    # Print summary
    print_summary(results)
    
    # Create plots
    plot_results(results)
    plot_tradeoffs(results)
    
    print("\nVisualization complete!")