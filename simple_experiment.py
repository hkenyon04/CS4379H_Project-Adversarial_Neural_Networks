"""
Simplified Batch Size Experiment for Neural Cryptography
Uses TensorFlow 2.x and focuses on key metrics only.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from datagen import get_random_block
from net import build_network


def calculate_bit_errors(true_msg, predicted_msg):
    """Calculate number of bit errors."""
    return tf.reduce_mean(tf.abs(true_msg - predicted_msg)) / 2 * 16


class CryptoTrainer:
    """Handles training of Alice, Bob, and Eve."""
    
    def __init__(self, message_length=16, learning_rate=0.0008):
        self.message_length = message_length
        self.alice, self.bob, self.eve = build_network(message_length, message_length)
        
        # Optimizers
        self.alice_bob_optimizer = keras.optimizers.Adam(learning_rate)
        self.eve_optimizer = keras.optimizers.Adam(learning_rate)
        
        # Track metrics
        self.history = {
            'bob_bit_errors': [],
            'eve_bit_errors': [],
            'bob_accuracy': [],
            'eve_accuracy': [],
            'secrecy_score': []
        }
    
    @tf.function
    def train_alice_bob_step(self, msg, key):
        """One training step for Alice and Bob."""
        with tf.GradientTape() as tape:
            # Alice encrypts
            ciphertext = self.alice([msg, key], training=True)
            
            # Bob decrypts
            bob_decrypted = self.bob([ciphertext, key], training=True)
            
            # Eve tries to decrypt (for adversarial loss)
            eve_decrypted = self.eve(ciphertext, training=False)
            
            # Bob's reconstruction loss
            bob_loss = tf.reduce_mean(tf.abs(msg - bob_decrypted)) / 2
            
            # Eve's reconstruction loss
            eve_loss = tf.reduce_mean(tf.abs(msg - eve_decrypted)) / 2
            
            # Combined loss: Bob should succeed, Eve should fail
            total_loss = bob_loss + tf.square(0.5 - eve_loss)
        
        # Update Alice and Bob
        trainable_vars = self.alice.trainable_variables + self.bob.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.alice_bob_optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return bob_loss, eve_loss
    
    @tf.function
    def train_eve_step(self, msg, key):
        """One training step for Eve."""
        # Get ciphertext from Alice
        ciphertext = self.alice([msg, key], training=False)
        
        with tf.GradientTape() as tape:
            # Eve tries to decrypt
            eve_decrypted = self.eve(ciphertext, training=True)
            
            # Eve's loss
            eve_loss = tf.reduce_mean(tf.abs(msg - eve_decrypted)) / 2
        
        # Update Eve
        gradients = tape.gradient(eve_loss, self.eve.trainable_variables)
        self.eve_optimizer.apply_gradients(zip(gradients, self.eve.trainable_variables))
        
        return eve_loss
    
    def evaluate(self, msg, key):
        """Evaluate current performance."""
        # Get predictions
        ciphertext = self.alice([msg, key], training=False)
        bob_decrypted = self.bob([ciphertext, key], training=False)
        eve_decrypted = self.eve(ciphertext, training=False)
        
        # Calculate bit errors
        bob_bit_errors = float(calculate_bit_errors(msg, bob_decrypted).numpy())
        eve_bit_errors = float(calculate_bit_errors(msg, eve_decrypted).numpy())
        
        # Calculate accuracies (percentage of correct bits)
        bob_accuracy = 1.0 - (bob_bit_errors / self.message_length)
        eve_accuracy = 1.0 - (eve_bit_errors / self.message_length)
        
        # Secrecy score (how much better Bob is than Eve)
        secrecy_score = eve_bit_errors - bob_bit_errors
        
        return {
            'bob_bit_errors': bob_bit_errors,
            'eve_bit_errors': eve_bit_errors,
            'bob_accuracy': bob_accuracy,
            'eve_accuracy': eve_accuracy,
            'secrecy_score': secrecy_score
        }
    
    def train(self, batch_size, num_iterations=100, verbose=True):
        """Train the networks."""
        print(f"\nTraining with batch size: {batch_size}")
        print("="*60)
        
        for iteration in range(num_iterations):
            # Generate data
            msg = tf.constant(get_random_block(self.message_length, batch_size), dtype=tf.float32)
            key = tf.constant(get_random_block(self.message_length, batch_size), dtype=tf.float32)
            
            # Train Alice and Bob (20 steps)
            for _ in range(20):
                self.train_alice_bob_step(msg, key)
            
            # Train Eve (40 steps)
            for _ in range(40):
                self.train_eve_step(msg, key)
            
            # Evaluate every iteration
            metrics = self.evaluate(msg, key)
            
            # Store metrics
            self.history['bob_bit_errors'].append(metrics['bob_bit_errors'])
            self.history['eve_bit_errors'].append(metrics['eve_bit_errors'])
            self.history['bob_accuracy'].append(metrics['bob_accuracy'])
            self.history['eve_accuracy'].append(metrics['eve_accuracy'])
            self.history['secrecy_score'].append(metrics['secrecy_score'])
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration:3d} | Bob: {metrics['bob_accuracy']:.2%} | "
                      f"Eve: {metrics['eve_accuracy']:.2%} | "
                      f"Secrecy: {metrics['secrecy_score']:.2f}")
        
        return self.history


def calculate_summary_metrics(history):
    """Calculate summary statistics from training history."""
    # Use last 10 iterations for final metrics
    bob_acc = np.mean(history['bob_accuracy'][-10:])
    eve_acc = np.mean(history['eve_accuracy'][-10:])
    secrecy = np.mean(history['secrecy_score'][-10:])
    
    # Calculate stability (lower variance = more stable)
    bob_std = np.std(history['bob_accuracy'][-20:])
    eve_std = np.std(history['eve_accuracy'][-20:])
    stability = 1.0 / (1.0 + bob_std + eve_std)  # Normalized stability score
    
    # Find convergence point (when Bob first exceeds 90% accuracy)
    convergence_iter = -1
    for i, acc in enumerate(history['bob_accuracy']):
        if acc >= 0.90:
            convergence_iter = i
            break
    
    return {
        'communication_accuracy': bob_acc,
        'eve_success_rate': eve_acc,
        'secrecy_score': secrecy,
        'stability_score': stability,
        'convergence_iteration': convergence_iter,
        'bob_accuracy_std': bob_std,
        'eve_accuracy_std': eve_std
    }


def run_batch_experiment(batch_sizes=[512, 1024, 2048, 4096], 
                        num_runs=3, 
                        num_iterations=100):
    """Run experiments across different batch sizes."""
    
    print("="*80)
    print("BATCH SIZE EXPERIMENT - NEURAL CRYPTOGRAPHY")
    print("="*80)
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch: {num_runs}")
    print(f"Iterations: {num_iterations}")
    print("="*80)
    
    all_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"{'='*80}")
        
        batch_results = []
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Set random seed for reproducibility
            np.random.seed(42 + run)
            tf.random.set_seed(42 + run)
            
            # Train
            start_time = time.time()
            trainer = CryptoTrainer()
            history = trainer.train(batch_size, num_iterations, verbose=True)
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_summary_metrics(history)
            metrics['batch_size'] = batch_size
            metrics['run'] = run + 1
            metrics['training_time'] = training_time
            
            batch_results.append(metrics)
            
            # Print summary
            print(f"\nRun {run + 1} Summary:")
            print(f"  Communication Accuracy: {metrics['communication_accuracy']:.2%}")
            print(f"  Eve Success Rate: {metrics['eve_success_rate']:.2%}")
            print(f"  Secrecy Score: {metrics['secrecy_score']:.2f}")
            print(f"  Stability: {metrics['stability_score']:.3f}")
            print(f"  Training Time: {training_time:.1f}s")
        
        all_results[batch_size] = batch_results
        
        # Print aggregate statistics
        print(f"\n{'='*80}")
        print(f"BATCH SIZE {batch_size} - AGGREGATE RESULTS")
        print(f"{'='*80}")
        comm_accs = [r['communication_accuracy'] for r in batch_results]
        eve_accs = [r['eve_success_rate'] for r in batch_results]
        secrecies = [r['secrecy_score'] for r in batch_results]
        stabilities = [r['stability_score'] for r in batch_results]
        
        print(f"Communication Accuracy: {np.mean(comm_accs):.2%} ± {np.std(comm_accs):.2%}")
        print(f"Eve Success Rate: {np.mean(eve_accs):.2%} ± {np.std(eve_accs):.2%}")
        print(f"Secrecy Score: {np.mean(secrecies):.2f} ± {np.std(secrecies):.2f}")
        print(f"Stability Score: {np.mean(stabilities):.3f} ± {np.std(stabilities):.3f}")
    
    # Save results
    with open('batch_experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print("Results saved to: batch_experiment_results.json")
    
    # Print recommendations
    print_recommendations(all_results)
    
    return all_results


def print_recommendations(all_results):
    """Print recommendations based on results."""
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    batch_sizes = list(all_results.keys())
    
    # Best for communication accuracy
    comm_scores = {bs: np.mean([r['communication_accuracy'] for r in all_results[bs]]) 
                   for bs in batch_sizes}
    best_comm = max(comm_scores, key=comm_scores.get)
    print(f"Best Communication Accuracy: Batch Size {best_comm} ({comm_scores[best_comm]:.2%})")
    
    # Best for security
    secrecy_scores = {bs: np.mean([r['secrecy_score'] for r in all_results[bs]]) 
                     for bs in batch_sizes}
    best_secrecy = max(secrecy_scores, key=secrecy_scores.get)
    print(f"Best Security (Secrecy): Batch Size {best_secrecy} ({secrecy_scores[best_secrecy]:.2f})")
    
    # Best for stability
    stability_scores = {bs: np.mean([r['stability_score'] for r in all_results[bs]]) 
                       for bs in batch_sizes}
    best_stability = max(stability_scores, key=stability_scores.get)
    print(f"Best Stability: Batch Size {best_stability} ({stability_scores[best_stability]:.3f})")
    
    # Most reproducible
    repro_scores = {bs: 1.0 - np.std([r['communication_accuracy'] for r in all_results[bs]]) 
                   for bs in batch_sizes}
    best_repro = max(repro_scores, key=repro_scores.get)
    print(f"Most Reproducible: Batch Size {best_repro}")
    
    print(f"\nOverall Recommendation: Batch Size {best_comm} offers the best balance.")


if __name__ == "__main__":
    # Run quick test (2 batch sizes, 2 runs, 50 iterations)
    # Uncomment for full experiment:
    # run_batch_experiment([512, 1024, 2048, 4096], num_runs=3, num_iterations=100)
    
    # Quick test (faster):
    run_batch_experiment([512, 1024, 2048, 4096], num_runs=3, num_iterations=100)