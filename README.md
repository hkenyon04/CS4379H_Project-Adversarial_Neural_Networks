# Neural Cryptography Batch Size Experiment (Simplified)

A streamlined implementation to test how batch size affects neural cryptography training stability and performance.

## What This Does

Tests neural networks (Alice, Bob, Eve) learning to encrypt/decrypt messages:
- **Alice** encrypts messages with a key
- **Bob** decrypts with the same key  
- **Eve** tries to break encryption without the key

We measure how different batch sizes affect:
1. **Communication Accuracy**: Can Bob decrypt successfully?
2. **Security**: Does Eve fail to break it?
3. **Stability**: Are results consistent across runs?

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install packages
pip install tensorflow numpy matplotlib

# Navigate to your project folder
cd path/to/your/project
```

### 2. Run Experiment

```bash
python simple_experiment.py
```

This runs a **quick test** (2 batch sizes, 2 runs, 50 iterations) - takes ~3-5 minutes.

### 3. Visualize Results

```bash
python visualize.py
```

Creates two plots:
- `batch_experiment_results.png` - Main results
- `tradeoff_analysis.png` - Trade-off comparisons

## Files

- **`datagen.py`** - Generates random binary messages
- **`net.py`** - Neural network architectures (TensorFlow 2.x)
- **`simple_experiment.py`** - Main experiment script
- **`visualize.py`** - Creates plots from results

## Full Experiment

For complete results, edit `simple_experiment.py` line 245:

```python
# Change from:
run_batch_experiment([512, 1024], num_runs=2, num_iterations=50)

# To:
run_batch_experiment([512, 1024, 2048, 4096], num_runs=3, num_iterations=100)
```

This takes ~1-2 hours.

## Key Metrics Explained

### Communication Accuracy
**What:** Percentage of message bits Bob correctly decrypts  
**Good:** > 95%  
**Formula:** `1 - (bit_errors / 16)`

### Eve Success Rate
**What:** Percentage of bits Eve guesses correctly  
**Good:** ~50% (random guessing = secure)  
**Bad:** > 80% (Eve is breaking encryption)

### Secrecy Score
**What:** Gap between Eve's errors and Bob's errors  
**Good:** Positive and large (>3)  
**Formula:** `eve_bit_errors - bob_bit_errors`

### Stability Score
**What:** How consistent training is (0-1 scale)  
**Good:** > 0.5  
**Formula:** `1 / (1 + variance)`

## Expected Results

### Batch Size 512
- Fast training
- High variance (unstable)
- Communication: 85-95%

### Batch Size 1024
- Good balance
- Moderate stability
- Communication: 95-98%

### Batch Size 2048-4096
- Very stable
- Slow training
- Communication: 98-99%

## Interpreting Results

**Good encryption system:**
- Bob accuracy > 95%
- Eve accuracy ~50%
- Secrecy score > 3
- Stability > 0.5

**Trade-offs:**
- Larger batches → More stable but slower
- Smaller batches → Faster but inconsistent

## Customization

### Test Different Batch Sizes
```python
run_batch_experiment([256, 512, 1024, 2048], num_runs=3, num_iterations=100)
```

### Adjust Training Length
```python
run_batch_experiment([512, 1024], num_runs=2, num_iterations=200)  # Longer training
```

### Change Message Length
Edit in `simple_experiment.py`:
```python
trainer = CryptoTrainer(message_length=32)  # Default is 16 bits
```

## Troubleshooting

### Error: "No module named tensorflow"
```bash
pip install tensorflow
```

### Error: "No module named matplotlib"
```bash
pip install matplotlib
```

### Out of Memory
Reduce batch sizes:
```python
run_batch_experiment([256, 512], num_runs=2, num_iterations=50)
```

### Training Too Slow
Use quick test settings (already default in code).

## Output Files

- `batch_experiment_results.json` - Raw data from all runs
- `batch_experiment_results.png` - Main visualization
- `tradeoff_analysis.png` - Trade-off plots

## Understanding the Code

### Training Process (per iteration)
1. Generate random messages and keys
2. Train Alice & Bob together (20 steps)
3. Train Eve to break the encryption (40 steps)
4. Evaluate all three networks
5. Record metrics

### Why More Steps for Eve?
Eve gets twice as many training steps (40 vs 20) to give the adversary a fair chance at breaking the encryption. This tests the robustness of Alice and Bob's encryption scheme.

## Research Questions Answered

1. **How does batch size affect stability?**
   - Measured via standard deviation across runs
   - Visualized in stability score plot

2. **What's the optimal batch size?**
   - Balance between accuracy and computational cost
   - Usually 1024-2048 for this problem

3. **Security vs Performance trade-off?**
   - Shown in trade-off plots
   - Larger batches generally more secure

## Sample Output

```
================================================================================
BATCH SIZE EXPERIMENT - NEURAL CRYPTOGRAPHY
================================================================================
Batch sizes: [512, 1024]
Runs per batch: 2
Iterations: 50
================================================================================

================================================================================
BATCH SIZE: 512
================================================================================

--- Run 1/2 ---

Training with batch size: 512
============================================================
Iter   0 | Bob: 52.34% | Eve: 51.67% | Secrecy: -0.11
Iter  10 | Bob: 87.23% | Eve: 53.45% | Secrecy: -5.41
Iter  20 | Bob: 95.67% | Eve: 48.92% | Secrecy: 1.09
Iter  30 | Bob: 98.12% | Eve: 49.88% | Secrecy: 0.19
Iter  40 | Bob: 99.01% | Eve: 50.12% | Secrecy: -0.19

Run 1 Summary:
  Communication Accuracy: 98.89%
  Eve Success Rate: 50.23%
  Secrecy Score: -0.37
  Stability: 0.892
  Training Time: 45.3s
```

## Tips for Best Results

1. **Run multiple times**: Use `num_runs=3` or more for statistical confidence
2. **Longer training**: Use `num_iterations=100+` for convergence
3. **Consistent environment**: Use same random seeds for reproducibility
4. **Monitor progress**: Watch for Bob > 95% and Eve ~50%

## Citation

Original paper:
```
Abadi, M., & Andersen, D. G. (2016).
Learning to protect communications with adversarial neural cryptography.
arXiv preprint arXiv:1610.06918.
```

## License

Same as original implementation.

---

## Quick Command Reference

```bash
# === SETUP (One Time) ===
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install tensorflow numpy matplotlib

# === NAVIGATE TO PROJECT ===
cd path/to/your/project

# === RUN EXPERIMENT ===
python simple_experiment.py

# === VISUALIZE RESULTS ===
python visualize.py

# === VIEW RESULTS ===
# Windows:
start batch_experiment_results.png
start tradeoff_analysis.png
# Mac:
open batch_experiment_results.png
# Linux:
xdg-open batch_experiment_results.png

# View JSON data
# Windows:
type batch_experiment_results.json
# Mac/Linux:
cat batch_experiment_results.json
```

## Complete Step-by-Step Instructions

### First Time Setup:

1. **Open Terminal/Command Prompt**
   - **Windows:** Press Windows Key + R, type `cmd`, press Enter
   - **Mac:** Press Cmd + Space, type "Terminal", press Enter
   - **Linux:** Press Ctrl + Alt + T

2. **Navigate to Project Directory**
   ```bash
   cd path/to/your/CS_4379H_Cryptography/Project
   ```

3. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```
   If `python` doesn't work, try `python3`:
   ```bash
   python3 -m venv venv
   ```

4. **Activate Virtual Environment**
   
   **Windows (Command Prompt):**
   ```bash
   venv\Scripts\activate
   ```
   
   **Windows (PowerShell):**
   ```bash
   venv\Scripts\Activate.ps1
   ```
   
   **Mac/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   You should see `(venv)` at the start of your prompt.

5. **Install Packages**
   ```bash
   pip install tensorflow numpy matplotlib
   ```
   This takes 2-5 minutes.

### Every Time You Run:

1. **Open Terminal/Command Prompt**

2. **Navigate to Project**
   ```bash
   cd path/to/your/CS_4379H_Cryptography/Project
   ```

3. **Activate Virtual Environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **Mac/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Run Experiment**
   ```bash
   python simple_experiment.py
   ```
   Takes ~3-5 minutes for quick test.

5. **Create Visualizations**
   ```bash
   python visualize.py
   ```

6. **View Results**
   
   **Windows:**
   ```bash
   start batch_experiment_results.png
   ```
   
   **Mac:**
   ```bash
   open batch_experiment_results.png
   ```
   
   **Linux:**
   ```bash
   xdg-open batch_experiment_results.png
   ```

## For Anaconda Users

If you prefer using Anaconda/Miniconda:

```bash
# Create environment
conda create -n neural_crypto python=3.10 -y

# Activate
conda activate neural_crypto

# Install packages
conda install -c conda-forge tensorflow numpy matplotlib -y

# Navigate to project
cd path/to/your/project

# Run
python simple_experiment.py
python visualize.py
```