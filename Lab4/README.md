# Lab 4: Competitive Neural Network

## Objective

Learn the topology and learning procedure of a competitive neural network. Implement a program that clusters and recognizes mathematical symbols.

## Theory

A competitive network is a simple self-organizing system trained without supervision. It naturally clusters the input space.

### Network architecture

```
    Input layer         Competitive layer
    (36 neurons)        (5 neurons)
    ┌───┐               ┌───┐
    │x₁ │──────────────►│ 1 │ → Cluster 1 (≤)
    └───┘               └───┘
    ┌───┐               ┌───┐
    │x₂ │──────────────►│ 2 │ → Cluster 2 (≥)
    └───┘               └───┘
      ⋮                   ⋮
    ┌───┐               ┌───┐
    │x₃₆│──────────────►│ 5 │ → Cluster 5 (≅)
    └───┘               └───┘
         Winner: maximum dot product
```

### Key formulas

**Neuron output:**
```
y_j = Σ w_ij × x_i = ||w_j|| × ||x|| × cos(α)
```

**Winner weight update:**
```
w_ij := w_ij + β × (x_i - w_ij)
```

**Normalization:**
```
w_ij := w_ij / ||w_j||
```

**Frequency-dependent criterion:**
```
winner = argmin_j (||x - w_j|| × (1 + f_j))
```

## Assignment variant

**Variant 5:** Symbols **≤**, **≥**, **≠**, **≈**, **≅** (size 6×6)

## Requirements

- C++ compiler with C++17 support (g++ 8+, clang++ 7+)
- Standard C++ library (filesystem)

## Build & Run

### 1. Compile

```bash
cd Lab4
g++ -std=c++17 solution.cpp -o solution
```

### 2. Run with console output

```bash
./solution
```

### 3. Redirect output to files

```bash
# Save output only to a file
./solution > output.txt

# Save output and show it in the console
./solution | tee output.txt

# Save stdout and stderr
./solution > output.txt 2>&1
```

### 4. Compile the LaTeX report

```bash
xelatex report.tex
```

XeLaTeX and DejaVu fonts are required.

## File layout

```
Lab4/
├── README.md           # This file
├── description.md      # Assignment description
├── solution.cpp        # Source code
├── solution            # Compiled binary
├── report.tex          # LaTeX report
├── report.pdf          # Compiled report
├── output.txt          # Console log (if redirected)
├── patterns/           # Reference patterns
│   ├── LE.txt          # Symbol ≤
│   ├── GE.txt          # Symbol ≥
│   ├── NE.txt          # Symbol ≠
│   ├── AP.txt          # Symbol ≈
│   └── CO.txt          # Symbol ≅
└── tests/              # Noisy samples per class
    ├── LE/
    │   ├── noise_10/   # 10% noise
    │   ├── noise_20/   # 20% noise
    │   └── ...
    ├── GE/
    ├── NE/
    ├── AP/
    └── CO/
```

## Output files

After running the program, the following files are created:

| File | Description |
|------|-------------|
| `tests/*/noise_*/test_*.txt` | Generated noisy samples |
| `output.txt` | Full console log (when `> output.txt` is used) |

## Sample output

```
========================================================
    LAB 4: COMPETITIVE NEURAL NETWORK
    Variant 5: symbols ≤, ≥, ≠, ≈, ≅
========================================================

1. Loading reference patterns from patterns/...
   15 training samples loaded

Reference patterns (6x6):
------------------------
Class ≤ (LE):
  □ □ □ □ ■ ■ 
  □ □ □ ■ ■ □ 
  □ □ ■ ■ □ □ 
  □ □ ■ ■ □ □ 
  □ □ □ ■ ■ □ 
  ■ ■ ■ ■ ■ ■ 

3. Training the network...
  Training converged at iteration 23 (max distance: 0.0089)

4. Assigning neurons to classes:
----------------------------------------------
Class ≤ → Winner neuron: #0
Class ≥ → Winner neuron: #1
Class ≠ → Winner neuron: #2
Class ≈ → Winner neuron: #3
Class ≅ → Winner neuron: #4

8. Testing noisy samples:
=======================================
┌────────┬───────┬──────────────┬──────────────┐
│ Class  │ Noise │ Correct      │ Incorrect    │
├────────┼───────┼──────────────┼──────────────┤
│   ≤    │   10  │     10/10    │       0/10    │
│   ≤    │   20  │      9/10    │       1/10    │
│   ≤    │   30  │      8/10    │       2/10    │
│   ≤    │   40  │      7/10    │       3/10    │
│   ≤    │   50  │      5/10    │       5/10    │
... (remaining rows omitted)
└────────┴───────┴──────────────┴──────────────┘
```

## Network parameters

| Parameter | Value |
|----------|-------|
| Input neurons | 36 (6×6) |
| Competitive neurons | 5 |
| Learning rate β | 0.1 |
| Max distance | 0.01 |
| Training samples | 15 (three per class) |

## Test results

| Noise level | Recognition accuracy |
|-------------|----------------------|
| 10%         | 100%                 |
| 20%         | 90%                  |
| 30%         | 80%                  |
| 40%         | 64%                  |
| 50%         | 52%                  |

## Comparison with other networks

| Characteristic | MLP | RBF | Competitive |
|----------------|-----|-----|-------------|
| Learning mode | Supervised | Supervised | Unsupervised |
| Training speed | Slow | Fast | Fast |
| Intended use | Classification | Classification | Clustering |

## Conclusions

1. The competitive network clusters symbols without supervision
2. Frequency-dependent learning prevents dead neurons
3. Accuracy remains ≥90% up to 20% noise
4. Suitable for clustering tasks without labeled data
