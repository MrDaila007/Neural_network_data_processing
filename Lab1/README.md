# Lab 1: Hopfield Neural Network

## Objective

Understand the topology and dynamics of a Hopfield neural network. Implement a program that recognizes noisy patterns of Russian letters.

## Theory

A Hopfield network is a single-layer, symmetric, nonlinear auto-associative memory that stores binary or bipolar patterns.

### Network architecture

```
    ┌─────────────────────────────────────────────┐
    │                                             │
    │   ┌───┐         ┌───┐         ┌───┐        │
    │   │a₁ │◄───────►│a₂ │◄───────►│a₃ │        │
    │   └───┘         └───┘         └───┘        │
    │     ▲             ▲             ▲          │
    │     │             │             │          │
    │     └─────────────┴─────────────┘          │
    │           Fully connected network          │
    │          (every neuron connected)          │
    │                                             │
    │               100 neurons                  │
    │             (10×10 patterns)               │
    └─────────────────────────────────────────────┘
```

### Key formulas

**Training (Hebbian rule):**
```
w_ij = Σ(k=1 to m) a_i^k × a_j^k,  i ≠ j
w_ii = 0
```

**Recall:**
```
a_i(t+1) = sign(Σ(j=1 to n) w_ij × a_j(t))
```

## Assignment variant

**Variant 5:** Letters **D**, **N**, **X** (size 10×10)

## Requirements

- C++ compiler with C++17 support (g++ 8+, clang++ 7+)
- Standard C++ library (filesystem)

## Build & Run

### 1. Compile

```bash
cd Lab1
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
Lab1/
├── README.md           # This file
├── description.md      # Assignment description
├── solution.cpp        # Source code
├── solution            # Compiled binary
├── report.tex          # LaTeX report
├── report.pdf          # Compiled report
├── output.txt          # Program logs (created when redirected)
├── patterns/           # Reference patterns
│   ├── D.txt           # Letter D
│   ├── N.txt           # Letter N
│   └── X.txt           # Letter X
└── tests/              # Noisy patterns generated for evaluation
    ├── results.csv     # Test summary (CSV)
    ├── D/
    │   ├── noise_10/   # 10% noise
    │   ├── noise_20/   # 20% noise
    │   └── ...
    ├── N/
    └── X/
```

## Output files

After running the program, the following files are generated:

| File | Description |
|------|-------------|
| `tests/results.csv` | Evaluation results in CSV format |
| `tests/*/noise_*/test_*.txt` | Generated noisy samples |
| `output.txt` | Full console log (when `> output.txt` is used) |

## Sample output

```
========================================================
    LAB 1: HOPFIELD NEURAL NETWORK
    Variant 5: letters D, N, X
========================================================

1. Loading reference patterns from patterns/...

Reference patterns (10x10):
------------------------
Letter D:
      ############    
      ##        ##    
      ##        ##    
      ...

2. Training with the Hebbian rule...
   Weight matrix computed (100x100)

3. Generating test patterns with different noise levels...
   Test patterns saved to tests/

4. Recognition demo (30% noise):
   Recognized as: D
   Similarity to reference: 100.0%

5. Summary statistics by noise level:
┌───────┬─────────────────┬─────────────────┐
│ Noise │ Synchronous     │ Asynchronous    │
│   %   │ success rate %  │ success rate %  │
├───────┼─────────────────┼─────────────────┤
│   10  │      100.0%     │      100.0%     │
│   20  │      100.0%     │      100.0%     │
│   30  │      100.0%     │      100.0%     │
│   35  │      100.0%     │      100.0%     │
│   40  │       70.0%     │       70.0%     │
│   50  │       17.0%     │       27.0%     │
└───────┴─────────────────┴─────────────────┘
```

## Results

| Noise level | Synchronous mode | Asynchronous mode |
|-------------|------------------|-------------------|
| 10-35%      | 100%             | 100%              |
| 40%         | ~70%             | ~70%              |
| 50%         | ~17%             | ~27%              |
| 60%+        | ~0%              | ~0%               |

## Conclusions

1. The Hopfield network recognizes patterns reliably up to ~35% noise
2. Accuracy drops sharply beyond 40-50% noise
3. Asynchronous updates perform slightly better at high noise but require more iterations
4. Recommended to operate within 30-35% noise levels
