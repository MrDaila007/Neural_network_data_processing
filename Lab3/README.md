# Lab 3: Radial Basis Function Network

## Objective

Explore the architecture and training of an RBF network. Implement a program that recognizes noisy patterns.

## Theory

An RBF network is similar to an MLP but trains faster. The hidden layer uses Gaussian activation functions.

### Network architecture

```
    Input layer         RBF layer             Output layer
    (36 inputs)         (5 cells)            (5 neurons)
    ┌───┐               ┌───────┐           ┌───┐
    │x₁ │──────────────►│ RBF₁  │──────────►│y₁ │ → Class N
    └───┘               │exp(-d²)│           └───┘
    ┌───┐               └───────┘           ┌───┐
    │x₂ │──────────────►│ RBF₂  │──────────►│y₂ │ → Class F
    └───┘               │exp(-d²)│           └───┘
      ⋮                    ⋮                   ⋮
    ┌───┐               ┌───────┐           ┌───┐
    │x₃₆│──────────────►│ RBF₅  │──────────►│y₅ │ → Class D
    └───┘               │exp(-d²)│           └───┘
```

### Key formulas

**Gaussian activation (RBF output):**
```
g_j = exp(-||x - t_j||² / σ_j²)
```

**Output layer:**
```
y_k = Σ w_jk × g_j
```

**Output weight update:**
```
w_jk := w_jk + α × (y_k^target - y_k) × g_j
```

## Assignment variant

**Variant 2:** Classes **N**, **F**, **I**, **P**, **D** (size 6×6)

## Requirements

- C++ compiler with C++17 support (g++ 8+, clang++ 7+)
- Standard C++ library (filesystem)

## Build & Run

### 1. Compile

```bash
cd Lab3
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
Lab3/
├── README.md           # This file
├── description.md      # Assignment description
├── solution.cpp        # Source code
├── solution            # Compiled binary
├── report.tex          # LaTeX report
├── report.pdf          # Compiled report
├── output.txt          # Console log (if redirected)
├── patterns/           # Reference patterns
│   ├── N.txt           # Letter N
│   ├── F.txt           # Letter F
│   ├── I.txt           # Letter I
│   ├── P.txt           # Letter P
│   └── D.txt           # Letter D
└── tests/
    ├── N/
    │   ├── noise_10/   # 10% noise
    │   ├── noise_20/   # 20% noise
    │   └── ...
    ├── F/
    ├── I/
    ├── P/
    └── D/
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
    LAB 3: RADIAL BASIS FUNCTION NETWORK
    Variant 2: classes N, F, I, P, D
========================================================

1. Loading reference patterns from patterns/...

Ideal training patterns (6x6):
-----------------------------------
Class 1 (N):
  ■ □ □ □ □ ■
  ■ ■ □ □ □ ■
  ...

2. Initializing RBF cells...
   Total cells: 5
   Class 1 (N): sigma = 2.449
   Class 2 (F): sigma = 2.449
   ...

3. Training the output layer via gradient descent...
   Training completed in 87 steps

┌─────────────────────────────────────────────────┐
│ Recognized pattern (6×6):                     │
│   ■ □ □ □ □ ■                                   │
│   ■ ■ □ □ □ ■                                   │
├─────────────────────────────────────────────────┤
│ Similarity scores (RBF outputs):               │
│   Class 1 (N): 85.2%  ◄── Recognized as "N"     │
│   Class 2 (F): 3.1%                             │
│   Class 3 (I): 5.4%                             │
│   Class 4 (P): 4.8%                             │
│   Class 5 (D): 1.5%                             │
└─────────────────────────────────────────────────┘
```

## Network parameters

| Parameter | Value |
|----------|-------|
| Input neurons | 36 (6×6) |
| RBF cells | 5 (one per class) |
| Output neurons | 5 |
| Learning rate α | 0.1 |
| σ (sigma) | Automatic (half the distance to the nearest center) |

## Comparison with MLP

| Characteristic | MLP | RBF |
|----------------|-----|-----|
| Training speed | Slow | Fast |
| Trainable layers | 2 | 1 (output only) |
| Accuracy | High | Limited |

## Conclusions

1. RBF trains much faster than the MLP (under 100 steps)
2. Recognition quality is comparable to the MLP at low noise levels
3. Suitable for datasets with well-clustered classes
