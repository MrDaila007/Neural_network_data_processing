# Lab 2: Multilayer Perceptron

## Objective

Study the structure and learning algorithm of a multilayer perceptron. Implement a program that recognizes noisy patterns using backpropagation.

## Theory

A multilayer perceptron is a feedforward network trained with supervised learning. It can approximate any continuous function given sufficient capacity.

### Network architecture

```
    Input layer         Hidden layer         Output layer
    (36 neurons)        (20 neurons)        (5 neurons)
    ┌───┐               ┌───┐               ┌───┐
    │x₁ │──────────────►│h₁ │──────────────►│y₁ │ → Class N
    └───┘               └───┘               └───┘
    ┌───┐               ┌───┐               ┌───┐
    │x₂ │──────────────►│h₂ │──────────────►│y₂ │ → Class F
    └───┘               └───┘               └───┘
      ⋮                   ⋮                   ⋮
    ┌───┐               ┌───┐               ┌───┐
    │x₃₆│──────────────►│h₂₀│──────────────►│y₅ │ → Class D
    └───┘               └───┘               └───┘
```

### Key formulas

**Sigmoid activation:**
```
f(x) = 1 / (1 + e^(-x))
```

**Backpropagation update:**
```
δ_k = (y_k^target - y_k) × f'(S_k)
w_jk := w_jk + α × δ_k × h_j
```

## Assignment variant

**Variant 2:** Letters **N**, **F**, **I**, **P**, **D** (size 6×6)

## Requirements

- C++ compiler with C++17 support (g++ 8+, clang++ 7+)
- Standard C++ library (filesystem)

## Build & Run

### 1. Compile

```bash
cd Lab2
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
Lab2/
├── README.md           # This file
├── description.md      # Assignment description
├── solution.cpp        # Source code
├── solution            # Compiled binary
├── report.tex          # LaTeX report
├── report.pdf          # Compiled report
├── output.txt          # Console log (if redirected)
└── patterns/           # Reference patterns
    ├── N.txt           # Letter N
    ├── F.txt           # Letter F
    ├── I.txt           # Letter I
    ├── P.txt           # Letter P
    └── D.txt           # Letter D
```

## Output files

After running the program, the following file is created:

| File | Description |
|------|-------------|
| `output.txt` | Full console log (when `> output.txt` is used) |

## Sample output

```
========================================================
    LAB 2: MULTILAYER PERCEPTRON
    Variant 2: letters N, F, I, P, D
========================================================

1. Loading reference patterns from patterns/...
   5 patterns loaded

2. Building the training set...
   5 training examples created

3. Training the network...
   Training completed in 1542 epochs

4. Testing on clean patterns:
======================================

┌─────────────────────────────────┐
│ Recognized pattern (6×6):       │
│   ■ □ □ □ □ ■                  │
│   ■ □ □ □ □ ■                  │
│   ■ ■ □ □ □ ■                  │
│   ■ □ ■ □ □ ■                  │
│   ■ □ □ ■ □ ■                  │
│   ■ □ □ □ ■ ■                  │
├─────────────────────────────────┤
│ Similarity scores:             │
│   Class 1 (N): 99.2%  ◄── Detected
│   Class 2 (F): 0.1%            │
│   Class 3 (I): 0.3%            │
│   Class 4 (P): 0.2%            │
│   Class 5 (D): 0.2%            │
└─────────────────────────────────┘
```

## Network parameters

| Parameter | Value |
|----------|-------|
| Input neurons | 36 (6×6) |
| Hidden neurons | 20 |
| Output neurons | 5 |
| Learning rate α | 0.5 |
| Learning rate β | 0.5 |
| Max error | 0.01 |

## Conclusions

1. The MLP successfully learns five pattern classes
2. It recognizes noisy inputs up to ~30-40% noise
3. The similarity score demonstrates the confidence for each class
4. Training takes 500-3000 epochs depending on initialization
