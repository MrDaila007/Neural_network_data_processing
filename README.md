# Neural Network Data Processing

Laboratory assignments for the course **"Neural Networks for Data Processing"**.

## Contents

| Lab | Topic | Description |
|-----|-------|-------------|
| [Lab1](Lab1/) | Hopfield Neural Network | Auto-associative network with feedback for pattern recognition |
| [Lab2](Lab2/) | Multilayer Perceptron | Feedforward network with backpropagation algorithm |
| [Lab3](Lab3/) | RBF Network | Radial Basis Functions for classification |
| [Lab4](Lab4/) | Competitive Neural Network | Self-organizing network for clustering |

## Project Structure

```
Neural_network_data_processing/
├── Doc/
│   └── Theory.md              # General theory on neural networks
├── Lab1/
│   ├── description.md         # Lab assignment description
│   ├── solution.cpp           # C++ solution
│   ├── report.tex             # LaTeX report
│   └── README.md              # Build and run instructions
├── Lab2/
│   ├── description.md
│   ├── solution.cpp
│   ├── report.tex
│   └── README.md
├── Lab3/
│   ├── description.md
│   ├── solution.cpp
│   ├── report.tex
│   └── README.md
├── Lab4/
│   ├── description.md
│   ├── solution.cpp
│   ├── report.tex
│   └── README.md
└── README.md
```

## Neural Network Types

### Hopfield Network (Lab1)
- **Type:** Auto-associative with feedback connections
- **Training:** Unsupervised (Hebbian learning rule)
- **Applications:** Associative memory, noisy pattern recognition

### Multilayer Perceptron (Lab2)
- **Type:** Feedforward
- **Training:** Supervised (backpropagation)
- **Applications:** Classification, function approximation

### RBF Network (Lab3)
- **Type:** Feedforward with radial basis functions
- **Training:** Hybrid (clustering + gradient descent)
- **Applications:** Classification with well-clustered data

### Competitive Network (Lab4)
- **Type:** Self-organizing
- **Training:** Unsupervised (competitive learning)
- **Applications:** Clustering, data compression

## Requirements

- **Compiler:** g++ with C++17 support
- **LaTeX:** XeLaTeX for report compilation
- **Pattern sizes:** 
  - Lab1: 10×10 (binary/bipolar)
  - Lab2-4: 6×6

## Quick Start

```bash
# Build and run any lab
cd Lab1
g++ -std=c++17 -O2 solution.cpp -o solution
./solution

# Compile LaTeX report
xelatex report.tex
```

## References

1. **Aleksander I., Morton H.** An Introduction to Neural Computing. — London: Chapman & Hall, 1990.

2. **Bishop C.M.** Neural Networks for Pattern Recognition. — Oxford: Clarendon Press, 1995. — 482 p.

3. **Hopfield J.J.** Neural networks and physical systems with emergent collective computational abilities // Proc. Natl. Acad. Sci. USA. — 1982. — Vol. 79. — P. 2554.

4. **Kohonen T.** Self-organization and associative memory. — Springer-Verlag, 1989. — 312 p.

5. **Kohonen T.** Self-organizing maps. — Springer-Verlag, 1995. — 362 p.

6. **Rumelhart D.E., Hinton G.E., Williams R.J.** Learning internal representation by error propagation. Parallel Distributed Processing. — MIT Press, 1986. — Vol. 1. — P. 318-362.

7. **Haykin S.** Neural Networks: A Comprehensive Foundation, 2nd ed. — Prentice Hall, 1998.

## License

MIT License
