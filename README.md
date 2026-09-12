<div align="center">

# mydl

### A Deep Learning Library Built from Scratch Using NumPy

*Learn how modern neural networks work by building every component from first principles.*

---

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![NumPy](https://img.shields.io/badge/NumPy-2.x-green)
![Tests](https://img.shields.io/badge/Tests-Passing-success)
![License](https://img.shields.io/badge/License-MIT-orange)

---

*A personal educational project focused on understanding deep learning implementations rather than using existing frameworks.*

</div>

---

# Overview

**mydl** is an educational deep learning library implemented entirely with **NumPy**.

The objective of this project is **not** to compete with TensorFlow or PyTorch.

Instead, the goal is to understand how modern deep learning libraries work internally by implementing every algorithm from scratch.

Every important component—including forward propagation, backpropagation, gradient descent, neural network layers, activation functions, and loss functions—is written manually using only NumPy.

This repository is being developed while studying the **Deep Learning Specialization by Andrew Ng**, with each newly learned concept implemented from first principles before moving on to the next.

---

# Features

## Models

- Logistic Regression
- Two-Layer Neural Network
- Deep L-Layer Neural Network

---

## Core Modules

- Activation Functions
- Loss Functions
- Linear Layers
- Parameter Initializers
- Evaluation Metrics

---

## Implemented Algorithms

- Forward Propagation
- Backpropagation
- Gradient Descent
- Binary Cross Entropy
- He Initialization
- Vectorized Computation

---

# Architecture

```text
                                   mydl
                                     │
     ┌───────────────────────────────┼───────────────────────────────┐
     │                               │                               │
 Models                         Core Components                 Examples
     │                               │                               │
     │                               │                               │
 ┌───┼──────────────┐        ┌────────┼──────────┐               Training Scripts
 │   │              │        │        │          │
 │   │              │        │        │          │
Logistic        TwoLayer   Activations Layers   Losses
Regression          NN         │         │         │
                               │         │         │
                         Initializers  Metrics  Utilities
                               │
                               │
                         NumPy Operations
```

---

# Project Structure

```text
mydl/
│
├── src/
│   └── mydl/
│       ├── activations.py
│       ├── initializers.py
│       ├── layers.py
│       ├── losses.py
│       ├── metrics.py
│       │
│       └── models/
│           ├── logistic_regression.py
│           ├── two_layer_nn.py
│           └── l_layer_nn.py
│
├── examples/
│
├── tests/
│
├── docs/
│
├── pyproject.toml
│
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/mydl.git
```

Move into the project

```bash
cd mydl
```

Install the package

```bash
pip install -e .
```

---

# Quick Start

```python
from mydl.models import LLayerNN

model = LLayerNN(
    hidden_layer=[32, 16, 8],
    learning_rate=0.01,
    epochs=1000,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")
```

---

# Example Results

The models have been evaluated on the Breast Cancer Wisconsin dataset.

| Model | Test Accuracy |
|--------|--------------:|
| Logistic Regression | ~96% |
| TwoLayerNN | ~95% |
| LLayerNN | ~95% |
| Scikit-Learn MLPClassifier | ~96% |

---

# Testing

Run the complete test suite

```bash
pytest -v
```

Current test coverage

- Activation Functions
- Initializers
- Layers
- Loss Functions
- Metrics
- Logistic Regression
- Two-Layer Neural Network
- Deep Neural Network

All tests currently pass successfully.

---

# Learning Roadmap

## Completed

- [x] Logistic Regression
- [x] Forward Propagation
- [x] Backpropagation
- [x] Two-Layer Neural Network
- [x] Deep Neural Network
- [x] He Initialization
- [x] Comprehensive Unit Tests
- [x] Example Scripts

---

## Coming Next

### Optimization

- [ ] Xavier Initialization
- [ ] Mini-Batch Gradient Descent
- [ ] Momentum
- [ ] RMSProp
- [ ] Adam Optimizer

### Regularization

- [ ] L2 Regularization
- [ ] Dropout
- [ ] Batch Normalization

### Additional Models

- [ ] Softmax Classifier
- [ ] CNN
- [ ] RNN
- [ ] Transformer (Educational)

---

# Design Philosophy

The emphasis of this project is **clarity over abstraction**.

Rather than hiding mathematical details behind high-level APIs, every component is implemented explicitly so that readers can understand exactly how deep learning algorithms work internally.

The code is intentionally modular, heavily tested, and designed to be easy to extend as new concepts are learned.

---

# Why Build Yet Another Deep Learning Library?

Because implementing algorithms from scratch is one of the best ways to understand them.

Reading equations is valuable.

Using TensorFlow is valuable.

Building the algorithms yourself bridges the gap between theory and practice.

This repository documents that journey.

---

# Acknowledgements

This project is inspired by

- Andrew Ng's Deep Learning Specialization
- CS231n (Stanford University)
- NumPy
- Scikit-Learn

---

# Contributing

Suggestions, improvements, bug reports, and pull requests are welcome.

If you notice an issue or have an idea for improving the project, feel free to open an issue.

---

# License

