<div align="center">

# mydl

### A Deep Learning Library Built from Scratch with NumPy

Build and understand neural networks by implementing their core mathematics from first principles.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-green)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-89%20Passing-success)](#testing)
[![License](https://img.shields.io/badge/License-MIT-orange)](#license)

</div>

---

## Overview

**mydl** is a small educational deep learning library implemented from scratch using **NumPy**.

The project focuses on making the mathematics and internal mechanics of neural networks explicit instead of hiding them behind high-level deep learning frameworks.

The library includes:

- Logistic Regression
- Two-Layer Neural Networks
- L-Layer Neural Networks
- Forward and backward propagation
- Mini-batch training
- Multiple optimization algorithms
- L1, L2, and combined L1/L2 regularization
- Dropout
- Training and validation history
- Binary classification metrics
- Training-history visualization

The project was built alongside the **Deep Learning Specialization by Andrew Ng**, with an emphasis on translating the mathematical concepts into working implementations.

---

## Features

### Models

- Logistic Regression
- Two-Layer Neural Network
- L-Layer Neural Network

### Neural Network Components

- ReLU activation
- Sigmoid activation
- Binary Cross-Entropy loss
- Linear layers
- He initialization
- Vectorized NumPy computation
- Dropout

### Optimizers

- Gradient Descent
- Momentum
- RMSProp
- Adam

### Regularization

- L1 Regularization
- L2 Regularization
- Combined L1 + L2 Regularization

### Training and Evaluation

- Mini-batch training
- Training loss history
- Training accuracy history
- Validation loss
- Validation accuracy
- Configurable classification threshold
- Accuracy
- Precision
- Recall
- F1 Score
- Binary confusion matrix

### Visualization

- Training loss vs. epoch
- Validation loss vs. epoch
- Training accuracy vs. epoch
- Validation accuracy vs. epoch

---

# Installation

## Requirements

- Python 3.11+
- NumPy 2.x

Clone the repository:

```bash
git clone https://github.com/Anujpandey91/Neural-Network-implementation-in-Python-.git