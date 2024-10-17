# IntelliLab Neural Network Framework 🧠🚀

Welcome to **IntelliLab**! This project was born out of a desire to deeply understand the inner workings of neural networks. Instead of relying on high-level, abstract libraries like TensorFlow or PyTorch, I decided to create my own lightweight framework with just a few basic yet fundamental features.

The goal of IntelliLab is to **demystify** how neural networks are built and trained. By stripping away the layers of abstraction provided by larger libraries, this project brings you closer to the underlying mechanics, allowing you to truly grasp what happens behind the scenes during forward propagation, backpropagation, and learning rate adjustments. 

## Folder Structure 📂

```
IntelliLab/
├── IntelliLab/
│   ├── learning/
│   │   └── FloatingRate.py   # Contains custom learning rate strategies
│   ├── models/
│   │   ├── Layer.py          # Implements a single neural network layer with various activation functions
│   │   └── Sequential.py     # Implements the Sequential neural network model
└── README.md                 # Project documentation
```

## Overview 🌟

The **IntelliLab** framework is designed to help you:
1. **Build** neural networks from scratch.
2. **Understand** how learning rate strategies impact training.
3. **Explore** core components like forward and backward propagation.
4. **Learn** the mechanics behind activation functions and optimization without relying on extensive, abstract libraries.

---

## Components Breakdown 🔍

### 1. FloatingRate.py 📉📈
The `FloatingRate` class provides dynamic learning rate functions. It allows you to explore the impact of different learning rate strategies on the training process, which are typically hidden behind abstractions in larger libraries.

### 2. Layer.py 🧩
The `Layer` class defines individual neural network layers, where you can experiment with different activation functions like ReLU, Sigmoid, and Tanh, and understand how they influence learning.

### 3. Sequential.py 🔗
The `Sequential` class combines multiple layers to form a full neural network, enabling you to perform forward propagation, backpropagation, and track the learning process.

---

## Features 🚀
- **Custom Learning Rates**: Understand how learning rates evolve during training.
- **Layer-by-Layer Control**: Manually define and adjust layers for full customization.
- **Transparent Backpropagation**: See exactly how gradients are calculated and applied.
- **Minimalistic Design**: Focus on core neural network concepts without unnecessary complexity.

---

## Installation & Setup 🔧

1. Clone the repository:
```bash
git clone https://github.com/your-repo/IntelliLab.git
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

---

## Why Use IntelliLab? 🤔
If you want to:
- **Build** a neural network from scratch without relying on high-level libraries.
- **Understand** each component of the learning process.
- **Learn** by doing, rather than relying on black-box abstractions.

Then IntelliLab is for you! This project is designed to give you control and insight into how neural networks function at their core.
