# Deep Learning Basics — From-Scratch Implementations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Educational-orange)](https://numpy.org/)

Educational implementations of foundational deep learning components using **only NumPy** to demonstrate core understanding of neural network fundamentals, backpropagation, and gradient descent.

## 🎯 Learning Objectives

This repository demonstrates deep understanding of:
- **Backpropagation**: Computing gradients using the chain rule
- **Gradient Descent**: Optimizing neural network parameters
- **Neural Architectures**: Building blocks from scratch (layers, activations, loss functions)
- **Numerical Stability**: Handling edge cases in training
- **Testing**: Gradient checks to verify correctness

## 📦 Installation

```bash
git clone https://github.com/Vishnusai17/DL-Basic.git
cd DL-Basic
pip install -r requirements.txt
```

## 🚀 Quick Start

### Project 1: Neural Network from Scratch

Train a fully connected network on MNIST:

```bash
cd neural_network
python demo_mnist.py
```

**Expected Output**:
- Test Accuracy: >90%
 - Training visualizations saved to `training_history.png` and `mnist_predictions.png`

**Architecture**: `784 → 128 (ReLU) → 10 (Softmax)`

### Running Tests

Verify implementation with gradient checks and sanity tests:

```bash
cd tests
python test_neural_network.py
```

**Tests include**:
- ✅ Numerical gradient checks (Dense, ReLU layers)
- ✅ XOR problem (non-linear separability)
- ✅ Overfitting test (memorization capability)

---

## 📚 Implementations

### 1. Neural Network (`neural_network/`)

**Components**:
- `Dense`: Fully connected layer with Xavier initialization
- `ReLU`: Rectified Linear Unit activation
- `Sigmoid`: Sigmoid activation  
- `Softmax`: Softmax activation for classification
- `CrossEntropyLoss`: Multi-class classification loss
- `MSELoss`: Regression loss

**Math**:

**Forward Pass (Dense layer)**:
```
y = Wx + b
```

**Backward Pass (Chain Rule)**:
```
∂L/∂W = X^T · ∂L/∂y
∂L/∂b = sum(∂L/∂y, axis=0)
∂L/∂x = ∂L/∂y · W^T
```

**Gradient Descent**:
```
W ← W - α · ∂L/∂W
b ← b - α · ∂L/∂b
```

**Usage Example**:
```python
from neural_network.nn_from_scratch import NeuralNetwork, Dense, ReLU, Softmax, CrossEntropyLoss

# Build network
model = NeuralNetwork(
    layers=[
        Dense(784, 128),
        ReLU(),
        Dense(128, 10),
        Softmax()
    ],
    loss_fn=CrossEntropyLoss()
)

# Train
history = model.train(X_train, y_train, epochs=50, learning_rate=0.1, batch_size=128)

# Predict
predictions = model.predict(X_test)
```

---

### 2. CNN (Coming Soon)

Will include:
- Conv2D layer with im2col optimization
- MaxPooling2D
- CIFAR-10 demo

### 3. RNN/LSTM (Coming Soon)

Will include:
- Vanilla RNN cell
- LSTM with forget/input/output gates
- Text generation demo

---

## 🧪 Testing & Verification

### Gradient Checks

All implementations use **numerical gradient verification**:

```python
def numerical_gradient(f, x, eps=1e-5):
    # Central difference: (f(x+ε) - f(x-ε)) / 2ε
    ...
```

**Results**:
```
Dense layer gradients:
  Input gradient diff:  2.54e-11 ✅
  Weight gradient diff: 3.37e-11 ✅

ReLU gradients:
  Gradient diff: 1.50e-11 ✅
```

### XOR Problem

Tests that 2-layer network can solve non-linearly separable XOR:

| Input | Target | Prediction |
|-------|---------|------------|
| [0, 0] | 0 | 0 |
| [0, 1] | 1 | 1 |
| [1, 0] | 1 | 1 |
| [1, 1] | 0 | 0 |

**Accuracy**: 100% ✅

---

## 📖 Mathematical Derivations

### Backpropagation in Dense Layer

**Given**:
- Input: `X` (batch_size, in_features)
- Forward: `Y = XW + b`
- Gradient from next layer: `∂L/∂Y`

**Compute gradients**:

1. **Gradient wrt weights**:
   ```
   ∂L/∂W = X^T · ∂L/∂Y
   ```
   *Derivation*: Using matrix calculus chain rule.

2. **Gradient wrt bias**:
   ```
   ∂L/∂b = Σ(∂L/∂Y, axis=0)
   ```
   *Derivation*: Bias is broadcasted, so sum over batch.

3. **Gradient wrt input** (pass to previous layer):
   ```
   ∂L/∂X = ∂L/∂Y · W^T
   ```

---

## 🎓 Educational Resources

**Recommended Reading**:
1. [CS231n: Convolutional Neural Networks](http://cs231n.github.io/)
2. [Michael Nielsen: Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
3. [Deep Learning Book by Goodfellow et al.](https://www.deeplearningbook.org/)

**Key Concepts Demonstrated**:
- **Computational Graphs**: Forward/backward pass abstraction
- **Automatic Differentiation**: Manual backprop to understand autodiff
- **Numerical Stability**: Clip values, subtract max in softmax
- **Initialization**: Xavier/Glorot for better gradient flow

---

## 🏗️ Project Structure

```
DL-Basic/
├── neural_network/
│   ├── nn_from_scratch.py       # Core implementation
│   ├── demo_mnist.py             # MNIST demo
│   └── __init__.py
├── cnn/                          # (Coming soon)
├── rnn/                          # (Coming soon)
├── tests/
│   └── test_neural_network.py   # Gradient checks & tests
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

This is an educational project. Feel free to fork and extend with:
- Additional optimizers (Adam, RMSProp)
- Batch normalization
- Dropout regularization
- More architectures (GAN, Autoencoder)

---

## 📄 License

MIT License - Educational purposes

---

## ✨ Acknowledgments

Built to demonstrate foundational understanding of deep learning without high-level frameworks. All implementations use **only NumPy** for core training logic.

**Author**: Vishnu Sai Reddy Alla  
**Contact**: avishnusaireddy17@gmail.com  
**Portfolio**: [vishnu-portfolio-ai.vercel.app](https://vishnu-portfolio-ai.vercel.app)
