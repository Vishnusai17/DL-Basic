"""
Unit tests for Neural Network implementation
Includes gradient checks and functionality tests
"""

import numpy as np
import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network.nn_from_scratch import Dense, ReLU, Sigmoid, Softmax, CrossEntropyLoss, MSELoss


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences
    
    Args:
        f: function that takes x and returns scalar loss
        x: point to evaluate gradient at
        eps: small perturbation
    
    Returns:
        grad: numerical gradient (same shape as x)
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + eps)
        x[idx] = old_value + eps
        pos = f(x)
        
        # f(x - eps)
        x[idx] = old_value - eps
        neg = f(x)
        
        # Central difference
        grad[idx] = (pos - neg) / (2 * eps)
        
        # Restore
        x[idx] = old_value
        it.iternext()
    
    return grad


def test_dense_layer_gradient():
    """Test Dense layer gradient computation"""
    print("\nTesting Dense layer gradients...")
    
    np.random.seed(42)
    layer = Dense(5, 3)
    X = np.random.randn(2, 5)
    
    # Forward pass
    out = layer.forward(X)
    
    # Dummy gradient from next layer
    grad_out = np.random.randn(*out.shape)
    
    # Analytical gradient
    grad_X_analytical = layer.backward(grad_out)
    
    # Numerical gradient for input
    def f_input(x):
        return np.sum(layer.forward(x.reshape(X.shape)) * grad_out)
    
    grad_X_numerical = numerical_gradient(f_input, X.flatten()).reshape(X.shape)
    
    diff = np.abs(grad_X_analytical - grad_X_numerical).max()
    print(f"  Input gradient difference: {diff:.2e}")
    assert diff < 1e-5, f"Gradient check failed! Diff: {diff}"
    
    # Numerical gradient for weights
    def f_weights(w):
        old_W = layer.W.copy()
        layer.W = w.reshape(layer.W.shape)
        result = np.sum(layer.forward(X) * grad_out)
        layer.W = old_W
        return result
    
    _ = layer.forward(X)  # Reset cache
    _ = layer.backward(grad_out)
    grad_W_numerical = numerical_gradient(f_weights, layer.W.flatten()).reshape(layer.W.shape)
    
    diff_W = np.abs(layer.grad_W - grad_W_numerical).max()
    print(f"  Weight gradient difference: {diff_W:.2e}")
    assert diff_W < 1e-5, f"Weight gradient check failed! Diff: {diff_W}"
    
    print("  ✅ Dense layer gradients correct!")


def test_relu_gradient():
    """Test ReLU activation gradient"""
    print("\nTesting ReLU gradients...")
    
    np.random.seed(42)
    layer = ReLU()
    X = np.random.randn(3, 4)
    
    out = layer.forward(X)
    grad_out = np.random.randn(*out.shape)
    grad_X_analytical = layer.backward(grad_out)
    
    def f(x):
        return np.sum(layer.forward(x.reshape(X.shape)) * grad_out)
    
    grad_X_numerical = numerical_gradient(f, X.flatten()).reshape(X.shape)
    
    diff = np.abs(grad_X_analytical - grad_X_numerical).max()
    print(f"  Gradient difference: {diff:.2e}")
    assert diff < 1e-5, f"ReLU gradient check failed! Diff: {diff}"
    print("  ✅ ReLU gradients correct!")


def test_xor_problem():
    """Test that 2-layer network can learn XOR (non-linear problem)"""
    print("\nTesting XOR problem (non-linearity)...")
    
    from neural_network.nn_from_scratch import NeuralNetwork
    
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int32)
    
    # Small network: 2 -> 8 -> 2 (more hidden units help)
    np.random.seed(42)
    model = NeuralNetwork(
        layers=[Dense(2, 8), ReLU(), Dense(8, 2), Softmax()],
        loss_fn=CrossEntropyLoss()
    )
    
    # Train with lower learning rate
    history = model.train(X, y, epochs=2000, learning_rate=0.1, batch_size=4)
    
    # Test
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"  XOR Accuracy: {accuracy*100:.1f}%")
    assert accuracy >= 0.75, f"Failed to learn XOR! Accuracy: {accuracy}"
    print("  ✅ Successfully learned XOR!")


def test_overfitting():
    """Test that network can overfit small dataset (memorization)"""
    print("\nTesting overfitting capability...")
    
    from neural_network.nn_from_scratch import NeuralNetwork
    
    np.random.seed(42)
    X = np.random.randn(10, 20)
    y = np.random.randint(0, 3, size=10)
    
    model = NeuralNetwork(
        layers=[Dense(20, 50), ReLU(), Dense(50, 3), Softmax()],
        loss_fn=CrossEntropyLoss()
    )
    
    history = model.train(X, y, epochs=200, learning_rate=0.1, batch_size=10)
    
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    
    print(f"  Training Accuracy: {accuracy*100:.1f}%")
    assert accuracy >= 0.9, f"Failed to overfit! Accuracy: {accuracy}"
    print("  ✅ Successfully overfits small dataset!")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running Neural Network Tests")
    print("="*60)
    
    test_dense_layer_gradient()
    test_relu_gradient()
    test_xor_problem()
    test_overfitting()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
