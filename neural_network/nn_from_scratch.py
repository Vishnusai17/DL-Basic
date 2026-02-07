"""
Neural Network from Scratch using NumPy
Educational implementation demonstrating backpropagation and gradient descent
"""

import numpy as np
from typing import Tuple, List, Optional


class Layer:
    """Base class for neural network layers"""
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: compute output given input"""
        raise NotImplementedError
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradient wrt input given gradient wrt output"""
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer: y = Wx + b"""
    
    def __init__(self, in_features: int, out_features: int):
        # Xavier/Glorot initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        
        # Cache for backward pass
        self.X_cache = None
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (batch_size, in_features)
        Returns:
            out: (batch_size, out_features)
        """
        self.X_cache = X
        return X @ self.W + self.b
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Args:
            grad: gradient wrt output (batch_size, out_features)
        Returns:
            grad_X: gradient wrt input (batch_size, in_features)
        """
        # Gradient wrt weights: X^T @ grad
        self.grad_W = self.X_cache.T @ grad
        
        # Gradient wrt bias: sum over batch dimension
        self.grad_b = np.sum(grad, axis=0, keepdims=True)
        
        # Gradient wrt input: grad @ W^T
        grad_X = grad @ self.W.T
        
        return grad_X


class ReLU(Layer):
    """ReLU activation: f(x) = max(0, x)"""
    
    def __init__(self):
        self.X_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X_cache = X
        return np.maximum(0, X)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Gradient: 1 if x > 0, else 0"""
        return grad * (self.X_cache > 0)


class Sigmoid(Layer):
    """Sigmoid activation: f(x) = 1 / (1 + exp(-x))"""
    
    def __init__(self):
        self.out_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out_cache = 1 / (1 + np.exp(-np.clip(X, -500, 500)))  # Clip for numerical stability
        return self.out_cache
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Gradient: sigmoid(x) * (1 - sigmoid(x))"""
        return grad * self.out_cache * (1 - self.out_cache)


class Softmax(Layer):
    """Softmax activation: exp(x_i) / sum(exp(x_j))"""
    
    def __init__(self):
        self.out_cache = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        # Numerical stability: subtract max
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.out_cache = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.out_cache
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Gradient is complex for general case.
        When used with CrossEntropy loss, simplifies to (pred - target).
        """
        return grad  # Simplified, assuming CrossEntropy handles it


class CrossEntropyLoss:
    """Cross-entropy loss for classification"""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            predictions: (batch_size, num_classes) - softmax probabilities
            targets: (batch_size,) - class indices
        Returns:
            loss: scalar
        """
        batch_size = predictions.shape[0]
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        
        # Select the probability of the correct class for each example
        correct_logprobs = -np.log(predictions[np.arange(batch_size), targets])
        
        return np.mean(correct_logprobs)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Gradient of cross-entropy + softmax: pred - one_hot(target)
        
        Args:
            predictions: (batch_size, num_classes)
            targets: (batch_size,)
        Returns:
            grad: (batch_size, num_classes)
        """
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        
        # Convert targets to one-hot
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(batch_size), targets] = 1
        
        # Gradient: (pred - target) / batch_size
        return (predictions - one_hot) / batch_size


class MSELoss:
    """Mean Squared Error loss for regression"""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        batch_size = predictions.shape[0]
        return 2 * (predictions - targets) / batch_size


class NeuralNetwork:
    """Simple feedforward neural network"""
    
    def __init__(self, layers: List[Layer], loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, grad: np.ndarray):
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update_weights(self, learning_rate: float):
        """Gradient descent weight update"""
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.W -= learning_rate * layer.grad_W
                layer.b -= learning_rate * layer.grad_b
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int, learning_rate: float, batch_size: int = 32,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train the network
        
        Returns:
            history: dict with 'train_loss' and optionally 'val_loss' and 'val_acc'
        """
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn.forward(predictions, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                grad = self.loss_fn.backward(predictions, y_batch)
                self.backward(grad)
                
                # Update weights
                self.update_weights(learning_rate)
            
            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.loss_fn.forward(self.forward(X_val), y_val)
                val_acc = np.mean(val_pred == y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
