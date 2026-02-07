"""
Convolutional Neural Network from Scratch using NumPy
Educational implementation demonstrating convolution, pooling, and backpropagation
"""

import numpy as np
from typing import Tuple
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neural_network.nn_from_scratch import Layer, Dense, ReLU, Softmax, CrossEntropyLoss


def im2col(X: np.ndarray, filter_h: int, filter_w: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform 4D image batch into 2D matrix for efficient convolution
    
    Args:
        X: (batch_size, channels, height, width)
        filter_h: filter height
        filter_w: filter width
        stride: stride
        padding: padding
    
    Returns:
        col: (batch_size * out_h * out_w, channels * filter_h * filter_w)
    """
    batch_size, channels, height, width = X.shape
    
    # Apply padding
    if padding > 0:
        X = np.pad(X, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode='constant')
        height += 2 * padding
        width += 2 * padding
    
    # Output dimensions
    out_h = (height - filter_h) // stride + 1
    out_w = (width - filter_w) // stride + 1
    
    # Create column matrix
    col = np.zeros((batch_size, channels, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = X[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
    return col


def col2im(col: np.ndarray, X_shape: Tuple, filter_h: int, filter_w: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform 2D matrix back to 4D image batch (inverse of im2col)
    
    Args:
        col: (batch_size * out_h * out_w, channels * filter_h * filter_w)
        X_shape: original input shape (batch_size, channels, height, width)
        filter_h: filter height
        filter_w: filter width
        stride: stride
        padding: padding
    
    Returns:
        X: (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = X_shape
    
    # Add padding to height/width
    if padding > 0:
        height += 2 * padding
        width += 2 * padding
    
    out_h = (height - filter_h) // stride + 1
    out_w = (width - filter_w) // stride + 1
    
    col = col.reshape(batch_size, out_h, out_w, channels, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    X = np.zeros((batch_size, channels, height, width))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            X[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    # Remove padding
    if padding > 0:
        return X[:, :, padding:-padding, padding:-padding]
    return X


class Conv2D(Layer):
    """
    2D Convolution layer
    
    Implements convolution using im2col for efficiency
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        """
        Args:
            in_channels: number of input channels
            out_channels: number of output filters
            kernel_size: size of square kernel
            stride: stride
            padding: padding
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights: He initialization for ReLU
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        
        # Cache for backward pass
        self.X_cache = None
        self.col_cache = None
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            X: (batch_size, in_channels, height, width)
        
        Returns:
            out: (batch_size, out_channels, out_height, out_width)
        """
        self.X_cache = X
        batch_size, channels, height, width = X.shape
        
        # Output dimensions
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # im2col transformation
        col = im2col(X, self.kernel_size, self.kernel_size, self.stride, self.padding)
        self.col_cache = col
        
        # Reshape weights for matrix multiplication
        W_col = self.W.reshape(self.out_channels, -1)
        
        # Convolution as matrix multiplication
        out = W_col @ col.T + self.b.reshape(-1, 1)
        
        # Reshape to output shape
        out = out.reshape(self.out_channels, batch_size, out_h, out_w)
        out = out.transpose(1, 0, 2, 3)
        
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass
        
        Args:
            grad: (batch_size, out_channels, out_height, out_width)
        
        Returns:
            grad_X: (batch_size, in_channels, height, width)
        """
        # Reshape gradient
        grad = grad.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)
        
        # Gradient wrt bias
        self.grad_b = np.sum(grad, axis=1)
        
        # Gradient wrt weights
        self.grad_W = (grad @ self.col_cache).reshape(self.W.shape)
        
        # Gradient wrt input
        W_col = self.W.reshape(self.out_channels, -1)
        grad_col = W_col.T @ grad
        grad_col = grad_col.T
        
        grad_X = col2im(grad_col, self.X_cache.shape, self.kernel_size, self.kernel_size, 
                       self.stride, self.padding)
        
        return grad_X


class MaxPool2D(Layer):
    """
    2D Max Pooling layer
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Args:
            pool_size: size of pooling window
            stride: stride (typically equals pool_size for non-overlapping)
        """
        self.pool_size = pool_size
        self.stride = stride
        self.X_cache = None
        self.max_indices = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            X: (batch_size, channels, height, width)
        
        Returns:
            out: (batch_size, channels, out_height, out_width)
        """
        self.X_cache = X
        batch_size, channels, height, width = X.shape
        
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        
        # Reshape for pooling
        X_reshaped = X.reshape(batch_size * channels, 1, height, width)
        col = im2col(X_reshaped, self.pool_size, self.pool_size, self.stride, padding=0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        # Max pooling
        self.max_indices = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        
        out = out.reshape(batch_size, out_h, out_w, channels).transpose(0, 3, 1, 2)
        
        return out
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass
        
        Args:
            grad: (batch_size, channels, out_height, out_width)
        
        Returns:
            grad_X: (batch_size, channels, height, width)
        """
        grad = grad.transpose(0, 2, 3, 1)
        batch_size, channels, height, width = self.X_cache.shape
        
        grad_col = np.zeros((grad.size, self.pool_size * self.pool_size))
        grad_col[np.arange(self.max_indices.size), self.max_indices.flatten()] = grad.flatten()
        
        grad_col = grad_col.reshape(grad.shape[0] * grad.shape[1] * grad.shape[2] * grad.shape[3], -1)
        
        grad_X = col2im(grad_col, (batch_size * channels, 1, height, width),
                       self.pool_size, self.pool_size, self.stride, padding=0)
        
        grad_X = grad_X.reshape(batch_size, channels, height, width)
        
        return grad_X


class Flatten(Layer):
    """Flatten layer to convert 4D feature maps to 2D for fully connected layers"""
    
    def __init__(self):
        self.X_shape = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (batch_size, channels, height, width)
        Returns:
            out: (batch_size, channels * height * width)
        """
        self.X_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, grad: np.ndarray) ->np.ndarray:
        """
        Args:
            grad: (batch_size, flattened_size)
        Returns:
            grad_X: (batch_size, channels, height, width)
        """
        return grad.reshape(self.X_shape)


class SimpleCNN:
    """
    Simple CNN for image classification
    Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> Softmax
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Args:
            input_shape: (channels, height, width)
            num_classes: number of output classes
        """
        channels, height, width = input_shape
        
        # Build architecture
        self.layers = [
            Conv2D(channels, 32, kernel_size=3, padding=1),  # Keep spatial dimensions
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),  # Halve spatial dimensions
            Conv2D(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten()
        ]
        
        # Calculate flattened size
        temp_h = height // 4  # Two pooling layers
        temp_w = width // 4
        flattened_size = 64 * temp_h * temp_w
        
        self.layers.extend([
            Dense(flattened_size, 128),
            ReLU(),
            Dense(128, num_classes),
            Softmax()
        ])
        
        self.loss_fn = CrossEntropyLoss()
    
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
            if isinstance(layer, (Dense, Conv2D)):
                layer.W -= learning_rate * layer.grad_W
                layer.b -= learning_rate * layer.grad_b
    
    def train(self, X_train, y_train, epochs: int, learning_rate: float, 
              batch_size: int = 32, X_val=None, y_val=None) -> dict:
        """Train the CNN"""
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                predictions = self.forward(X_batch)
                loss = self.loss_fn.forward(predictions, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward
                grad = self.loss_fn.backward(predictions, y_batch)
                self.backward(grad)
                
                # Update
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
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)
