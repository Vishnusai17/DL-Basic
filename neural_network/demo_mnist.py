"""
Demo: Train Neural Network on MNIST Digits Dataset
Achieves >90% accuracy with a simple 2-layer network
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn_from_scratch import Dense, ReLU, Softmax, NeuralNetwork, CrossEntropyLoss


def load_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Convert to numpy arrays and normalize
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def visualize_predictions(X_test, y_test, y_pred, num_samples=10):
    """Visualize some predictions"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image = X_test[idx].reshape(28, 28)
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("Saved predictions to 'mnist_predictions.png'")
    plt.show()


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss')
    if history['val_loss']:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    if history['val_acc']:
        ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Saved training history to 'training_history.png'")
    plt.show()


def main():
    # Load data
    X_train, X_test, y_train, y_test = load_mnist()
    
    # Build network: 784 -> 128 -> 10
    print("\nBuilding neural network: 784 -> 128 -> 10")
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
    print("\nTraining network...")
    history = model.train(
        X_train, y_train,
        epochs=50,
        learning_rate=0.1,
        batch_size=128,
        X_val=X_test,
        y_val=y_test
    )
    
    # Final evaluation
    print("\n" + "="*50)
    y_pred = model.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print("="*50)
    
    # Visualizations
    plot_training_history(history)
    visualize_predictions(X_test, y_test, y_pred)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for digit in range(10):
        mask = y_test == digit
        acc = np.mean(y_pred[mask] == y_test[mask])
        print(f"  Digit {digit}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
