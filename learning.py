import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore


def sigmoid_function(z):
    print("Calculating Sigmoid....\n")
    return 1 / (1 + np.exp(-z))

def linear_regression(X, weights):
    return np.dot(X, weights)

def SGD(weights_old, learning_rate, y, y_hat, X):
    gradient = np.dot(X.T, (y_hat - y)) / len(y)
    weights_new = weights_old - learning_rate * gradient
    return weights_new

def loss_function(y, y_hat, epsilon):
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)  # To avoid log(0)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def train_model(X, y, epochs, learning_rate, epsilon):
    num_features = X.shape[1]
    weights = np.zeros((num_features, 1))
    loss_values = []

    for _ in range(epochs):
        # Forward propagation
        z = linear_regression(X, weights)
        y_hat = sigmoid_function(z)
        loss = loss_function(y, y_hat, epsilon)
        weights = SGD(weights, learning_rate, y, y_hat, X)
        loss_values.append(loss)

    return loss_values, weights

def preprocess_data(df):
    df = pd.read_csv('SpotifyFeatures.csv')
    df = df[df['genre'].isin(['Pop', 'Classical'])]
    df['label'] = df['genre'].apply(lambda x: 1 if x == 'Pop' else 0)
    
    features = df[['liveness', 'loudness']].values
    labels = df['label'].values

    # Add bias term
    ones = np.ones((features.shape[0], 1))
    features = np.concatenate((features, ones), axis=1)  # Add bias term to features
    return features, labels

def compute_and_display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Classical', 'Pop'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
    return cm

def main():
    # Load and preprocess data
    df = pd.read_csv('SpotifyFeatures.csv')
    features, labels = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

    # Reshape labels
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Train model
    epochs = 1000
    learning_rate = 0.01
    epsilon = 1e-8
    loss_list, weights = train_model(X_train, y_train, epochs, learning_rate, epsilon)

    # Predictions
    train_predictions = sigmoid_function(linear_regression(X_train, weights))
    train_predictions = (train_predictions >= 0.5).astype(int)
    
    test_predictions = sigmoid_function(linear_regression(X_test, weights))
    test_predictions = (test_predictions >= 0.5).astype(int)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Compute and display confusion matrix
    cm = compute_and_display_confusion_matrix(y_test, test_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Plot loss over epochs
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss_list, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot weights over epochs
    plt.subplot(1, 2, 2)
    for i in range(weights.shape[0]):
        plt.plot(weights[i, :], label=f'Weight{i+1}')
    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.title('Weights Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()      

if __name__ == "__main__":
    main()
