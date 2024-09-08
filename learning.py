
import pandas as pd # type: ignore
import numpy as np # type: ignore
from math import *

from lib.common import *
from sklearn.model_selection import * # type: ignore
from sklearn.metrics import * # type: ignore
import matplotlib.pyplot as plt

def sigmoid_function(z):
    sigmoid = 1/(1 + np.exp(-z))
    print("Calculating Sigmoid....\n")
    print(f'z {z.shape}: {z}\n')
    print(f'sigmoid output {sigmoid.shape}: {sigmoid}')
    return sigmoid

def linear_regression(x, w):
    """
    Shapes of x and w must be dottable    
    """
    z = x@w.T
    print("Calculating Z....\n")
    print(f'x {x.shape}: {x}\n')
    print(f'w.T {w.T.shape}: {w.T}\n')
    print(f'z output {z.shape}: {z}\n')
    return z

def SGD(w_old, learning_rate, y, y_hat, X):
    w_new = w_old - learning_rate * np.dot(X.T, (y_hat - y))
    print("Calculating SGD....\n")
    print(f'w_old {w_old.shape}: {w_old}\n')
    print(f'learning rate: {learning_rate}\n')
    print(f'X {X.shape}: {X}\n')
    print(f'y_hat {y_hat.shape}: {y_hat}\n')
    print(f'y {y.shape}: {y}\n')
    print(f'y {y.shape}: {y}\n')
    print(f'w_new {w_new.shape}: {w_new}\n')

    return w_new

def loss_function(y, y_hat, epsilon):
    loss = -np.sum((y * np.log2(y_hat+epsilon)) + (1 - y) * np.log2(1-y_hat+epsilon))
    print("Calculating Loss....\n")
    print(f'y {y.shape}: {y}\n')
    print(f'y_hat {y_hat.shape}: {y_hat}\n')
    print(f'epsilon: {epsilon}\n')
    print(f'y {y.shape}: {y}\n')
    print(f'Loss: {loss}')

    return loss

def train_model(x, y, epochs, lr, epsilon):
    w = np.array([[1,1,1]])
    loss_values = []

    for _ in range(epochs):
        # Forward propagate
        z = linear_regression(x, w)
        y_hat = sigmoid_function(z)
        loss = loss_function(y, y_hat, epsilon)
        loss_values.append(loss)
        w = SGD(w, lr, y, y_hat, x)

    
    # accuracy_score(y, y_hat)
    return loss_values, w

if __name__ == "__main__":

    # Constants for data
    SONGS_FILE = 'SpotifyFeatures.csv'
    ALL_SONGS = get_songs(SONGS_FILE)
    FEATURES_USED = ['genre','liveness','loudness']
    POP_SONGS = get_songs_by_genre(ALL_SONGS, 'Pop')
    CLASSICAL_SONGS = get_songs_by_genre(ALL_SONGS, 'Classical')
    EPSILON = 0.000001
    songs = ALL_SONGS
    

    # Constants for training
    epochs = 100
    LR = 0.005

    # Sdd labels to genre feature,
    # NB! This changes ALL_SONGS, POP_SONGS, CLASSICAL_SONGS since songs stores a refrence of ALL_SONGS
    add_label(songs, 'genre')

    # Stripped all other features from dataset and convert to numpy array
    songs_matrix = data_filter(songs, FEATURES_USED)
    songs_matrix = songs_matrix.to_numpy()

    # Add biases to the matrix
    ones = np.ones((songs_matrix.shape[0], 1))
    songs_matrix = np.concatenate((songs_matrix, ones), 1)
    # print(songs_matrix)

    # reshape true y
    y = songs_matrix[:, 0].reshape(-1, 1)

    # remove true values from songs matrix
    X = np.delete(songs_matrix, 0,axis=1)

    # stratify gets more distributed split
    # Shuffles songs_matrix and y correspondingly

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, stratify = y)

    print("Xtrain:", X_train)
    print("Xtest:", X_test)
    print("ytrain:", y_train)
    print("ttest:", y_test)

    loss_list, weights = train_model(X_train, y_train, epochs, LR, EPSILON)
    print("loss_list after:",loss_list)
    print("weights",weights)

    epochs_list = np.linspace(0,epochs, 100)
    # Plot loss_list
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss_list,  label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Plot weights
    plt.subplot(1, 2, 2)
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i], label=f'Weight{i+1}' )
    plt.xlabel('Epochs')
    plt.ylabel('Weights')
    plt.title('Weights Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


