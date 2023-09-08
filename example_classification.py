from minigradient.lib import Layer
from sklearn.datasets import make_moons, make_blobs, make_classification
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# dataset
X, Y = make_moons(n_samples=100, noise=0.14)
Y = Y*2 - 1

def plot_dataset(X, Y):
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=Y, s=20)
    plt.show()
plot_dataset(X, Y)

X_train, Y_train = np.array(X), np.array(Y).reshape((len(Y), 1))
