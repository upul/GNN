import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi
from sklearn.datasets import make_gaussian_quantiles

N = 600
X_features = np.zeros((N,2))
y_label = np.zeros(N) # class labels
def plot_polar(f, start=0, end=2*pi):

    X_features, y_label = make_gaussian_quantiles(cov=2.,
                                 n_samples=1200, n_features=2,
                                 n_classes=3, random_state=1)

    plt.scatter(X_features[:, 0], X_features[:, 1], c = y_label, s=50)
    plt.title("xxx")
    plt.show()

plot_polar(lambda theta: 5 * sin(2 * theta))
