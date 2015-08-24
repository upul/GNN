import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

N = 400 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = make_gaussian_quantiles(cov=2.,
                             n_samples=N*D, n_features=2,
                             n_classes=3, random_state=1)
#X = np.zeros((N*K,D)) # data matrix (each row = single example)
#y = np.zeros(N*K, dtype='uint8') # class labels
#for j in range(K):
#    ix = range(N*j,N*(j+1))
#    r = np.linspace(0.0,1,N) # radius
#    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
#    y[ix] = j

# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

#Initilize parameters
W = 00.01*np.random.randn(D, K)
b = np.zeros((1, K))

#hyperparameters
step_size = 1e-1
reg = 1e-3

#gradient descent looop
num_examples = X.shape[0]
for i in range(1000):
    scores = np.dot(X, W) + b

    # to probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis =1, keepdims=True)

    #cross-entropy loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples

    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print("Iteration:  {:d} loss: {:f}".format(i, loss))

    # compute gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    #backpropagate gradients
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis = 0, keepdims =  True)

    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {:f}'.format(np.mean(predicted_class == y)))

x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()

x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.c_[x_mesh.ravel(), y_mesh.ravel()]

scores = np.dot(Z, W) + b
predicted_class = np.argmax(scores, axis=1)

predicted_class = predicted_class.reshape(x_mesh.shape)
predicted_class = predicted_class.reshape(x_mesh.shape)

plt.contourf(x_mesh, y_mesh, predicted_class,cmap=plt.cm.Spectral, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral, alpha=0.9)
#plt.axis('off')
plt.show()
