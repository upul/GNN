import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles


N = 400 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = make_gaussian_quantiles(cov=2.,
                             n_samples=N*D, n_features=2,
                             n_classes=3, random_state=1)


# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))


# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(5000):

  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print("Iteration:  {:d} loss: {:f}".format(i, loss))

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)

  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W

  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {:f}'.format(np.mean(predicted_class == y)))

x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()

x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.c_[x_mesh.ravel(), y_mesh.ravel()]

hidden_layer = np.maximum(0, np.dot(Z, W) + b)
scores = np.dot(hidden_layer, W2) + b2

predicted_class = np.argmax(scores, axis=1)
#print(predicted_class.shape)
predicted_class = predicted_class.reshape(x_mesh.shape)

plt.contourf(x_mesh, y_mesh, predicted_class,cmap=plt.cm.Spectral, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral, alpha=0.9)
#plt.axis('off')
plt.show()
