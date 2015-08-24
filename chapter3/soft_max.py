import numpy as np
import matplotlib.pyplot as plt
#Train a Linear Classifier

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
print(y.shape)
for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(200):

  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b
  

  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print("Iteration:  {:d} loss: {:f}".format(i, loss))

  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)

  dW += reg*W # regularization gradient

  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db

scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: {:f}'.format(np.mean(predicted_class == y)))

x_min, x_max = -2, 2
y_min, y_max = -2, 2
x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = np.c_[x_mesh.ravel(), y_mesh.ravel()]
scores = np.dot(Z, W) + b
print(scores)
predicted_class = np.argmax(scores, axis=1)
print(predicted_class.shape)
predicted_class = predicted_class.reshape(x_mesh.shape)
print(predicted_class)
plt.contourf(x_mesh, y_mesh, predicted_class,   cmap=plt.cm.Paired, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y, s=60, cmap=plt.cm.Paired, alpha=0.9)
#plt.axis('off')
plt.show()
