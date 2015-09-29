import sys
import numpy as np

from layers import SoftmaxLayer
from datareader import load_mnist
from constants import *

x_train, y_train = load_mnist(MNIST_TRAINING_X , MNIST_TRAINING_y)
x_train = x_train.reshape(MNIST_NUM_TRAINING, MNIST_NUM_FEATURES)
y_train = y_train.reshape(MNIST_NUM_TRAINING)

# initialize parameters randomly
W = 0.1 * np.random.randn(MNIST_NUM_FEATURES, MNIST_NUM_OUTPUT)
b = np.zeros((1, MNIST_NUM_OUTPUT))

learning_rate = 0.1 # step size of the gradient descent algorithm
reg_parameter = 0.01  # regularization strength
softmax = SoftmaxLayer(W, b, reg_parameter, MNIST_NUM_OUTPUT)

num_iter = 1000
BATCH_SIZE = 500
for i in range(num_iter):

    idx = np.random.choice(MNIST_NUM_TRAINING, BATCH_SIZE, replace=True)
    x_batch = x_train[idx, :]
    y_batch = y_train[idx]
    output_prob, loss = softmax.forward_pass(x_batch, y_batch)
    if i % 50 == 0:
        print('iteration: {:3d} loss: {:3e}'.format(i, loss))
    gradW, gradB = softmax.backward_pass(output_prob, x_batch, y_batch)
    W = W - learning_rate*gradW
    b = b - learning_rate*gradB
    softmax.update_parameters(W, b)

# prediction
pred_prob = np.dot(x_train, W) + b
predicted_class = np.argmax(pred_prob, axis=1)
print('-----------------------------')
print('training setaccuracy: {:f}'.format(np.mean(predicted_class == y_train)))
print('-----------------------------')
print('\n')

x_test, y_test = load_mnist(MNIST_TESTING_X, MNIST_TESTING_y)
x_test = x_test.reshape(MNIST_NUM_TESTING, MNIST_NUM_FEATURES)
y_test = y_test.reshape(MNIST_NUM_TESTING)

pred_prob = np.dot(x_test, W) + b
predicted_class = np.argmax(pred_prob, axis=1)
print('-----------------------------')
print('training set accuracy: {:f}'.format(np.mean(predicted_class == y_test)))
print('-----------------------------')
