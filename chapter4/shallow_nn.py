import numpy as np

from layers import SoftmaxLayer
from layers import TanhLayer
from datareader import load_mnist

from constants import MNIST_TRAINING_X
from constants import MNIST_TRAINING_y
from constants import MNIST_NUM_TRAINING
from constants import MNIST_NUM_FEATURES
from constants import MNIST_NUM_OUTPUT
from constants import MNIST_TESTING_X
from constants import MNIST_TESTING_y
from constants import MNIST_NUM_TESTING


x_train, y_train = load_mnist(MNIST_TRAINING_X , MNIST_TRAINING_y)
x_train = x_train.reshape(MNIST_NUM_TRAINING, MNIST_NUM_FEATURES)
y_train = y_train.reshape(MNIST_NUM_TRAINING)

# initialize parameters randomly
HIDDEN_LAYER_SIZE = 500
W1 = 0.1 * np.random.randn(MNIST_NUM_FEATURES, HIDDEN_LAYER_SIZE)
b1 = np.zeros((1, HIDDEN_LAYER_SIZE))

W2 = 0.1 * np.random.randn(HIDDEN_LAYER_SIZE, MNIST_NUM_OUTPUT)
b2 = np.zeros((1, MNIST_NUM_OUTPUT))

learning_rate = 0.1  # step size of the gradient descent algorithm
reg_parameter = 0.001  # regularization strength
softmax = SoftmaxLayer()
hidden = TanhLayer()

num_iter = 12000
BATCH_SIZE = 100

for i in range(num_iter):
 
    idx = np.random.choice(MNIST_NUM_TRAINING, BATCH_SIZE, replace=True)
    x_batch = x_train[idx, :]
    y_batch = y_train[idx]
     
    pre_activation, hidden_output = hidden.forward_pass(x_batch, W1, b1)
    hidden_layer_weights = [W1]
    output_prob, loss = softmax.forward_pass(hidden_output, y_batch, W2, b2, reg_parameter, hidden_layer_weights)
     
    g_W2, g_b2, g_output = softmax.backward_pass(output_prob, hidden_output, y_batch, W2, b2, reg_parameter)
    g_W1, g_b1 = hidden.backward_pass(x_batch, W1, pre_activation, g_output, W2, reg_parameter)
     
    W1 = W1 - learning_rate * g_W1
    b1 = b1 - learning_rate * g_b1
 
    W2 = W2 - learning_rate * g_W2
    b2 = b2 - learning_rate * g_b2
     
    if i % 50 == 0:
        print('iteration: {:3d} loss: {:3e}'.format(i, loss))
 
# prediction
hidden = np.tanh(np.dot(x_train, W1) + b1)
pred_prob = np.dot(hidden, W2) + b2
predicted_class = np.argmax(pred_prob, axis=1)
print('-----------------------------')
print('training set accuracy: {:f}'.format(np.mean(predicted_class == y_train)))
print('-----------------------------')
print('\n')
 
x_test, y_test = load_mnist(MNIST_TESTING_X, MNIST_TESTING_y)
x_test = x_test.reshape(MNIST_NUM_TESTING, MNIST_NUM_FEATURES)
y_test = y_test.reshape(MNIST_NUM_TESTING)
 
hidden = np.tanh(np.dot(x_test, W1) + b1)
pred_prob = np.dot(hidden, W2) + b2
predicted_class = np.argmax(pred_prob, axis=1)
print('-----------------------------')
print('training set accuracy: {:f}'.format(np.mean(predicted_class == y_test)))