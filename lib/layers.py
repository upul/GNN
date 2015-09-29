import numpy as np

class SoftmaxLayer:
    """
        SoftmaxLayer class represents teh Softmax layer.
        Parameters
        ----------
        W : matrix W represents the input to output connection weight
        b : bias vector
        reg_parameter : regularization parameter of the L2 regularizer
    """
    def __init__(self, W, b, reg_parameter, num_unique_categories):
        self.W = W
        self.b = b
        self.reg_parameter = reg_parameter
        self.num_unique_categories = num_unique_categories

    def forward_pass(self, x_input, y_input):
        """
        Performs forward pass and returns x_out_prob and total_loss
        """
        # calculates pre-activation using XW + b
        x_hid = np.dot(x_input, self.W) + self.b

        # subtract np.max(x_hid) from each element of the x_hid
        # for numerical stability
        # detials: http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        x_hid = x_hid - np.max(x_hid)
        # calculate output probabilities using Equation 4
        x_out_prob = np.exp(x_hid) / np.sum(np.exp(x_hid), axis=1, keepdims=True)

        # calculate data loss using -log_e(p_k)
        num_examples = x_input.shape[0]
        prob_target =  x_out_prob[range(num_examples), y_input]
        data_loss_vector = -np.log(prob_target)
        data_loss = np.sum(data_loss_vector) / num_examples

        reg_loss = self.reg_parameter * np.sum(self.W * self.W) * 0.5
        total_loss = data_loss + reg_loss

        return x_out_prob, total_loss

    def backward_pass(self, x_out_prob, x_input, y_input):
        """
        Performs backward pass and calculates error derivatives.
        """
        # calculates error derivaties w.r.t. pre-activation
        num_examples = y_input.shape[0]

        # creating one-hot encoding matrix from y_input
        one_hot = np.zeros((x_input.shape[0], self.num_unique_categories))
        one_hot[range(num_examples), y_input] = 1

        # Equation: 6
        grad_output = x_out_prob - one_hot
        grad_output = grad_output / num_examples

        # Equation: 8
        grad_w = np.dot(x_input.T, grad_output)

        # adding regularization loss
        grad_w = grad_w + self.reg_parameter*self.W

        # Equation : 7
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_w, grad_b

    def update_parameters(self, W, b):
        """
        Updating parameters of the model
        """
        self.W = W
        self.b = b
