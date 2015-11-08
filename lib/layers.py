import numpy as np


class TanhLayer:
    def __init__(self):
        pass

    def forward_pass(self, x_input, W, b):
        pre_activation = np.dot(x_input, W) + b
        return pre_activation, np.tanh(pre_activation)

    def backward_pass(self, x_input, W, pre_activation, g_output, child_W, reg_parameter):
        grad_h = np.dot(g_output, child_W.T)
        grad_activation = (1 - np.tanh(pre_activation) * np.tanh(pre_activation)) * grad_h
        grad_w = np.dot(x_input.T, grad_activation)
        grad_w += reg_parameter * W
        grad_b = np.sum(grad_activation, axis=0, keepdims=True)
        return grad_w, grad_b

class SoftmaxLayer:
    """
        SoftmaxLayer class represents teh Softmax layer.
        Parameters
        ----------
        W : matrix W represents the input to output connection weight
        b : bias vector
        reg_parameter : regularization parameter of the L2 regularizer
    """

    def __init__(self):
        pass

    def forward_pass(self, x_input, y_input, W, b, reg_parameter, previous_Ws):
        """
        Performs forward pass and returns x_out_prob and total_loss
        """
        # calculates pre-activation using XW + b
        x_hid = np.dot(x_input, W) + b
        # print(np.linalg.norm(x_hid))

        # subtract np.max(x_hid) from each element of the x_hid
        # for numerical stability
        # for details: http://www.iro.umontreal.ca/~bengioy/dlbook/numerical.html
        x_hid = x_hid - np.max(x_hid)
        # calculate output probabilities using Equation 4
        x_out_prob = np.exp(x_hid) / np.sum(np.exp(x_hid), axis=1, keepdims=True)

        # calculate data loss using -log_e(p_k)
        num_examples = x_input.shape[0]
        prob_target = x_out_prob[range(num_examples), y_input]
        data_loss_vector = -np.log(prob_target)
        data_loss = np.sum(data_loss_vector) / num_examples

        reg_loss = reg_parameter * np.sum(W * W) * 0.5 
        for w in previous_Ws:
            reg_loss += reg_parameter * np.sum(w * w) * 0.5
        total_loss = data_loss + reg_loss

        return x_out_prob, total_loss

    def backward_pass(self, x_out_prob, x_input, y_input, W, b, reg_parameter):
        """
        Performs backward pass and calculates error derivatives.
        """
        # calculates error derivatives w.r.t. pre-activation
        num_examples = y_input.shape[0]

        # creating one-hot encoding matrix from y_input
        one_hot = np.zeros((x_input.shape[0], W.shape[1]))
        one_hot[range(num_examples), y_input] = 1

        # Equation: 6
        grad_output = x_out_prob - one_hot
        grad_output = grad_output / num_examples

        # Equation: 8
        grad_w = np.dot(x_input.T, grad_output)

        # adding regularization loss
        grad_w = grad_w + reg_parameter * W

        # Equation : 7
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_w, grad_b, grad_output
