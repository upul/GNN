import numpy as np

class SoftmaxLayer:
    """ doc """
    def __init__(self, W, b, reg_parameter):
        self.W = W
        self.b = b
        self.reg_parameter = reg_parameter

    def forward_pass(self, x_input, y_input):
        """ doc """
        x_hid = np.dot(x_input, self.W) + self.b
        x_hid = x_hid - np.max(x_hid)
        x_output_prob = np.exp(x_hid) / np.sum(np.exp(x_hid), axis=1, keepdims=True)

        num_examples = x_input.shape[0]
        prob_target =  x_output_prob[range(num_examples), y_input]
        data_loss_vector = -np.log(prob_target)
        data_loss = np.sum(data_loss_vector) / num_examples
        reg_loss = self.reg_parameter * np.sum(self.W * self.W) * 0.5
        total_loss = data_loss + reg_loss

        return x_output_prob, total_loss

    def backward_pass(self, x_output_prob, x_input, y_input):
        """ doc """
        num_examples = y_input.shape[0]
        grad_output = x_output_prob
        grad_output[range(num_examples), y_input] = grad_output[range(num_examples), y_input] - 1
        grad_output = grad_output / num_examples

        # grad w.r.t. weights
        grad_w = np.dot(x_input.T, grad_output)
        grad_w = grad_w + self.reg_parameter*self.W
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_w, grad_b
