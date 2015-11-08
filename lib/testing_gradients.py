import numpy as np
from layers import SoftmaxLayer, TanhLayer
from check_gradient import eval_numerical_gradient
from check_gradient import eval_analytical_gradient
from check_gradient import eval_hidden_analytical_gradient
from check_gradient import eval_hidden_numerical_gradient
from check_gradient import check_gradient

def test_softmax_layer():
    x_train = np.array([[5.1, 3.5, 1.4, 0.2],
                    [4.9, 3.0, 1.4, 0.2],
                    [7.0, 3.2, 4.7, 1.4],
                    [6.4, 3.2, 4.5, 1.5],
                    [6.3, 3.3, 6.0, 2.5],
                    [5.8, 2.7, 5.1, 1.9]])
    y_train = np.array([0, 0, 1, 1, 2, 2])

    softmax = SoftmaxLayer()
    W = 0.001 * np.random.randn(4, 3)
    b = np.zeros((1, 3))
    reg_parameter = 0.001
    
    g_numerical_W = eval_numerical_gradient(softmax.forward_pass, x_train, y_train, W, b, reg_parameter)
    g_analytical_W = eval_analytical_gradient(softmax, x_train, y_train, W, b, reg_parameter)
    assert check_gradient(g_numerical_W, g_analytical_W) <= 1e-7, "Error in calculating gradient of the SoftmaxLayer"
    

def test_tanh_layer():
    x_train = np.array([[5.1, 3.5, 1.4, 0.2],
                    [4.9, 3.0, 1.4, 0.2],
                    [7.0, 3.2, 4.7, 1.4],
                    [6.4, 3.2, 4.5, 1.5],
                    [6.3, 3.3, 6.0, 2.5],
                    [5.8, 2.7, 5.1, 1.9]])
    y_train = np.array([0, 0, 1, 1, 2, 2])
    
    W1 = np.random.randn(4, 10) * 0.001
    b1 = np.zeros((1, 10))
    W2 = np.random.randn(10, 6) * 0.001
    b2 = np.zeros((1, 6))
    
    softmax = SoftmaxLayer()
    tanh = TanhLayer()
    reg_parameter = 0.001 
    g_numerical_W = eval_hidden_numerical_gradient(tanh, softmax, x_train, y_train, W1, b1, W2, b2, reg_parameter)
    g_analytical_W = eval_hidden_analytical_gradient(tanh, softmax, x_train, y_train, W1, b1, W2, b2, reg_parameter)
    assert check_gradient(g_numerical_W, g_analytical_W) <= 1e-7, "Error in calculating gradient of the TanhLayer"    
    

def run_tests():
    test_softmax_layer()
    test_tanh_layer()

if __name__ == '__main__':
    run_tests()
    
