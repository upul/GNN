def get_grad(x):
    """ This method returns the derivative of f(x)=x^2 function"""
    return 2*x

#initial guess
x = 10
#learning rete
eta = 0.01

num_iterations = 500
for i in range(num_iterations):
    x = x - eta*get_grad(x)
    if i % 50 == 0:
        print('Iteration: {:3d} x: {:.3e} f(x): {:.3e}'.format(i, x, x**2))
print('Iteration: {:3d} x: {:.3e} f(x): {:.3e}'.format(i, x, x**2))
