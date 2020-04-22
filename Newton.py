# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import PlotIterations

from importlib import reload
reload(PlotIterations) # Relaod the module, in case it has changed

from PlotIterations import plotIterations

def newton(f, df, x0, N, verbose=True, plot=False, debug=False):

    """
        Nexton method in dim 1.
    """

    if verbose:
        print("*** Newton-Raphson method ***")

    # if plot:
    x = []
    y = []

    for i in range(N):

        derivative = df(x0)

        if derivative == 0:
            print("Error in newton : derivative is equal to 0. Stopping at the actual value.")
            break;

        fx0 = f(x0)

        if plot:
            x.append(x0)
            y.append(fx0)

        x0 = x0 - f(x0)/derivative

        if verbose:
            print("Root = ", x0)

        if debug:
            print("Error = ", f(x0))

    if plot:

        x.append(x0)
        y.append(f(x0))

        if debug:
            print("x : ", x)
            print("y : ", y)

        plotIterations(f, x, y, "Newton_1dim")

    return x0

# def newtonDim2(f, grad, h0, X0, N, verbose=True, plot=False, debug=False):

#     """
#         Nexton method in dim 2.
#     """

#     # Use this alpha for every line search
#     alpha = np.linspace(0.1,1.0,N)

#     # Initialize delta_X, gamma and parts
#     delta_X = np.zeros((2,1))
#     gamma = np.zeros((2,1))
#     part1 = np.zeros((2,2))
#     part2 = np.zeros((2,2))
#     part3 = np.zeros((2,2))
#     part4 = np.zeros((2,2))
#     part5 = np.zeros((2,2))
#     part6 = np.zeros((2,1))
#     part7 = np.zeros((1,1))
#     part8 = np.zeros((2,2))
#     part9 = np.zeros((2,2))

#     # Initialize X
#     X = np.zeros((N+1,2))
#     X[0] = X0

#     # Initialize gradient storage
#     g = np.zeros((N+1,2))
#     g[0] = grad(X[0])

#     # Initialize hessian storage
#     h = np.zeros((N+1,2,2))
#     h[0] = h0

#     for i in range(N):

#         # Compute search direction and magnitude (dx)
#         #  with dx = -alpha * inv(h) * grad
#         delta_X = -np.dot(alpha[i],np.linalg.solve(h[i],g[i]))
#         X[i+1] = X[i] + delta_X

#         # Get gradient update for next step
#         g[i+1] = grad(X[i+1])

#         # Get hessian update for next step
#         gamma = g[i+1]-g[i]
#         part1 = np.outer(gamma,gamma)
#         part2 = np.outer(gamma,delta_X)
#         part3 = np.dot(np.linalg.pinv(part2),part1)
#         part4 = np.outer(delta_X,delta_X)
#         part5 = np.dot(h[i],part4)
#         part6 = np.dot(part5,h[i])
#         part7 = np.dot(delta_X,h[i])
#         part8 = np.dot(part7,delta_X)
#         part9 = np.dot(part6,1/part8)

#         h[i+1] = h[i] + part3 - part9

#     if plot:
#         plt.plot(X[:,0],X[:,1],'r-o')

#         # Save the figure as a PNG
#         plt.savefig('contour.png')

#         plt.show()

#     if verbose:
#         print("Minimum = ", X[N])

#     return X[N]