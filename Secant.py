# -*- coding: utf-8 -*-

import sys
# Adds directories to python modules path.
sys.path.append("./Utils")

import Utils.PlotIterations

from importlib import reload
reload(Utils.PlotIterations) # Relaod the module, in case it has changed

from Utils.PlotIterations import plotIterations

# source: https://www.codewithc.com/secant-method-matlab-program/

def secant(f, x0, x1, N, verbose=True, plot=False, debug=False):

    """
        Secant method is a numerical method to find the approximate root
        of polynomial equations. During the course of iteration,
        this method assumes the function to be approximately linear
        in the region of interest

        Advantages of Secant Method over other Root Finding Methods:
        - Its rate of convergence is more rapid than that of bisection method.
          So, secant method is considered to be a much faster root finding method.
        - In this method, there is no need to find the derivative of the function
          as in Newton-Raphson method.

        Limitations of Secant Method:
        The method fails to converge when f(xn) = f(xn-1)
        If X-axis is tangential to the curve, it may not converge to the solution.
    """
    if verbose:
        print("*** Secant method ***")

    x = []
    y = []

    if plot:
        x.append(x0)
        y.append(f(x0))
        x.append(x1)
        y.append(f(x1))

    # Initialization of the n + 2 value.
    x2 = x1

    for n in range(N):

        if debug:
            print("x0 = ", x0)
            print("x1 = ", x1)

        fx0 = f(x0)
        fx1 = f(x1)

        deltaF = fx1 - fx0

        if debug:
            print("deltaF = ", deltaF)

        if(deltaF == 0):
            print("Error : f(x1) = f(x0), stopping at the actual value.")
            break;

        x2 = x1 - f(x1) * (x1 - x0 ) / deltaF

        if verbose:
            print("x2 = ", x2)

        error = f(x2)

        if plot:
            x.append(x2)
            y.append(error)

        if debug:
            print("f(x2)  = ", error)

        x0 = x1
        x1 = x2

    if plot:

        if debug:
            print("x : ", x)
            print("y : ", y)

        plotIterations(f, x, y, "secant")

    return x2




