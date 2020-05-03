# -*- coding: utf-8 -*-
import sys
import numpy as np
from importlib import reload

sys.path.append("./Methods")
import Methods.Newton
import Methods.Secant
reload(Methods.Newton)
reload(Methods.Secant)
from Methods.Newton import newton
from Methods.Secant import secant

# Function to find its root.
def f(x):
    return np.tanh(x)+0.1

# Derivative.
def df(x):
    return 1 - np.tanh(x)**2

# Initializing the values.
nbIter = 10
x0 = 1
x1 = -x0

# Running methods.
newton(f, df, x0, nbIter, plot=True)
secant(f, x0, -x0, nbIter, plot=True);
