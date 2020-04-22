# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def plotIterations(f, x, y, name="", debug = False):

    plt.figure()
    plt.plot(x, y, 'kx')

    minX = min(x)
    maxX = max(x)
    stepX = (maxX-minX)/100

    if debug:
        print("minX = ", minX)
        print("maxX = ", maxX)

    t = np.arange(minX, maxX, stepX)
    plt.plot(t, f(t), 'grey')

    for i in range(len(x)):
        plt.text(x[i], y[i], str(i), {'color': 'red', 'fontsize': 14})

    # Save the figure as a PNG
    plt.savefig(name + '.png')

    plt.show()