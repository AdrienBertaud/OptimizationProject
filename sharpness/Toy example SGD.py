# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:44:20 2020

@author: yanis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random as rd

def f1(x):
    return min(x**2, 0.1*(x-1)**2)

def f2(x):
    return min(x**2, 1.9*(x-1)**2)

def f(x):
    return 0.5*(f1(x)+f2(x))

STARTING = 2.0
LEARNINGRATE = 0.1

x = torch.tensor(STARTING, requires_grad = True)

optimizer = torch.optim.SGD([x], lr = LEARNINGRATE)

xlist = []
ylist = []
for i in range(100):
    if i == 0:
        xlist.append(x.item())
        ylist.append(f(x).item())
        # print(i, x, f(x))
    rand = rd.randint(1, 2)
    if rand == 1:
        optimizer.zero_grad()
        f1(x).backward(retain_graph = True)
        optimizer.step()
        xlist.append(x.item())
        ylist.append(f(x).item())
    else:
        optimizer.zero_grad()
        f2(x).backward(retain_graph = True)
        optimizer.step()
        xlist.append(x.item())
        ylist.append(f(x).item())
    # print(i, x, f(x))

# plot
x_axis = [i for i in np.linspace(-1,2)]
f_axis = [f(i) for i in np.linspace(-1,2)]
f1_axis = [f1(i) for i in np.linspace(-1,2)]
f2_axis = [f2(i) for i in np.linspace(-1,2)]

plt.plot(x_axis, f1_axis, label = 'f1(x)', color = 'green', linestyle = '--')
plt.plot(x_axis, f2_axis, label = 'f2(x)', color = 'blue', linestyle = '--')
plt.plot(x_axis, f_axis, label = 'f(x)', color = 'black')
for i in range(len(xlist)-1):
    # plt.plot(xlist[i:i+2], ylist[i:i+2], 'r')
    plt.annotate('', xy=(xlist[i+1], ylist[i+1]), xytext=(xlist[i], ylist[i]), arrowprops=dict(arrowstyle="->", edgecolor = 'red'))
    
plt.xlabel('x')
plt.legend(loc='best')
plt.ylim(0,1)
plt.title('SGD'+', Learning Rate='+str(LEARNINGRATE)+', Start at='+str(STARTING))
plt.show()