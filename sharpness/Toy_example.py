# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:52:16 2020

@author: yanis
"""

import torch

def f1(x):
    x1 = x**2
    x2 = 0.1*(x-1)**2
    return min(x1, x2)

def f2(x):
    x1 = x**2
    x2 = 1.9*(x-1)**2
    return min(x1, x2)

def f(x):
    return 0.5*(f1(x) + f2(x))

x = torch.tensor(2., requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

print(x, f(x))

for i in range(10):
    optimizer.zero_grad()
    y = f(x)
    y.backward()
    optimizer.step()
    print(x, y)