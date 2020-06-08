# -*- coding: utf-8 -*-
import random as rd
import torch
import numpy as np
import matplotlib.pyplot as plt

def f11(x):
    return 0.1*x**2

def f12(x):
    return 0.1*(x-1)**2

def f1(x):
    return min(f11(x), f12(x))

def f21(x):
    return 10*x**6

def f22(x):
    return 10*(x-1)**2

def f2(x):
    return min(f21(x), f22(x))

def f(x):
    return 0.5*(f1(x)+f2(x))

def h1(x):
    if f11(x) == f12(x):
        raise ValueError('Undifferentialable point of f1(x)')
    else:
        return 0.2

def h2(x):
    if f21(x) < f22(x):
        return 300*x**4
    if f21(x) == f22(x):
        raise ValueError('Undifferentialable point of f2(x)')
    else:
        return 20

def h(x):
    return 0.5*(h1(x)+h2(x))

def condition(x):
    return 2/h(x)
    
def toy_example_gd(f, h, start_value, learning_rate, n_epoch = 100):
    '''
    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.
    start_value : TYPE
        DESCRIPTION.
    learning_rate : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    x = torch.tensor(start_value, requires_grad = True)
    optimizer = torch.optim.SGD([x], lr = learning_rate)

    xlist = []
    ylist = []
    for i in range(n_epoch):
        if i == 0:
            xlist.append(x.item())
            ylist.append(f(x).item())
        optimizer.zero_grad()
        f(x).backward(retain_graph = True)
        optimizer.step()
        xlist.append(x.item())
        ylist.append(f(x).item())

    plot(xlist, ylist, -1, 2, n_epoch, learning_rate, start_value, 'gd')


    condition_0 = condition(.0)
    condition_1 = condition(1.0)
    condition_optimal = condition(xlist[-1])
    print("Condition at x=0: ", condition_0)
    print("Condition at x=1: ", condition_1)
    print("Condition at optimal: ", condition_optimal)


def toy_example_sgd(f, h, start_value, learning_rate, n_epoch = 100):

    x = torch.tensor(start_value, requires_grad = True)
    optimizer = torch.optim.SGD([x], lr = learning_rate)

    xlist = []
    ylist = []
    for i in range(n_epoch):
        if i == 0:
            xlist.append(x.item())
            ylist.append(f(x).item())
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

    plot(xlist, ylist, -1, 2, n_epoch, learning_rate, start_value, 'sgd')



def plot(xlist, ylist, llim, rlim, n_epoch, lr, start_value, optimizer):
    
    # plot f, f1, f2
    x_axis = [i for i in np.linspace(llim, rlim, 1000)]
    f_axis = [f(i) for i in np.linspace(llim, rlim, 1000)]
    f1_axis = [f1(i) for i in np.linspace(llim, rlim, 1000)]
    f2_axis = [f2(i) for i in np.linspace(llim, rlim, 1000)]

    plt.plot(x_axis, f1_axis, label = 'f1(x)', color = 'green', linestyle = '--')
    plt.plot(x_axis, f2_axis, label = 'f2(x)', color = 'blue', linestyle = '--')
    plt.plot(x_axis, f_axis, label = 'f(x)', color = 'black')
    
    # plot trajactory
    for i in range(len(xlist)-1):
        plt.annotate('', xy=(xlist[i+1], ylist[i+1]), xytext=(xlist[i], ylist[i]), arrowprops=dict(arrowstyle="->", edgecolor = 'red'))
        
    title = optimizer + ', lr=' + str(lr) + ', x0=' + str(start_value)
    plt.xlabel('x')
    plt.legend(loc='best')
    plt.ylim(0,1)
    plt.title(title)
    plt.savefig(title+'+TRAJACTORY.pdf')
    plt.show()
    
    # plot changes of loss
    plt.plot(range(n_epoch+1), ylist, color = 'red')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(title+'+LOSS.pdf')
    plt.show()


if __name__ == '__main__':
    start_value = 1-1e-3
    lrs_gd = [0.1, 0.2, 0.3]
    lrs_sgd = [0.05, 0.1, 0.2]
    
    for lr in lrs_gd:
        toy_example_gd(f, h, start_value, lr, 300)
    
    for lr in lrs_sgd:
        toy_example_sgd(f, h, start_value, lr)
        
    
    # toy_example_gd(f, h, start_value=1.2, learning_rate=0.2)
    # toy_example_gd(f, h, start_value=1.3, learning_rate=0.19)

    # toy_example_sgd(f, h, start_value=1.2, learning_rate=0.2)
    # toy_example_sgd(f, h, start_value=1.3, learning_rate=0.19)
