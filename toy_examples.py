# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt

def f11(x):
    return 1/10*x**2

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
    if f11(x) < f12(x):
        return 0.2
    else:
        return 0.2

def h2(x):
    if f21(x) < f22(x):
        return 300*x**4
    else:
        return 20

def h(x):
    return 0.5*(h1(x) + h2(x))


def toy_example_gd(f, h, start_value, learning_rate):

    x = torch.tensor(start_value, requires_grad = True)

    optimizer = torch.optim.SGD([x], lr = learning_rate)

    n_epoch = 40

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
        # print(i, x, f(x))

    # plot
    x_axis = [i for i in np.linspace(-1,2,2000)]
    f_axis = [f(i) for i in np.linspace(-1,2,2000)]
    f1_axis = [f1(i) for i in np.linspace(-1,2,2000)]
    f2_axis = [f2(i) for i in np.linspace(-1,2,2000)]

    plt.plot(x_axis, f1_axis, label = 'f1(x)', color = 'green', linestyle = '--')
    plt.plot(x_axis, f2_axis, label = 'f2(x)', color = 'blue', linestyle = '--')
    plt.plot(x_axis, f_axis, label = 'f(x)', color = 'black')
    for i in range(len(xlist)-1):
        # plt.plot(xlist[i:i+2], ylist[i:i+2], 'r')
        plt.annotate('', xy=(xlist[i+1], ylist[i+1]), xytext=(xlist[i], ylist[i]), arrowprops=dict(arrowstyle="->", edgecolor = 'red'))

    plt.xlabel('x')
    plt.legend(loc='best')
    plt.ylim(0,1)
    plt.title('GD'+', Learning Rate='+str(learning_rate)+', Start at='+str(start_value))
    plt.show()

    hessian1 = h(1.0)
    hessian0 = h(.0)

    hessian = h(xlist[-1])

    print("hessian0 = ", hessian0)
    print("hessian1 = ", hessian1)
    print("hessian = ", hessian)

    condition0 = 2 / hessian0
    condition1 = 2 / hessian1

    print("condition0 = ", condition0)
    print("condition1 = ", condition1)


def toy_example_sgd():

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

if __name__ == '__main__':
    toy_example_gd(f, h, start_value=1.2, learning_rate=0.2)
    toy_example_gd(f, h, start_value=1.3, learning_rate=0.19)

    toy_example_sgd(f, h, start_value=1.2, learning_rate=0.2)
    toy_example_sgd(f, h, start_value=1.3, learning_rate=0.19)
