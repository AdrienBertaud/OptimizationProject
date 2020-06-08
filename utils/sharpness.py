# -*- coding: utf-8 -*-
import math
import utils.linalg
from importlib import reload
reload(utils.linalg)
from utils.linalg import eigen_variance, eigen_hessian


def compute_sharpness(net, criterion, optimizer, data_loader):

    n_iters=10
    tol=1e-4

    print('Compute sharpness :')
    sharpness = eigen_hessian(net, criterion, data_loader, \
                      n_iters=n_iters, tol=tol, verbose=True)
    print('Sharpness is %.2e\n'%(sharpness))
    return sharpness

