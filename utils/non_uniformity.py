# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import math
import utils.linalg
from importlib import reload
reload(utils.linalg)
from utils.linalg import eigen_variance


def eval_non_uniformity(net, criterion, optimizer, data_loader):

    n_iters=10
    tol=1e-4

    print('Compute non_uniformity :')
    v = eigen_variance(net, criterion, data_loader, \
                      n_iters=n_iters, tol=tol, verbose=True)

    print('non_uniformity is %.2e\n'%(v))

    if v<0:
        print("ERROR: eigen variance is negative : ", v)
        return 0
    else:
        return math.sqrt(v)