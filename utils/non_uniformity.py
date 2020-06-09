# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import math
import utils.linalg
from importlib import reload
reload(utils.linalg)
from utils.linalg import eigen_variance


def eval_non_uniformity(net, criterion, optimizer, data_loader):

    n_iters=20
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


def get_nonuniformity_theorical_limit(learning_rate, data_size = 1000, batch_size = 1000):
    '''
    return theorical limit of non-uniformity depending on given learning rate, data size and batch size

    learning_rate: learning rate
    data_size: number of data
    batch_size: batch size
    '''
    return np.sqrt(batch_size*(data_size-1)/(data_size-batch_size+1))/learning_rate