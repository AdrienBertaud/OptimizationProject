# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:29:55 2020

@author: berta
"""

import torch
import utils.trainer
import utils.utils

from importlib import reload
reload(utils.utils)

def get_optimizer(net, optimizer, learning_rate):

    print("optimizer = ", optimizer_name)

    if optimizer == 'gd':
        return torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer == 'adagrad':
         return torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    elif optimizer == 'lbfgs':
         return torch.optim.LBFGS(net.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
         return torch.optim.AdamW(net.parameters(), lr=learning_rate)
    else:
        raise ValueError('optimizer %s is not supported'%(optimizer))