# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:26:52 2020

@author: berta
"""

import os
import src.trainer
import src.utils
import pandas as pd

from importlib import reload
reload(src.utils)


def save(optimizer_name, learning_rate, batch_size, num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness, non_uniformity):

    if os.path.exists('results2.csv'):
        df = pd.read_csv('results2.csv', sep = ',')
    else:
        df = pd.DataFrame(columns=['optimizer', 'lr', 'batch size', 'num iteration',
                                   'train loss', 'train accuracy', 'test loss', 'test accuracy',
                                   'sharpness', 'non uniformity'])

    df = df.append({'optimizer': optimizer_name, 'lr': learning_rate, 'batch size': batch_size, 'num iteration': num_iter,
                                'train loss': train_loss, 'train accuracy': train_accuracy, 'test loss': test_loss, 'test accuracy': test_accuracy,
                                'sharpness': sharpness, 'non uniformity': non_uniformity}, ignore_index = True)

    df.to_csv('results2.csv', sep = ',', index = False)