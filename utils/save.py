# -*- coding: utf-8 -*-

import os
import pandas as pd


def save_to_csv(optimizer_name, \
                learning_rate, \
                batch_size, \
                num_iter, \
                duration, \
                train_loss, \
                train_accuracy, \
                test_loss, \
                test_accuracy, \
                sharpness_train, \
                non_uniformity_train, \
                sharpness_test, \
                non_uniformity_test):

    file_name = 'results.csv'

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep = ',')
    else:
        df = pd.DataFrame(columns=['optimizer',
                                   'lr',
                                   'batch size',
                                   'num iteration',
                                   'duration',
                                   'train loss',
                                   'train accuracy',
                                   'test loss',
                                   'test accuracy',
                                   'sharpness train',
                                   'non uniformity train',
                                   'sharpness test',
                                   'non uniformity test'])

    df = df.append({'optimizer': optimizer_name,
                    'lr': learning_rate,
                    'batch size': batch_size,
                    'num iteration': num_iter,
                    'duration': duration,
                    'train loss': train_loss,
                    'train accuracy': train_accuracy,
                    'test loss': test_loss,
                    'test accuracy': test_accuracy,
                    'sharpness train': sharpness_train,
                    'non uniformity train': non_uniformity_train,
                    'sharpness test': sharpness_test,
                    'non uniformity test': non_uniformity_test},
                   ignore_index = True)

    df.to_csv(file_name, sep = ',', index = False)