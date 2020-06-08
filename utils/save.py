# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import time

def save_results_to_csv(optimizer_name, \
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
                    'duration': round(duration),
                    'train loss': round(train_loss,5),
                    'train accuracy': round(train_accuracy,1),
                    'test loss': round(test_loss,5),
                    'test accuracy': round(test_accuracy,1),
                    'sharpness train': round(sharpness_train),
                    'non uniformity train': round(non_uniformity_train),
                    'sharpness test': round(sharpness_test),
                    'non uniformity test': round(non_uniformity_test)},
                   ignore_index = True)

    df.to_csv(file_name, sep = ',', index = False)


def save_model(model, name):

    directory = 'saved_net/'
    t = time.localtime()
    timestamp = time.strftime('_%b-%d-%Y_%H%M', t)
    file_name = (name + timestamp)
    save_path = (directory + file_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    print("save model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
