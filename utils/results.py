# -*- coding: utf-8 -*-

import os
import pandas as pd


DEFAULT_FILE_NAME = 'results.csv'

def save_results_to_csv(optimizer_name, \
                learning_rate, \
                batch_size, \
                num_iter, \
                duration, \
               _loss, \
               _accuracy, \
                test_loss, \
                test_accuracy, \
                sharpness_train, \
                non_uniformity_train, \
                sharpness_test, \
                non_uniformity_test):

    file_name = DEFAULT_FILE_NAME

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
                                   'sharpness',
                                   'non uniformity',
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
                    'sharpness': round(sharpness_train,2),
                    'non uniformity': round(non_uniformity_train),
                    'sharpness test': round(sharpness_test),
                    'non uniformity test': round(non_uniformity_test)},
                   ignore_index = True)

    df.to_csv(file_name, sep = ',', index = False)


def load_results_from_csv(results_file = DEFAULT_FILE_NAME):
    '''
    read stored evaluations and return a data frame
    '''

    if not os.path.exists(results_file):
        print((results_file + ' does not exist, not possible to plot graphs'))
        return pd.DataFrame.empty()

    print(('Loading results from ' + results_file))

    results_data_frame = pd.read_csv(results_file, sep = ',')

    return filter_not_relevant_data(results_data_frame)


def filter_not_relevant_data(results_data_frame):
    '''
    filter the evaluations, so as to use only the relevant ones
    '''
    return results_data_frame[(results_data_frame['train loss'] <= 5e-4) &
                            (results_data_frame['lr'] > 1e-3) &
                            (results_data_frame['lr'] != 0.75) &
                            (results_data_frame['optimizer'] != 'adam')]


