# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:43:05 2020

@author: berta
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# COLOR_LIST = ['blue', 'orange', 'green', 'red', 'yellow', 'purple', 'pink']
COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9']

def get_sharpness_ub(lr):
    return 2/lr

def get_nonuniformity_ub(lr, n = 1000, B = 1000):
    '''
    n: number of data
    lr: learning rate
    B: batch size
    '''
    return np.sqrt(B*(n-1)/(n-B+1))/lr

# figure5: sharpness and non-uniformity change w.r.t. batch size

def plot_B(df, lr = 0.5, n = 1000, ymax = 15):
    '''
    plot nonuniformity against sharpness for a fixed learning rate but different batch size
    '''
    df_lr = df[df['lr'] == lr]

#     f, ax = plt.subplots(figsize = (5, 4))
    plt.vlines(get_sharpness_ub(lr), 0, ymax,
               colors = 'gray', linestyles = '-', linewidth = 0.8)
    plt.text(get_sharpness_ub(lr)-0.3, 1, '2/'+r'$\eta$', c='gray')
    i = 0
    for _, optimizer, batch_size in df_lr[['optimizer', 'batch size']].drop_duplicates().itertuples():
        data = df[(df['optimizer'] == optimizer) & (df['batch size'] == batch_size)]
        plt.scatter(data['sharpness train'], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', B='+str(batch_size))
        plt.hlines(get_nonuniformity_ub(lr, n = n, B = batch_size), 0, get_sharpness_ub(lr),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('sharpness', fontsize = 15)
    plt.ylabel('non-uniformity', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(lr)+0.2)
    plt.legend(loc = 'best')
    plt.savefig('Sharpness vs Nonuniformity for different batch size (lr='+str(lr)+'.pdf')
    plt.show()


def plot_lr(df, batch_size = 10, n = 1000, ymax = 32):
    '''
    plot nonuniformity against sharpness for a fixed batch size but different learning rate
    '''
    df_bs = df[df['batch size'] == batch_size]

    i = 0
    for _, optimizer, lr in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():
        data = df[(df['optimizer'] == optimizer) & (df['lr'] == lr)]
        plt.scatter(data['sharpness train'], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(lr))
        plt.hlines(get_nonuniformity_ub(lr, n = n, B = batch_size), 0, get_sharpness_ub(lr),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(lr), 0, get_nonuniformity_ub(lr, n = n, B = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('sharpness', fontsize = 15)
    plt.ylabel('non-uniformity', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(lr)+1)
    plt.legend(loc = 'best')
    plt.savefig('Sharpness vs Nonuniformity for different learning rate (bs='+str(batch_size)+'.pdf')
    plt.show()

def main():

    file_name = 'results.csv'

    if not os.path.exists(file_name):
        return

    df = pd.read_csv(file_name, sep = ',')

    plot_B(df, lr = 0.5, n = 1000, ymax = 15)
    plot_lr(df, batch_size = 10, n = 1000, ymax = 32)

if __name__ == '__main__':
    main()