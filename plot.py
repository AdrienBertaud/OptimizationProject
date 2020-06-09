# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math


COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9']

directory_to_save_img = "graph"

def get_sharpness_ub(learning_rate):
    return 2/learning_rate


def get_nonuniformity_ub(learning_rate, data_size = 1000, batch_size = 1000):
    '''
    learning_rate: learning rate
    data_size: number of data
    batch_size: batch size
    '''
    return np.sqrt(batch_size*(data_size-1)/(data_size-batch_size+1))/learning_rate


def plot_batch(df, batch_size = 10, data_size = 1000, ymax = 200, eval_algo='sharpness'):
    '''
    plot sharpness for a fixed batch size but different learning rate
    '''

    if batch_size == 'all':
        return

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():

        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]

        plt.scatter(data['lr'], data[(eval_algo + ' train')],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(learning_rate), 0, get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('learning rate', fontsize = 15)
    plt.ylabel(eval_algo, fontsize = 15)
    plt.ylim(0, ymax)
    max_lr =df_bs['lr'].max()
    plt.xlim(0, get_sharpness_ub(max_lr)+1)
    plt.legend(loc = 'best')
    plt.savefig((directory_to_save_img + '/' 'Sharpness vs Nonuniformity for different learning rate bs '+str(batch_size)+'.png'))
    plt.title('batch size = %d'%(batch_size))
    plt.show()


def plot_vs_learning_rate(df, batch_size = 'all', eval_algo='sharpness'):
    '''
    plot sharpness against learning rate with one or many batch size for different algorithms

    '''
    if batch_size == 'all':
        df_plot = df.groupby(['optimizer'])
        plt.title('batch size = all')
    # elif batch_size not in df['batch size']:
    #     raise ValueError('No results for such a batch_size')
    else:
        df_plot = df[df['batch size'] == batch_size].groupby(['optimizer'])
        plt.title('batch size = %d'%(batch_size))

    for optimizer, values in df_plot:
        plt.scatter(values['lr'], values[(eval_algo + ' train')], label = optimizer)#, s = values['batch size'])

    lr_max = df_plot['lr'].max()

    ymax = df_plot[(eval_algo + ' train')].max().max() * 5
    if math.isnan(ymax) :
        return

    if eval_algo == 'sharpness':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_ub(lr) for lr in np.linspace(.0005, lr_max)], 'k--')
    elif eval_algo == 'non uniformity' and batch_size != 'all':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_ub(lr, data_size = 1000, batch_size = batch_size) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)
    plt.xlabel('learning rate', fontsize = 12)
    plt.ylabel(eval_algo, fontsize = 12)
    plt.savefig((directory_to_save_img + '/' 'sharpness vs learning rate.png'))
    plt.legend()

    plt.show()



def plot_vs_batch_size(df, eval_algo='sharpness', optim_1 = 'adagrad', optim_2 = ''):
    '''
    plot sharpness against batch size with different learning rate for only sgd and gd data
    '''

    if optim_2 == '':
        optim_2 = optim_1

    df_sgd = df[(df['optimizer'] == optim_1) | (df['optimizer'] == optim_2)]
    df_plot = pd.DataFrame(columns = ['lr', 'bs', eval_algo])
    for lr_bs, value in df_sgd.groupby(['lr', 'batch size']):
        df_plot = df_plot.append({'lr': lr_bs[0], 'bs': int(lr_bs[1]), eval_algo: round(value.median()[(eval_algo + ' train')],1)}, ignore_index=True)

    df_plot['bs'] = df_plot['bs'].astype(int)
    bs_order = [str(i) for i in sorted(list(set(df_plot['bs'])))]
    df_plot['bs'] = df_plot['bs'].astype(str)

    lrs = sorted(list(set(df_plot['lr'])))

    plt.plot(bs_order, [0 for i in range(len(bs_order))], color = 'white')
    for lr in lrs:
        data = df_plot[df_plot['lr'] == lr]
        plt.plot(data['bs'], data[eval_algo], marker='o', label = 'lr=' + str(lr))

    plt.xlabel('batch size', fontsize = 12)
    plt.ylabel(eval_algo, fontsize = 12)
    plt.legend()
    fig_name = (optim_2 + ' ' + eval_algo + ' vs batch size')
    plt.savefig((directory_to_save_img + '/' + fig_name + '.png'))
    plt.title(fig_name)
    plt.show()


def main():
    '''
    Plot graphs from results stored.
    '''

    # read stored evaluations
    results_file = 'results.csv'
    if not os.path.exists(results_file):
        print((results_file + ' does not exist, not possible to plot graphs'))
        return
    df = pd.read_csv(results_file, sep = ',')

    # create directory to save images
    if not os.path.exists(directory_to_save_img):
        os.makedirs(directory_to_save_img)

    # define the parameters from which we want to plot graphs
    batch_size_list = [1000, 100, 25, 10, 5, 'all']
    eval_algo_list = ['sharpness','non uniformity']

    # filter the evaluations, so as to use only the relevant ones
    df_filtered = df[(df['train loss'] <= 1e-3) &
                     (df['lr'] > 3e-4) &
                     (df['optimizer'] != 'adam')]

    # call the different plots
    for batch_size in batch_size_list:
        for eval_algo in eval_algo_list:
            plot_vs_batch_size(df_filtered, eval_algo, 'gd', 'sgd')
            plot_vs_batch_size(df_filtered, eval_algo, 'adagrad')
            plot_vs_learning_rate(df_filtered, batch_size, eval_algo)


if __name__ == '__main__':
    main()