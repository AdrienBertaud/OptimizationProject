# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import math


COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9']


def get_sharpness_ub(learning_rate):
    return 2/learning_rate


def get_nonuniformity_ub(learning_rate, data_size = 1000, batch_size = 1000):
    '''
    learning_rate: learning rate
    data_size: number of data
    batch_size: batch size
    '''
    return np.sqrt(batch_size*(data_size-1)/(data_size-batch_size+1))/learning_rate


def plot_batch_nonu_sharp(df, learning_rate = 0.5, data_size = 1000, ymax = 15):
    '''
    plot nonuniformity against sharpness for a fixed learning rate but different batch size
    '''

    df_lr = df[df['lr'] == learning_rate]

    plt.vlines(get_sharpness_ub(learning_rate), 0, ymax,
               colors = 'gray', linestyles = '-', linewidth = 0.8)

    plt.text(get_sharpness_ub(learning_rate)-0.3, 1, '2/'+r'$\eta$', c='gray')

    i = 0

    for _, optimizer, batch_size in df_lr[['optimizer','batch size']].drop_duplicates().itertuples():

        data = df[(df['optimizer'] == optimizer) & (df['batch size'] == batch_size)]

        plt.scatter(data[(name + ' train')], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', batch_size='+str(batch_size))

        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1

        if i > 9:
            raise ValueError('color out of range')

    plt.xlabel(name, fontsize = 15)
    plt.ylabel('non-uniformity', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(learning_rate)+0.2)
    plt.legend(loc = 'best')
    plt.savefig('Sharpness vs Nonuniformity for different batch size (lr='+str(learning_rate)+'.pdf')
    plt.title('learning rate = %d'%(learning_rate))
    plt.show()


def plot_lr_nonu_sharp(df, batch_size = 10, data_size = 1000, ymax = 32):
    '''
    plot nonuniformity against sharpness for a fixed batch size but different learning rate
    '''

    if batch_size == 'all':
        return

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():
        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]
        plt.scatter(data[(name + ' train')], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(learning_rate), 0, get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')

    plt.xlabel(name, fontsize = 15)
    plt.ylabel('non-uniformity', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(learning_rate)+1)
    plt.legend(loc = 'best')
    # plt.savefig('Sharpness vs Nonuniformity for different learning rate (bs='+str(batch_size)+'.pdf')
    plt.title('batch size = %d'%(batch_size))
    plt.show()

def plot_batch(df, batch_size = 10, data_size = 1000, ymax = 200, name='sharpness'):
    '''
    plot sharpness for a fixed batch size but different learning rate
    '''

    if batch_size == 'all':
        return

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():

        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]

        plt.scatter(data['lr'], data[(name + ' train')],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(learning_rate), 0, get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('learning rate', fontsize = 15)
    plt.ylabel(name, fontsize = 15)
    plt.ylim(0, ymax)
    max_lr =df_bs['lr'].max()
    plt.xlim(0, get_sharpness_ub(max_lr)+1)
    plt.legend(loc = 'best')
    plt.savefig('Sharpness vs Nonuniformity for different learning rate (bs='+str(batch_size)+'.pdf')
    plt.title('batch size = %d'%(batch_size))
    plt.show()


def plot_lr(df, batch_size = 'all', name='sharpness'):
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
        plt.scatter(values['lr'], values[(name + ' train')], label = optimizer)#, s = values['batch size'])

    lr_max = df_plot['lr'].max()

    ymax = df_plot[(name + ' train')].max().max() * 10
    if math.isnan(ymax) :
        return

    if name == 'sharpness':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_ub(lr) for lr in np.linspace(.0005, lr_max)], 'k--')
    elif name == 'non uniformity' and batch_size != 'all':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_ub(lr, data_size = 1000, batch_size = batch_size) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)
    plt.xlabel('learning rate', fontsize = 12)
    plt.ylabel(name, fontsize = 12)
    plt.savefig('sharpness vs learning rate')
    plt.legend()

    plt.show()



def plot_bs(df, name='sharpness', optim_1= 'gd', optim_2= 'sgd'):
    '''
    plot sharpness against batch size with different learning rate for only sgd and gd data
    '''

    df_sgd = df[(df['optimizer'] == optim_1) | (df['optimizer'] == optim_2)]
    df_plot = pd.DataFrame(columns = ['lr', 'bs', name])
    for lr_bs, value in df_sgd.groupby(['lr', 'batch size']):
        df_plot = df_plot.append({'lr': lr_bs[0], 'bs': int(lr_bs[1]), name: round(value.median()[(name + ' train')],1)}, ignore_index=True)

    df_plot['bs'] = df_plot['bs'].astype(int)
    bs_order = [str(i) for i in sorted(list(set(df_plot['bs'])))]
    df_plot['bs'] = df_plot['bs'].astype(str)

    lrs = sorted(list(set(df_plot['lr'])))

    plt.plot(bs_order, [0 for i in range(len(bs_order))], color = 'white')
    for lr in lrs:
        data = df_plot[df_plot['lr'] == lr]
        plt.plot(data['bs'], data[name], marker='o', label = 'lr=' + str(lr))
    plt.xlabel('batch size', fontsize = 12)
    plt.ylabel(name, fontsize = 12)
    plt.legend()
    fig_name = (optim_2 + ' ' + name + ' vs batch size')
    plt.savefig((fig_name + '.pdf'))
    plt.title(fig_name)
    plt.show()



if __name__ == '__main__':

    file_name = 'results.csv'
    # file_name = 'results_to_test_plot.csv'

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep = ',')

        batch_size_list = [1000, 100, 25, 10, 5, 'all']
        eval_list = ['sharpness','non uniformity']

        df_filtered = df[df['train loss']<=1e-3]
        df_filtered = df_filtered[df_filtered['lr']>3e-4]
        df_filtered = df_filtered[df_filtered['optimizer']!='adam']

        for i in batch_size_list:
            for e in eval_list:
                plot_bs(df_filtered, e)
                plot_bs(df_filtered, e, 'adagrad', 'adagrad')
                plot_lr(df_filtered, i, e)
                plot_batch(df_filtered, batch_size = i, data_size = 1000, ymax = 200, name=e)
                # plot_batch_nonu_sharp(df_filtered, learning_rate = 0.001, data_size = 1000, ymax = 15)
                # plot_lr_nonu_sharp(df_filtered, batch_size = 10, data_size = 1000, ymax = 32)


