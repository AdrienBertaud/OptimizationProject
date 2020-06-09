# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


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

        plt.scatter(data['sharpness train'], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', batch_size='+str(batch_size))

        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1

        if i > 9:
            raise ValueError('color out of range')

    plt.xlabel('sharpness', fontsize = 15)
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

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():
        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]
        plt.scatter(data['sharpness train'], data['non uniformity train'],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(learning_rate), 0, get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    
    plt.xlabel('sharpness', fontsize = 15)
    plt.ylabel('non-uniformity', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(learning_rate)+1)
    plt.legend(loc = 'best')
    # plt.savefig('Sharpness vs Nonuniformity for different learning rate (bs='+str(batch_size)+'.pdf')
    plt.title('batch size = %d'%(batch_size))
    plt.show()

def plot_batch_sharp(df, batch_size = 10, data_size = 1000, ymax = 200):
    '''
    plot sharpness for a fixed batch size but different learning rate
    '''

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():

        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]

        plt.scatter(data['lr'], data['sharpness train'],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_ub(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_ub(learning_rate), 0, get_nonuniformity_ub(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('learning rate', fontsize = 15)
    plt.ylabel('sharpness', fontsize = 15)
    plt.ylim(0, ymax)
    plt.xlim(0, get_sharpness_ub(learning_rate)+1)
    plt.legend(loc = 'best')
    plt.savefig('Sharpness vs Nonuniformity for different learning rate (bs='+str(batch_size)+'.pdf')
    plt.title('batch size = %d'%(batch_size))
    plt.show()

def plot_s_lr(df, batch_size = None):
    '''
    plot sharpness against learning rate with one or many batch size for different algorithms

    '''
    if batch_size == None:
        df_plot = df.groupby(['optimizer'])
    if batch_size not in df['batch size']:
        raise ValueError('No results for such a batch_size')
    else:
        df_plot = df[df['batch size'] == batch_size].groupby(['optimizer'])
        
    for optimizer, values in df_plot:
        plt.scatter(values['lr'], values['sharpness'], label = optimizer)#, s = values['batch size'])
    
    lr_max = df_plot['lr'].max()
    ymax = df_plot['sharpness'].max().max() + 5
    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_ub(lr) for lr in np.linspace(.0005, lr_max)], 'k--', label = 'condition')
    plt.ylim(0, ymax)
    plt.xlabel('learning rate', fontsize = 12)
    plt.ylabel('sharpness', fontsize = 12)
    plt.savefig('sharpness against learning rate')
    plt.legend()
    
def plot_s_bs(df):
    '''
    plot sharpness against batch size with different learning rate for only sgd and gd data
    '''
    
    df_sgd = df[(df['optimizer'] == 'sgd') | (df['optimizer'] == 'gd')]
    df_plot = pd.DataFrame(columns = ['lr', 'bs', 'sharpness'])
    for lr_bs, value in df_sgd.groupby(['lr', 'batch size']):
        df_plot = df_plot.append({'lr': lr_bs[0], 'bs': int(lr_bs[1]), 'sharpness': round(value.mean()['sharpness train'],1)}, ignore_index=True)
        
    df_plot['bs'] = df_plot['bs'].astype(int)
    bs_order = [str(i) for i in sorted(list(set(df_plot['bs'])))]
    df_plot['bs'] = df_plot['bs'].astype(str)
    
    lrs = sorted(list(set(df_plot['lr'])))

    plt.plot(bs_order, [0 for i in range(len(bs_order))], color = 'white')
    for lr in lrs:
        data = df_plot[df_plot['lr'] == lr]
        plt.plot(data['bs'], data['sharpness'], marker='o', label = 'lr=' + str(lr))
    plt.xlabel('batch size', fontsize = 12)
    plt.ylabel('sharpness', fontsize = 12)
    plt.legend()
    plt.savefig('sharpness vs batch size.pdf')
    plt.show()
        
    

if __name__ == '__main__':

    file_name = 'results.csv'
    # file_name = 'results_to_test_plot.csv'

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep = ',')

        # plot_batch_nonu_sharp(df, learning_rate = 0.001, data_size = 1000, ymax = 15)

        # plot_lr_nonu_sharp(df, batch_size = 10, data_size = 1000, ymax = 32)

        plot_batch_sharp(df, batch_size = 10, data_size = 1000, ymax = 200)
        
        plot_s_bs(df)
        plot_s_lr(df)
        plot_s_lr(df, 1000)