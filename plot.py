# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import utils.results
import utils.sharpness
import utils.non_uniformity
import utils.figure

from importlib import reload
reload(utils.results)
reload(utils.sharpness)
reload(utils.non_uniformity)
reload(utils.figure)

from utils.results import load_results
from utils.sharpness import get_sharpness_theorical_limit
from utils.non_uniformity import get_nonuniformity_theorical_limit
from utils.figure import save_and_show


COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9']


def plot_batch(df, batch_size = 10, data_size = 1000, ymax = 200, eval_algo='sharpness'):
    '''
    plot sharpness for a fixed batch size but different learning rates
    '''

    if batch_size == 'all':
        return

    df_bs = df[df['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():

        data = df[(df['optimizer'] == optimizer) & (df['lr'] == learning_rate)]

        plt.scatter(data['lr'], data[(eval_algo)],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_theorical_limit(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_theorical_limit(learning_rate), 0, get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')
    plt.xlabel('learning rate', fontsize = 15)
    plt.ylabel(eval_algo, fontsize = 15)
    plt.ylim(0, ymax)
    max_lr =df_bs['lr'].max()
    plt.xlim(0, get_sharpness_theorical_limit(max_lr)+1)
    plt.legend(loc = 'best')
    plt.savefig((SAVE_DIRECTORY + '/' 'Sharpness vs Nonuniformity for different learning rate bs '+str(batch_size)+'.png'))
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
        plt.scatter(values['lr'], values[(eval_algo)], label = optimizer)#, s = values['batch size'])

    lr_max = df_plot['lr'].max()

    ymax = df_plot[(eval_algo)].max().max() * 5
    if math.isnan(ymax) :
        return

    if eval_algo == 'sharpness':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_theorical_limit(lr) for lr in np.linspace(.0005, lr_max)], 'k--')
    elif eval_algo == 'non uniformity' and batch_size != 'all':
        plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_theorical_limit(lr, data_size = 1000, batch_size = batch_size) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)
    plt.xlabel('learning rate', fontsize = 12)
    plt.ylabel(eval_algo, fontsize = 12)
    fig_name = 'sharpness vs learning rate.png'
    plt.legend()
    save_and_show(fig_name)



def plot_vs_batch_size(df, eval_algo='sharpness', optim_1 = 'adagrad', optim_2 = ''):
    '''
    plot sharpness against batch size with different learning rate for only sgd and gd data
    '''

    if optim_2 == '':
        optim_2 = optim_1

    df_sgd = df[(df['optimizer'] == optim_1) | (df['optimizer'] == optim_2)]
    df_plot = pd.DataFrame(columns = ['lr', 'bs', eval_algo])
    for lr_bs, value in df_sgd.groupby(['lr', 'batch size']):
        df_plot = df_plot.append({'lr': lr_bs[0], 'bs': int(lr_bs[1]), eval_algo: round(value.median()[(eval_algo)],1)}, ignore_index=True)

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
    save_and_show(fig_name)


def plot_batch(results_data_frame, batch_size = 10, data_size = 1000, ymax = 200, eval_algo='sharpness train'):

    if batch_size == 'all':
        return

    df_bs = results_data_frame[results_data_frame['batch size'] == batch_size]

    i = 0

    for _, optimizer, learning_rate in df_bs[['optimizer', 'lr']].drop_duplicates().itertuples():

        data = results_data_frame[(results_data_frame['optimizer'] == optimizer) & (results_data_frame['lr'] == learning_rate)]

        plt.scatter(data['lr'], data[(eval_algo)],
                    c = COLOR_LIST[i], label = optimizer+', lr='+str(learning_rate))
        plt.hlines(get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size), 0, get_sharpness_theorical_limit(learning_rate),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        plt.vlines(get_sharpness_theorical_limit(learning_rate), 0, get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size),
                   colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            raise ValueError('color out of range')

    plt.xlabel('learning rate', fontsize = 15)
    plt.ylabel(eval_algo, fontsize = 15)
    plt.ylim(0, ymax)
    max_lr =df_bs['lr'].max()
    plt.xlim(0, get_sharpness_theorical_limit(max_lr)+1)
    plt.legend(loc = 'best')
    save_and_show('Sharpness vs Nonuniformity for different lr and batch size '+str(batch_size))


def plot_sharpness_limit(results_data_frame, legend='optimizer', all_values='batch size'):

    df_plot = results_data_frame.groupby([legend])

    lr_max = df_plot['lr'].max()

    ymax = df_plot[('sharpness train')].max().max() * 5
    if math.isnan(ymax) :
        return

    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_theorical_limit(lr) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)


def plot_data_frame(df_plot, abscissa, ordinate, legend):

    for legend, values in df_plot:
        plt.scatter(values[abscissa], values[(ordinate)], label = legend)

    plt.xlabel(abscissa, fontsize = 12)
    plt.ylabel(ordinate, fontsize = 12)
    plt.legend(loc = 'best')


def plot_save_and_show(results_data_frame, title, abscissa='lr', ordinate='sharpness train',  legend='optimizer', all_values='batch size'):

    df_plot = results_data_frame.groupby([legend])

    plot_data_frame(df_plot, abscissa, ordinate, legend)
    save_and_show(title)


def plot_results_with_no_fixed_value(results_data_frame, abscissa='lr', ordinate='sharpness train',  legend='optimizer', all_values='batch size'):

    title = ('all ' + all_values)

    plot_save_and_show(results_data_frame, title, abscissa, ordinate, legend, all_values)


def plot_results_with_fixed_value(results_data_frame, abscissa='lr', ordinate='sharpness train', legend='optimizer', type_of_fixed='batch size', fixed=100):

    title = '{0} {1}'.format(type_of_fixed, fixed)

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == fixed]

    plot_save_and_show(df_plot, title, abscissa, ordinate, legend, type_of_fixed)


def plot_nonuniformity_limit(results_data_frame, abscissa='lr', ordinate='sharpness train', legend='optimizer', type_of_fixed='batch size', fixed=100):

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == fixed].groupby([legend])

    lr_max = df_plot[abscissa].max()

    ymax = df_plot[(ordinate)].max().max() * 5
    if math.isnan(ymax) :
        return

    plt.ylim(0, ymax)
    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_theorical_limit(lr, data_size = 1000, batch_size = fixed) for lr in np.linspace(.0005, lr_max)], 'k--')


def plot_results(results_data_frame, abscissa='batch size', ordinate='sharpness train', legend='lr', type_of_fixed='optimizer', fixed = 'adagrad'):

    df_optimizer = results_data_frame[(results_data_frame[type_of_fixed] == fixed)]

    df_plot = pd.DataFrame(columns = [legend, abscissa, ordinate])

    for versus, value in df_optimizer.groupby([legend, abscissa]):
        df_plot = df_plot.append({legend: versus[0], abscissa: int(versus[1]), ordinate: round(value.mean()[(ordinate)],1)}, ignore_index=True)

    df_plot[abscissa] = df_plot[abscissa].astype(int)
    bs_order = [str(i) for i in sorted(list(set(df_plot[abscissa])))]
    df_plot[abscissa] = df_plot[abscissa].astype(str)

    versus_legends = sorted(list(set(df_plot[legend])))

    plt.plot(bs_order, [0 for i in range(len(bs_order))], color = 'white')

    for versus_legend_iter in versus_legends:
        data = df_plot[df_plot[legend] == versus_legend_iter]
        plt.plot(data[abscissa], data[ordinate], marker='o', label = legend+'=' + str(versus_legend_iter))

    plt.xlabel(abscissa, fontsize = 12)
    plt.ylabel(ordinate, fontsize = 12)
    plt.legend()
    save_and_show(fixed + ' ' + ordinate + ' vs ' + abscissa)

def main():
    '''
    Plot graphs from results stored.
    '''

    results_data_frame = load_results()

    # define the parameters from which we want to plot graphs
    eval_algo_list = ['sharpness train','non uniformity train']

    # call the different plots
    for eval_algo in eval_algo_list:
        plot_vs_batch_size(results_data_frame, eval_algo, 'gd', 'sgd')
        plot_vs_batch_size(results_data_frame, eval_algo, 'adagrad')
        plot_vs_learning_rate(results_data_frame, 100, eval_algo)

    plot_results(results_data_frame,
                abscissa='batch size',
                ordinate='sharpness train',
                legend='lr',
                type_of_fixed='optimizer',
                fixed = 'sgd')

    plot_sharpness_limit(results_data_frame,
                  legend='optimizer',
                  all_values='batch size')

    plot_results_with_fixed_value(results_data_frame,
                abscissa='lr',
                ordinate='sharpness train',
                legend='optimizer',
                type_of_fixed='batch size',
                fixed=1000)

    plot_nonuniformity_limit(results_data_frame,
                abscissa='lr',
                ordinate='sharpness train',
                legend='optimizer',
                type_of_fixed='batch size',
                fixed=100)

    plot_results_with_no_fixed_value(results_data_frame,
                  abscissa='lr',
                  ordinate='non uniformity train',
                  legend='optimizer',
                  all_values='batch size')


if __name__ == '__main__':
    main()