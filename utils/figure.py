# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import utils.results
import utils.sharpness
import utils.non_uniformity

from importlib import reload
reload(utils.results)
reload(utils.sharpness)
reload(utils.non_uniformity)

from utils.sharpness import get_sharpness_theorical_limit
from utils.non_uniformity import get_nonuniformity_theorical_limit


COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9', 'C9']


def save_fig(fig_name, save_directory = "figures", extension='.png'):
    '''
    save figure with given name
    '''

    path = (save_directory + '/' + fig_name + extension)

    print('saving ', path)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(path)


def save_and_show(title):
    plt.title(title)
    save_fig(title)
    plt.show()


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


def plot_sharpness_vs_learning_rate(df, batch_size = 'all'):

    eval_algo='sharpness train'

    if batch_size == 'all':
        df_plot = df.groupby(['optimizer'])
        title = ('sharpness vs learning rate with all batch sizes')
    # elif batch_size not in df['batch size']:
    #     raise ValueError('No results for such a batch_size')
    else:
        df_plot = df[df['batch size'] == batch_size].groupby(['optimizer'])
        title = ('sharpness vs learning rate with batch size = %d'%(batch_size))

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
    plt.ylabel(eval_algo, fontsize = 14)
    plt.legend()
    save_and_show(title)


def plot_sharpness_vs_batch_size(df, optim_1 = 'adagrad'):

    plot_results(results_data_frame = df,
            abscissa='batch size',
            ordinate='sharpness train',
            legend='lr',
            type_of_fixed='optimizer',
            fixed = optim_1)


def plot_sharpness_nonuniformity(results_data_frame, batch_size = 10, data_size = 1000, optimizer_name='sgd'):

    ordinate='sharpness train'
    abscissa='non uniformity train'

    if batch_size == 'all':
        return

    df_filter = results_data_frame[(results_data_frame['batch size'] == batch_size) &
                               (results_data_frame['optimizer'] == optimizer_name)]


    df_filter.sort_values(by = 'lr', inplace = True)

    i = 0

    previous_lr = df_filter['lr'].min()
    label = 'lr='+str(previous_lr)
    first_time = True

    min_nu_lim = 1000
    min_sharpness_lim = 1000

    for _, learning_rate, nu, sharpness in df_filter[['lr', abscissa, ordinate]].itertuples():

        print('learning_rate, nu, sharpness :')
        print(learning_rate, nu, sharpness)

        if first_time == False:
            if previous_lr != learning_rate:
                label = 'lr='+str(learning_rate)
                previous_lr = learning_rate
                i += 1
                if i > len(COLOR_LIST):
                    raise ValueError('color out of range')
            else:
                label = ''

        plt.scatter(nu, sharpness,
                    c = COLOR_LIST[i], label =label)

        first_time = False

        nu_lim = round(get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size))

        sharpness_lim = round(get_sharpness_theorical_limit(learning_rate))

        print('learning_rate, nu_lim, sharpness_lim :')
        print(learning_rate, nu_lim, sharpness_lim)

        min_nu_lim = min(min_nu_lim, nu_lim)
        min_sharpness_lim = min(sharpness_lim, min_sharpness_lim)

        plt.hlines(sharpness_lim, 0, nu_lim, color = COLOR_LIST[i], linestyles = '--', linewidth = 3)

        plt.vlines(nu_lim, 0, sharpness_lim, colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)

    plt.xlabel(abscissa, fontsize = 14)
    plt.ylabel(ordinate, fontsize = 14)

    plt.ylim(0, min_nu_lim*2)
    plt.xlim(0, min_sharpness_lim*2)

    plt.legend(loc = 'best')
    save_and_show('Sharpness vs Nonuniformity for different lr and batch size '+str(batch_size))


def plot_sharpness_limit(results_data_frame, legend='optimizer'):

    abscissa = 'lr'
    ordinate = 'sharpness train'

    df_plot = results_data_frame.groupby([legend])

    lr_max = df_plot[abscissa].max()

    ymax = df_plot[(ordinate)].max().max() * 5

    if math.isnan(ymax) :
        return

    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_theorical_limit(lr) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)

    title = 'sharpness vs lr with all batch sizes'

    plot_save_and_show(results_data_frame, title, abscissa, ordinate, legend)


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


def plot_nonuniformity_limit(results_data_frame, legend='optimizer', batch_size=100):

    abscissa='lr'
    ordinate='non uniformity train'
    type_of_fixed='batch size'

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == batch_size].groupby([legend])

    lr_max = df_plot[abscissa].max()

    ymax = df_plot[(ordinate)].max().max()
    if math.isnan(ymax) :
        return

    plt.ylim(0, ymax * 3)
    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_theorical_limit(lr, data_size = 1000, batch_size = batch_size) for lr in np.linspace(.0005, lr_max)], 'k--')

    title = 'non uniformity limit vs {2} with {0} = {1}'.format(type_of_fixed, batch_size, abscissa)

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == batch_size]

    plot_save_and_show(df_plot, title, abscissa, ordinate, legend, type_of_fixed)