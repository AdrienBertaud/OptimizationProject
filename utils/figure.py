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


COLOR_LIST = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C6', 'C7', 'C7', 'C8', 'C9']
TITLE_FONT_SIZE = 13
AXIS_FONT_SIZE = 16
TICKS_FONT_SIZE = 16
LEGEND_FONT_SIZE= 16

SAVE_FIGURE = False
PLOT_TITLE = not SAVE_FIGURE #We will add title on the  LATEX report


def save_fig(fig_name, save_directory = "figures", extension='.png'):
    '''
    save figure with given name
    '''

    path = (save_directory + '/' + fig_name + extension)

    print('saving ', path)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(path, bbox_inches='tight')


def save_and_show(title, plot_title=PLOT_TITLE, save_figure=SAVE_FIGURE):

    plt.legend(loc = 'best', fontsize=LEGEND_FONT_SIZE)

    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)

    if plot_title == True:
        plt.title(title, fontsize = TITLE_FONT_SIZE)

    if save_figure == True:
        save_fig(title)

    plt.show()


def plot_results(results_data_frame, abscissa='batch size', ordinate='sharpness', legend='learning rate', type_of_fixed='optimizer', fixed = 'adagrad'):

    df_optimizer = results_data_frame[(results_data_frame[type_of_fixed] == fixed)]

    df_plot = pd.DataFrame(columns = [legend, abscissa, ordinate])

    for versus, value in df_optimizer.groupby([legend, abscissa]):

        print('\n')
        print(abscissa, value[abscissa])
        print(ordinate, value[ordinate])
        mean_value = round(value.mean()[(ordinate)],2)
        print("mean_value ", round(value.mean()[(ordinate)],2))
        print(legend, value[legend])

        df_plot = df_plot.append({legend: versus[0], abscissa: int(versus[1]), ordinate: mean_value}, ignore_index=True)

    df_plot[abscissa] = df_plot[abscissa].astype(int)
    bs_order = [str(i) for i in sorted(list(set(df_plot[abscissa])))]
    df_plot[abscissa] = df_plot[abscissa].astype(str)

    versus_legends = sorted(list(set(df_plot[legend])))

    plt.plot(bs_order, [0 for i in range(len(bs_order))], color = 'white')

    for versus_legend_iter in versus_legends:
        data = df_plot[df_plot[legend] == versus_legend_iter]
        plt.plot(data[abscissa], data[ordinate], marker='o', label = legend+'=' + str(versus_legend_iter))
        scatter_data = df_optimizer[df_optimizer[legend] == versus_legend_iter]
        scatter_data[abscissa] = scatter_data[abscissa].astype(str)
        plt.scatter(scatter_data[abscissa], scatter_data[ordinate], marker='.', alpha=0.3)

    plt.xlabel(abscissa, fontsize = AXIS_FONT_SIZE)
    plt.ylabel(ordinate, fontsize = AXIS_FONT_SIZE)
    save_and_show(fixed + ' ' + ordinate + ' vs ' + abscissa)


def plot_sharpness_vs_batch_size(df, optimizer = 'adagrad'):

    plot_results(results_data_frame = df,
                 abscissa='batch size',
                 ordinate='sharpness',
                 legend='learning rate',
                 type_of_fixed='optimizer',
                 fixed = optimizer)


def plot_sharpness_limit(results_data_frame, legend='optimizer'):

    abscissa = 'learning rate'
    ordinate = 'sharpness'

    df_plot = results_data_frame.groupby([legend])

    lr_max = df_plot[abscissa].max()

    ymax = df_plot[(ordinate)].max().max() * 5

    if math.isnan(ymax) :
        return

    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_sharpness_theorical_limit(lr) for lr in np.linspace(.0005, lr_max)], 'k--')

    plt.ylim(0, ymax)

    title = 'Sharpness vs lr with all batch sizes'

    plot_save_and_show(results_data_frame, title, abscissa, ordinate, legend)


def plot_data_frame(df_plot, abscissa, ordinate, legend):

    for legend, values in df_plot:
        plt.scatter(values[abscissa], values[(ordinate)], label = legend)

    plt.xlabel(abscissa, fontsize = AXIS_FONT_SIZE)
    plt.ylabel(ordinate, fontsize = AXIS_FONT_SIZE)


def plot_save_and_show(results_data_frame, title, abscissa='learning rate', ordinate='sharpness',  legend='optimizer', all_values='batch size'):

    df_plot = results_data_frame.groupby([legend])

    plot_data_frame(df_plot, abscissa, ordinate, legend)
    save_and_show(title)


def plot_nonuniformity_limit(results_data_frame, legend='optimizer', batch_size=100):

    abscissa='learning rate'
    ordinate='non uniformity'
    type_of_fixed='batch size'

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == batch_size].groupby([legend])

    lr_max = df_plot[abscissa].max()

    max_y_value = df_plot[(ordinate)].max().max()
    if math.isnan(max_y_value) :
        return

    max_x_value = df_plot[(abscissa)].max().max()
    if math.isnan(max_x_value) :
        return

    plt.ylim(0, 6*batch_size)
        # plt.xlim(0, 6*max_x_value)

    plt.plot([lr for lr in np.linspace(.0005, lr_max)], [get_nonuniformity_theorical_limit(lr, data_size = 1000, batch_size = batch_size) for lr in np.linspace(.0005, lr_max)], 'k--')

    title = 'Non uniformity limit vs {2} with {0} = {1}'.format(type_of_fixed, batch_size, abscissa)

    df_plot = results_data_frame[results_data_frame[type_of_fixed] == batch_size]

    plot_save_and_show(df_plot, title, abscissa, ordinate, legend, type_of_fixed)


def plot_sharpness_nonuniformity_fixed_lr(df, lr = 0.1, data_size = 1000, optimizer = 'sgd', xmax=10, ymax = 20):
    '''
    plot nonuniformity against sharpness for a fixed learning rate but different batch size
    '''

    df_lr = df[(df['learning rate'] == lr) & (df['optimizer'] == optimizer)]

    i = 0
    for batch_size in sorted(list(set(df_lr['batch size']))):
        data = df_lr[df_lr['batch size'] == batch_size]
        plt.scatter(data['sharpness'], data['non uniformity'],
                    c = COLOR_LIST[i], label = 'B='+str(batch_size))
        plt.hlines(get_nonuniformity_theorical_limit(lr, data_size = data_size, batch_size = batch_size), 0, get_sharpness_theorical_limit(lr),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            i = 9

    plt.xlabel('sharpness', fontsize = AXIS_FONT_SIZE)
    plt.ylabel('non-uniformity', fontsize = AXIS_FONT_SIZE)
    plt.ylim(0, ymax)
    plt.xlim(0, xmax)
    save_and_show(optimizer+' non-uniformity vs sharpness for learning rate = '+str(lr))


def plot_sharpness_nonuniformity_all_lr(df, lr = 0.1, data_size = 1000, optimizer = 'lbfgs', xmax=10, ymax = 20):
    '''
    plot nonuniformity against sharpness for a fixed learning rate but different batch size
    '''

    df_lr = df[(df['optimizer'] == optimizer)]

    i = 0
    for lr in sorted(list(set(df_lr['learning rate']))):
        df_data = df_lr[(df_lr['learning rate'] == lr)]

        plt.scatter(df_data['sharpness'], df_data['non uniformity'],
                    c = COLOR_LIST[i], label = str(lr))
        plt.hlines(get_nonuniformity_theorical_limit(lr, data_size = data_size, batch_size = 1000), 0, get_sharpness_theorical_limit(lr),
                   color = COLOR_LIST[i], linestyles = '--', linewidth = 3)
        i += 1
        if i > 9:
            i = 9


    plt.xlabel('sharpness', fontsize = AXIS_FONT_SIZE)
    plt.ylabel('non-uniformity', fontsize = AXIS_FONT_SIZE)
    plt.ylim(0, ymax)
    plt.xlim(0, xmax)
    save_and_show(optimizer+' non-uniformity vs sharpness')


def plot_sharpness_nonuniformity_fixed_batch_size(results_data_frame, batch_size = 10, data_size = 1000, optimizer='sgd', plot_limits=False):

    ordinate='sharpness'
    abscissa='non uniformity'

    if batch_size == 'all':
        return

    df_filter = results_data_frame[(results_data_frame['batch size'] == batch_size) &
                               (results_data_frame['optimizer'] == optimizer)]

    df_filter = df_filter.sort_values(by = 'learning rate')

    i = 0

    previous_lr = df_filter['learning rate'].min()
    label = 'lr='+str(previous_lr)
    first_time = True

    if plot_limits == True:
        min_nu_lim = 100
        min_sharpness_lim = 100
    else:
        max_nu = 0
        max_sharpness = 0

    for _, learning_rate, nu, sharpness in df_filter[['learning rate', abscissa, ordinate]].itertuples():

        print('learning_rate, nu, sharpness :')
        print(learning_rate, nu, sharpness)

        max_nu = max(nu, max_nu)
        max_sharpness = max (sharpness, max_sharpness)

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

        if plot_limits == True:
            nu_lim = round(get_nonuniformity_theorical_limit(learning_rate, data_size = data_size, batch_size = batch_size))

            sharpness_lim = round(get_sharpness_theorical_limit(learning_rate))

            print('learning_rate, nu_lim, sharpness_lim :')
            print(learning_rate, nu_lim, sharpness_lim)

            # min_nu_lim = min(min_nu_lim, nu_lim)
            # min_sharpness_lim = min(sharpness_lim, min_sharpness_lim)

            plt.hlines(sharpness_lim, 0, nu_lim, color = COLOR_LIST[i], linestyles = '--', linewidth = 3)

            plt.vlines(nu_lim, 0, sharpness_lim, colors = COLOR_LIST[i], linestyles = '--', linewidth = 3)

    plt.xlabel(abscissa, fontsize = AXIS_FONT_SIZE)
    plt.ylabel(ordinate, fontsize = AXIS_FONT_SIZE)

    if plot_limits == True:
        print('min_nu_lim, min_sharpness_lim :')
        print(min_nu_lim, min_sharpness_lim)
    else:
        plt.ylim(0, max_sharpness+1)
        plt.xlim(0, max_nu+1)

    save_and_show(optimizer + ' sharpness vs non-uniformity for batch size = '+str(batch_size))


def plot_sharpness_vs_lr(df, optimizer = 'sgd', abscissa='learning rate', ordinate='sharpness', legend='batch size'):

    df_optimizer = df[(df['optimizer'] == optimizer)]

    df_plot = pd.DataFrame(columns = [legend, abscissa, ordinate])

    for versus, value in df_optimizer.groupby([legend, abscissa]):
        df_plot = df_plot.append({legend: int(versus[0]), abscissa: versus[1], ordinate: round(value.mean()[(ordinate)],1)}, ignore_index=True)

    lr_order = [str(i) for i in sorted(list(set(df_plot[abscissa])))]
    df_plot[abscissa] = df_plot[abscissa].astype(str)

    versus_legends = sorted(list(set(df_plot[legend])))

    plt.plot(lr_order, [0 for i in range(len(lr_order))], color = 'white')

    for versus_legend_iter in versus_legends:
        data = df_plot[df_plot[legend] == versus_legend_iter]
        plt.plot(data[abscissa], data[ordinate], marker='o', label = legend+'=' + str(round(versus_legend_iter)))
        scatter_data = df_optimizer[df_optimizer[legend] == versus_legend_iter]
        scatter_data[abscissa] = scatter_data[abscissa].astype(str)
        plt.scatter(scatter_data[abscissa], scatter_data[ordinate], marker='.', alpha=0.3)

    plt.xlabel(abscissa, fontsize = AXIS_FONT_SIZE)
    plt.ylabel(ordinate, fontsize = AXIS_FONT_SIZE)
    save_and_show(optimizer + ' ' + ordinate + ' vs ' + abscissa)