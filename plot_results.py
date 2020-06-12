# -*- coding: utf-8 -*-
import utils.results
import utils.sharpness
import utils.non_uniformity
import utils.figure

from importlib import reload
reload(utils.results)
reload(utils.figure)

from utils.results import load_results_from_csv
from utils.figure import plot_sharpness_vs_batch_size, plot_sharpness_limit, plot_nonuniformity_limit, plot_sharpness_nonuniformity_fixed_batch_size, plot_sharpness_nonuniformity_fixed_lr, plot_sharpness_vs_lr,plot_sharpness_nonuniformity_all_lr


def main():
    '''
    Plot graphs from results stored.
    '''

    results_data_frame = load_results_from_csv()

    plot_sharpness_nonuniformity_fixed_batch_size(results_data_frame, batch_size = 100, data_size = 1000, optimizer='sgd')
    plot_sharpness_nonuniformity_fixed_batch_size(results_data_frame, batch_size = 100, data_size = 1000, optimizer='adagrad')
    plot_sharpness_nonuniformity_fixed_batch_size(results_data_frame, batch_size = 1000, data_size = 1000, optimizer='lbfgs')

    plot_sharpness_nonuniformity_fixed_lr(results_data_frame, lr = 0.05, data_size = 10, optimizer = 'sgd', xmax=3, ymax = 10)
    plot_sharpness_nonuniformity_fixed_lr(results_data_frame, lr = 0.05, data_size = 10,optimizer='adagrad', xmax=1.5, ymax = 15)
    plot_sharpness_nonuniformity_all_lr(results_data_frame, lr = 0.05, data_size = 10,optimizer='lbfgs', xmax=0.3, ymax = 1)

    plot_sharpness_vs_batch_size(results_data_frame, 'sgd')
    plot_sharpness_vs_batch_size(results_data_frame, 'adagrad')

    plot_sharpness_vs_lr(results_data_frame, optimizer='adagrad')
    plot_sharpness_vs_lr(results_data_frame, optimizer='sgd')
    plot_sharpness_vs_lr(results_data_frame, optimizer='lbfgs')

    plot_sharpness_limit(results_data_frame)

    plot_nonuniformity_limit(results_data_frame, batch_size=10)


if __name__ == '__main__':
    main()