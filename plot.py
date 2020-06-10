# -*- coding: utf-8 -*-
import utils.results
import utils.sharpness
import utils.non_uniformity
import utils.figure

from importlib import reload
reload(utils.results)
reload(utils.figure)

from utils.results import load_results
from utils.figure import plot_sharpness_vs_batch_size, plot_sharpness_limit, plot_nonuniformity_limit, plot_sharpness_nonuniformity_fixed_batch_size, plot_B


def main():
    '''
    Plot graphs from results stored.
    '''

    results_data_frame = load_results()

    # call the different plots
    plot_sharpness_nonuniformity_fixed_batch_size(results_data_frame, batch_size = 5, data_size = 1000)

    plot_sharpness_vs_batch_size(results_data_frame, 'sgd')
    plot_sharpness_vs_batch_size(results_data_frame, 'adagrad')

    plot_sharpness_limit(results_data_frame)

    plot_nonuniformity_limit(results_data_frame, batch_size=10)

    plot_B(results_data_frame, lr = 0.05, data_size = 10, ymax = 40)


if __name__ == '__main__':
    main()