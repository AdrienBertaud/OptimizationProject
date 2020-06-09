# -*- coding: utf-8 -*-
import utils.results
import utils.sharpness
import utils.non_uniformity
import utils.figure

from importlib import reload
reload(utils.results)
reload(utils.figure)

from utils.results import load_results
from utils.figure import plot_sharpness_vs_learning_rate, plot_sharpness_vs_batch_size, plot_sharpness_limit, plot_results_with_fixed_value, plot_nonuniformity_limit, plot_results_with_no_fixed_value


def main():
    '''
    Plot graphs from results stored.
    '''

    results_data_frame = load_results()

    # call the different plots
    plot_sharpness_vs_learning_rate(results_data_frame, batch_size=100)
    plot_sharpness_vs_learning_rate(results_data_frame, batch_size='all')

    plot_sharpness_vs_batch_size(results_data_frame, 'sgd')
    plot_sharpness_vs_batch_size(results_data_frame, 'adagrad')

    plot_sharpness_limit(results_data_frame,
                  legend='optimizer',
                  all_values='batch size')

    plot_results_with_fixed_value(results_data_frame,
                abscissa='lr',
                ordinate='sharpness train',
                legend='optimizer',
                type_of_fixed='batch size',
                fixed=10)

    plot_nonuniformity_limit(results_data_frame,
                abscissa='lr',
                ordinate='sharpness train',
                legend='optimizer',
                type_of_fixed='batch size',
                fixed=10)

    plot_results_with_no_fixed_value(results_data_frame,
                  abscissa='lr',
                  ordinate='non uniformity train',
                  legend='optimizer',
                  all_values='batch size')


if __name__ == '__main__':
    main()