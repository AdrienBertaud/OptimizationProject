import torch
import numpy as np
import utils.net
import utils.optimizers
import utils.data
import utils.trainer
import utils.accuracy
import utils.sharpness
import utils.non_uniformity
import utils.results

from importlib import reload
reload(utils.net)
reload(utils.optimizers)
reload(utils.data)
reload(utils.trainer)
reload(utils.accuracy)
reload(utils.sharpness)
reload(utils.non_uniformity)
reload(utils.results)

from utils.net import load_net, save_net
from utils.optimizers import load_optimizer
from utils.data import load_data
from utils.trainer import train
from utils.accuracy import eval_accuracy
from utils.sharpness import eval_sharpness
from utils.non_uniformity import eval_non_uniformity
from utils.results import save_results_to_csv


def train_and_eval(train_size=1000, test_size=2000, batch_size=100, learning_rate=0.01, optimizer_name='sgd'):
    '''
    Train and evaluate with a given set of optimizer, learning rate and batch size.
    '''

    if batch_size > train_size:
        raise ValueError('batch size should not be larger than training set size')

    if batch_size == 0:
        raise ValueError('batch size should be superior to zero')

    loss_function = torch.nn.CrossEntropyLoss()

    print("load data for training")
    train_loader  = load_data(data_size=train_size, batch_size=batch_size)

    print("load data for testing")
    test_loader = load_data(data_size=test_size, batch_size=test_size)

    net = load_net()

    optimizer = load_optimizer(net, optimizer_name, learning_rate)

    num_iter, duration = train(net, loss_function, optimizer, optimizer_name, train_loader, batch_size)

    # save network, in case of future need
    save_net(net, '%s_lr%d_batch%d_on_%d'%(optimizer_name, learning_rate, batch_size, train_size))

    train_loss, train_accuracy = eval_accuracy(net, loss_function, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, loss_function, test_loader)
    print('train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    print("compute sharpness with train data")
    sharpness_train = eval_sharpness(net, loss_function, optimizer, train_loader)

    print("compute non-uniformity with train data")
    non_uniformity_train = eval_non_uniformity(net, loss_function, optimizer, train_loader)

    print("compute sharpness with test data")
    sharpness_test = 0#eval_sharpness(net, loss_function, optimizer, test_loader)

    print("compute non-uniformity with test data")
    non_uniformity_test = 0#eval_non_uniformity(net, loss_function, optimizer, test_loader)

    save_results_to_csv(optimizer_name, \
                learning_rate, \
                batch_size, \
                num_iter, \
                duration, \
                train_loss.item(), \
                train_accuracy.item(), \
                test_loss.item(), \
                test_accuracy.item(), \
                sharpness_train, \
                non_uniformity_train, \
                sharpness_test, \
                non_uniformity_test)


def train_and_eval_loop(train_size, test_size, learning_rate_list, batch_size_list, optimizer_list):
    '''
    Loops to train and evaluate with various learning rates, batch sizes and optimizers.
    '''

    for i in range(4):

        # We shuffle, because it is launched on other computers in parallel, and we want to consolidate data uniformly if loop computation is not finished.
        np.random.shuffle(batch_size_list)
        np.random.shuffle(learning_rate_list)
        np.random.shuffle(optimizer_list)

        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                for optimizer_name in optimizer_list:

                    # some optimizers have conditions on batch size
                    if optimizer_name == 'lbfgs':
                        if batch_size != train_size:
                            break

                    train_and_eval(train_size, test_size, batch_size, \
                                     learning_rate, optimizer_name)


if __name__ == '__main__':
    '''
    Train and evaluate with various learning rates, batch sizes and optimizers.
    '''
    # we only take a subset of the 60'000 data of fashion MNIST for faster computation
    train_size=1000

    # The full test data  of fashion MNIST is 10'000
    test_size=10000

    # list of the parameters we want to compare
    learning_rate_list = [.01, .025, .05, 0.075, .1]
    batch_size_list = [1000, 500, 100, 50, 25, 10, 5]
    optimizer_list = ['sgd', 'adagrad', 'lbfgs']

    # train and evaluate
    train_and_eval_loop(train_size, test_size, learning_rate_list, batch_size_list, optimizer_list)

