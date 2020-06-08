import torch
import utils.diagnose
import utils.trainer
import utils.optimizers
import utils.save
import utils.data
import utils.accuracy

from importlib import reload
reload(utils.trainer)
reload(utils.diagnose)
reload(utils.optimizers)
reload(utils.save)
reload(utils.data)
reload(utils.accuracy)

from utils.trainer import train
from utils.data import load_data
from utils.net import load_net
from utils.accuracy import eval_accuracy
from utils.diagnose import diagnose
from utils.optimizers import get_optimizer
from utils.save import save_to_csv


def compute(train_size=1000, test_size=1000, batch_size=512, learning_rate=0.01, optimizer_name='sgd'):

    if batch_size > train_size:
        raise ValueError('batch size should not be larger than training set size')

    if batch_size == 0:
        raise ValueError('batch size should be superior to zero')

    loss_function = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(training_size=train_size, test_size=test_size, batch_size=batch_size)

    net = load_net()

    optimizer = get_optimizer(net, optimizer_name, learning_rate)

    num_iter = train(net, loss_function, optimizer, optimizer_name, train_loader, batch_size, n_epochs=10000)

    train_loss, train_accuracy = eval_accuracy(net, loss_function, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, loss_function, test_loader)
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    sharpness_train, non_uniformity_train = diagnose(net, loss_function, optimizer, train_loader)
    print("sharpness train = ", sharpness_train)
    print("non uniformity train = ", non_uniformity_train)
    print('\n')

    sharpness_test, non_uniformity_test = diagnose(net, loss_function, optimizer, train_loader)
    print("sharpness test = ", sharpness_test)
    print("non uniformity test = ", non_uniformity_test)
    print('\n')

    save_to_csv(optimizer_name, \
                learning_rate, \
                batch_size, \
                num_iter, \
                train_loss.item(), \
                train_accuracy.item(), \
                test_loss.item(), \
                test_accuracy.item(), \
                sharpness_train, \
                non_uniformity_train, \
                sharpness_test, \
                non_uniformity_test)


def compute_loop(train_size=1000, test_size=1000, learning_rate_list = [.001, .01, .1], batch_size_list= [10,100,1000]):

    for batch_size in batch_size_list:
        for learning_rate in learning_rate_list:

            optimizerList = []

            optimizerList.append('adam')
            optimizerList.append('adagrad')

            if (batch_size == train_size):
                optimizerList.append('gd')
                optimizerList.append('lbfgs')
            else:
                optimizerList.append('sgd')

            for optimizer_name in optimizerList:
                compute(train_size, train_size, batch_size, \
                                 learning_rate, optimizer_name)


if __name__ == '__main__':
    train_size=1000
    test_size=1000
    learning_rate_list = [ .01, .1, .001]
    batch_size_list= [10,100,1000]

    compute_loop(train_size, test_size, learning_rate_list, batch_size_list)

