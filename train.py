import torch
import utils.diagnose
import utils.trainer
import utils.utils
import utils.optimizers
import utils.save

from importlib import reload
reload(utils.trainer)
reload(utils.utils)
reload(utils.diagnose)
reload(utils.optimizers)
reload(utils.save)

from utils.trainer import train
from utils.utils import load_net, load_data, eval_accuracy
from utils.diagnose import diagnose
from utils.optimizers import get_optimizer
from utils.save import save


def compute(n_samples_train=1000, n_samples_test=1000, batch_size=512, learning_rate=0.01, optimizer_name='sgd'):

    print('===> Parameters:')
    print("optimizer = ", optimizer_name)
    print("n_samples_train = ", n_samples_train)
    print("n_samples_test = ", n_samples_test)

    if batch_size > n_samples_train:
        raise ValueError('batch size should not be larger than training set size')

    if batch_size == 0:
        raise ValueError('batch size should be superior to zero')

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(training_size=n_samples_train, test_size=n_samples_test, batch_size=batch_size)

    print('===> Architecture:')
    net = load_net()
    print(net)

    print('===> optimizer:')
    optimizer = get_optimizer(net, optimizer_name, learning_rate)
    print(optimizer)

    print('===> Start training')
    num_iter = train(net, criterion, optimizer, optimizer_name, train_loader, batch_size, n_epochs=10000)

    print('===> Results: ')
    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    print('===> Diagnose: ')
    sharpness_train, non_uniformity_train, sharpness_test, non_uniformity_test = diagnose(net, criterion, optimizer, train_loader,test_loader)

    return num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness_train, non_uniformity_train, sharpness_test, non_uniformity_test


def compute_and_save(n_samples_train=1000, n_samples_test=1000, batch_size=512, learning_rate=0.01, optimizer_name='sgd'):

    num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness_train, non_uniformity_train, sharpness_test, non_uniformity_test = compute(n_samples_train, n_samples_test, batch_size, learning_rate, optimizer_name)

    save(optimizer_name, learning_rate, batch_size, num_iter, train_loss.item(), train_accuracy.item(), test_loss.item(), test_accuracy.item(), sharpness_train, non_uniformity_train, sharpness_test, non_uniformity_test)


def compute_loop(n_samples_train=1024, n_samples_test=1000, learning_rate_list = [.001, .01, .1],batch_size_list= [10,100,1000]):

    for batch_size in batch_size_list:

        for learning_rate in learning_rate_list:

            optimizerList = []

            if (batch_size == n_samples_train):
                optimizerList.append('gd')
            else:
                optimizerList.append('sgd')

            optimizerList.append('adam')
            optimizerList.append('adagrad')

            if(batch_size == n_samples_train):
                optimizerList.append('lbfgs')

            for optimizer_name in optimizerList:
                compute_and_save(n_samples_train, n_samples_train, batch_size, learning_rate, optimizer_name)


def main():
    n_samples_train=2000
    n_samples_test=2000
    learning_rate_list = [.001, .01, .1]
    batch_size_list= [10,100,1000]

    compute_loop(n_samples_train, n_samples_test, learning_rate_list, batch_size_list)


if __name__ == '__main__':
    main()
