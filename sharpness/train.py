import torch
import diagnose
import src.trainer
import src.utils
import optimizers

from importlib import reload
reload(src.trainer)
reload(src.utils)
reload(diagnose)
reload(optimizers)
from src.trainer import train
from src.utils import load_net, load_data, eval_accuracy
from diagnose import diagnose
from optimizers import get_optimizer

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
    num_iter = train(net, criterion, optimizer, optimizer_name, train_loader, batch_size, n_iters=10000, logFrequency=200)

    print('===> Results: ')
    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    print('===> Diagnose: ')
    sharpness, non_uniformity = diagnose(net, criterion, optimizer, train_loader,test_loader)
    print("sharpness = ", sharpness)
    print("non_uniformity = ", non_uniformity)

    return num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness, non_uniformity

def compute_and_save(n_samples_train=1000, n_samples_test=1000, batch_size=512, learning_rate=0.01, optimizer_name='sgd'):

    num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness, non_uniformity = compute(n_samples_train, n_samples_test, batch_size, learning_rate, optimizer_name)

    save(optimizer_name, learning_rate, batch_size, num_iter, train_loss.item(), train_accuracy.item(), test_loss.item(), test_loss.item(), sharpness, non_uniformity)

def compute_loop(n_samples_train=1024, learning_rate_list = [.01, .05, .1, .5],batch_size_list= [1024,512,256,128,64]):

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

            optimizerList.append('adamw')

            for optimizer_name in optimizerList:
                compute_and_save(n_samples_train, n_samples_train, batch_size, learning_rate, optimizer_name)

def main():
    n_samples_train=1000

    compute_loop(n_samples_train=n_samples_train, learning_rate_list = [.01, .05, .1, .5],batch_size_list= [n_samples_train,n_samples_train//2,n_samples_train//4,n_samples_train//8,n_samples_train//16])

if __name__ == '__main__':
    main()
