import os
# import argparse
# import json
import matplotlib.pyplot as plt
import torch
import diagnose
import src.trainer
import src.utils
import csv
import pandas as pd

from importlib import reload
reload(src.trainer)
reload(src.utils)
reload(diagnose)
from src.trainer import train
from src.utils import load_net, load_data, eval_accuracy
from diagnose import diagnose


def get_optimizer(net, optimizer, learning_rate, momentum):
    if optimizer == 'gd':
        return torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer == 'adagrad':
         return torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    elif optimizer == 'lbfgs':
         return torch.optim.LBFGS(net.parameters(), lr=learning_rate)
    elif optimizer == 'adamw':
         return torch.optim.AdamW(net.parameters(), lr=learning_rate)
    else:
        raise ValueError('optimizer %s is not supported'%(optimizer))

def compute(n_samples_train, batch_size, learning_rate, optimizerName):
    dataset='fashionmnist'
    momentum=0.0
    n_iters=100
    logFrequency=50
    n_iters_diagnose = 10
    tol_diagnose = 1e-4
    n_samples_test = n_samples_train

    print("optimizer = ", optimizerName)

    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(dataset, training_size=n_samples_train, test_size=n_samples_test, batch_size=batch_size)

    net = load_net(dataset)
    optimizer = get_optimizer(net, optimizerName, learning_rate, momentum)
    print(optimizer)

    print('===> Architecture:')
    print(net)

    print('===> Start training')
    num_iter = train(net, criterion, optimizer, optimizerName, train_loader, batch_size, n_iters, verbose=True, logFrequency=logFrequency)

    train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
    test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
    print('===> Solution: ')
    print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
    print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

    sharpness, non_uniformity = diagnose(net, criterion, optimizer, train_loader,test_loader, n_iters=n_iters_diagnose, tol=tol_diagnose, verbose=True)

    print("sharpness = ", sharpness)
    print("non_uniformity = ", non_uniformity)

    return num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness, non_uniformity

def main():

    torch.set_grad_enabled(True)

    n_samples_train=100
    learning_rate_list = [.01, .05, .1, .5]
    batch_size_list= [n_samples_train//2, n_samples_train//4, n_samples_train//8, n_samples_train//16]

    if os.path.exists('results2.csv'):
        df = pd.read_csv('results2.csv', sep = ',')
    else:
        df = pd.DataFrame(columns=['optimizer', 'lr', 'batch size', 'num iteration',
                                   'train loss', 'train accuracy', 'test loss', 'test accuracy',
                                   'sharpness', 'non uniformity'])
    for batch_size in batch_size_list:

        if batch_size > n_samples_train:
            raise ValueError('batch size should not be larger than training set size')

        if batch_size == 0:
            raise ValueError('batch size should superior to zero')

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

            for optimizerName in optimizerList:
                num_iter, train_loss, train_accuracy, test_loss, test_accuracy, sharpness, non_uniformity = compute(n_samples_train, batch_size, learning_rate, optimizerName)

                df = df.append({'optimizer': optimizerName, 'lr': learning_rate, 'batch size': batch_size, 'num iteration': num_iter,
                                'train loss': train_loss.item(), 'train accuracy': train_accuracy.item(), 'test loss': test_loss.item(), 'test accuracy': test_accuracy.item(),
                                'sharpness': sharpness, 'non uniformity': non_uniformity}, ignore_index = True)


            # plt.show()


    df.to_csv('results2.csv', sep = ',', index = False)


if __name__ == '__main__':
    main()
