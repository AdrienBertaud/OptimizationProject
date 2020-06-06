# import os
# import argparse
# import json
import matplotlib.pyplot as plt
import torch
import diagnose
import src.trainer
import src.utils
import csv

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

def compute(n_samples, batch_size, learning_rate, optimizerName):
    dataset='fashionmnist'
    momentum=0.0
    n_iters=10
    logFrequency=200
    n_iters_diagnose = 10
    tol_diagnose = 1e-4

    print("optimizer = ", optimizerName)

    # criterion = torch.nn.MSELoss().cuda()
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, test_loader = load_data(dataset,
                                          training_size=n_samples,
                                          batch_size=batch_size)
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

    # torch.save(net.state_dict(), optimizerName+'.pkl')

    sharpness, non_uniformity = diagnose(net, criterion, optimizer, train_loader,test_loader, n_iters=n_iters_diagnose, n_samples=n_samples, batch_size=batch_size, tol=tol_diagnose, verbose=True)

    print("sharpness = ", sharpness)
    print("non_uniformity = ", non_uniformity)

    return num_iter, sharpness, non_uniformity

def main():

    torch.set_grad_enabled(True)

    n_samples=1000
    learning_rate_list = [.01]#, .05, .1, .5
    # gpuid ='0,'
    batch_size_list= [n_samples//2]#, n_samples//2, n_samples//4, n_samples//8, n_samples//16]

    full_optim_list = []
    full_rate_list = []
    full_batch_list = []
    num_iter_size_list = []
    full_sharpness_list = []
    full_non_unifor_list = []

    for batch_size in batch_size_list:

        if batch_size > n_samples:
            raise ValueError('batch size should not be larger than training set size')

        if batch_size == 0:
            raise ValueError('batch size should superior to zero')

        # os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

        for learning_rate in learning_rate_list:

            optimizerList = []

            if (batch_size == n_samples):
                optimizerList.append('gd')
            else:
                optimizerList.append('sgd')
            optimizerList.append('adam')
            optimizerList.append('adagrad')
            if(batch_size == n_samples):
                optimizerList.append('lbfgs')
            # optimizerList.append('adamw')

            for optimizerName in optimizerList:
                num_iter, sharpness, non_uniformity = compute(n_samples, batch_size, learning_rate, optimizerName)
                plt.scatter(sharpness, non_uniformity, label=optimizerName)
                full_optim_list.append(optimizerName)
                full_rate_list.append(learning_rate)
                full_batch_list.append(batch_size)
                full_optim_list.append(num_iter_size_list)
                full_sharpness_list.append(sharpness)
                full_non_unifor_list.append(non_uniformity)

            plt.legend(loc='best')
            plt.title('$learning learning_rate = {r}$'.format(r=learning_rate))
            plt.xlabel('sharpness')
            plt.ylabel('non_uniformity')

             # Save the figure as a PNG
            plt.savefig('fig')
            plt.savefig('batch_size_{b}_learning_learning_rate_{r}.png'.format(b=batch_size,r=learning_rate))

            plt.show()

    print("oprim : ", full_optim_list)
    print("lr : ", full_rate_list)
    print("batch sizes : ", full_batch_list)
    print("num_iter_size_list : ", num_iter_size_list)
    print("sharpness : ", full_sharpness_list)
    print("non uniformities: ", full_non_unifor_list)

    with open('results.csv', 'w') as result_file:
        csv_writer = csv.writer(result_file, delimiter=',')
        csv_writer.writerow(full_optim_list)
        csv_writer.writerow(full_rate_list)
        csv_writer.writerow(full_batch_list)
        csv_writer.writerow(num_iter_size_list)
        csv_writer.writerow(full_sharpness_list)
        csv_writer.writerow(full_non_unifor_list)

if __name__ == '__main__':
    main()
