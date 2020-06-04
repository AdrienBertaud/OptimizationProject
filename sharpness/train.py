import os
import argparse
import json
import matplotlib.pyplot as plt
import torch
import diagnose
import src.trainer
import src.utils

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

def main():

    learninRateList = [.01, .05, .1, .5]
    # gpuid ='0,'
    dataset='fashionmnist'
    n_samples=60
    batch_size_list= [n_samples, n_samples//2, n_samples//4, n_samples//8, n_samples//16]
    n_iters=50
    momentum=0.0
    logFrequency=1
    number_of_diagnose_iterations = 10

    for batch_size in batch_size_list:

        if batch_size > n_samples:
            raise ValueError('batch size should not be larger than training set size')

        # os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

        for learning_rate in learninRateList:

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

            for optimizerItem in optimizerList:

                print("optimizerItem = ", optimizerItem)

                # criterion = torch.nn.MSELoss().cuda()
                criterion = torch.nn.MSELoss()

                train_loader, test_loader = load_data(dataset,
                                                      training_size=n_samples,
                                                      batch_size=batch_size)
                net = load_net(dataset)
                optimizer = get_optimizer(net, optimizerItem, learning_rate, momentum)
                print(optimizer)

                print('===> Architecture:')
                print(net)

                print('===> Start training')
                train(net, criterion, optimizer, optimizerItem, train_loader, batch_size, n_iters, verbose=True, logFrequency=logFrequency)

                train_loss, train_accuracy = eval_accuracy(net, criterion, train_loader)
                test_loss, test_accuracy = eval_accuracy(net, criterion, test_loader)
                print('===> Solution: ')
                print('\t train loss: %.2e, acc: %.2f' % (train_loss, train_accuracy))
                print('\t test loss: %.2e, acc: %.2f' % (test_loss, test_accuracy))

                torch.save(net.state_dict(), optimizerItem+'.pkl')

                sharpness, non_uniformity = diagnose(optimizerItem+'.pkl', number_of_diagnose_iterations)

                print("sharpness = ", sharpness)
                print("non_uniformity = ", non_uniformity)

                plt.scatter(sharpness, non_uniformity, label=optimizerItem)

            plt.legend(loc='best')
            plt.title('$learning learning_rate = {r}$'.format(r=learning_rate))
            plt.xlabel('sharpness')
            plt.ylabel('non_uniformity')

            # Save the figure as a PNG
            plt.savefig('fig')
            plt.savefig('batch_size_{b}_learning_learning_rate_{r}.png'.format(b=batch_size,r=learning_rate))

            plt.show()

if __name__ == '__main__':
    main()
