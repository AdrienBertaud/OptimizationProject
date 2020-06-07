from .models.vgg import vgg11
from .models.mnist import fnn
from .data import load_fmnist,load_cifar10
import torch

def load_net(dataset = 'fashionmnist'):
    if dataset == 'fashionmnist':
            return fnn()
    elif dataset == 'cifar10':
        return vgg11(num_classes=2)
    else:
        raise ValueError('Dataset %s is not supported'%(dataset))


def load_data(dataset='fashionmnist', training_size=60000, test_size=10000, batch_size=1000):
    if dataset == 'fashionmnist':
            return load_fmnist(training_size=training_size, test_size=test_size, batch_size=batch_size)
    elif dataset == 'cifar10':
            return load_cifar10(training_size, batch_size)
    else:
        raise ValueError('Dataset %s is not supported'%(dataset))

def eval_accuracy(model, criterion, dataloader):
    model.eval()
    n_batchs = len(dataloader)
    dataloader.idx = 0

    loss_t, acc_t = 0.0, 0.0

    with torch.no_grad():

        for i in range(n_batchs):
            inputs,targets = next(dataloader)

            targets_indices = torch.argmax(targets,1)

            y_hat = model(inputs)
            loss_t += criterion(y_hat,targets_indices)
            acc_t += accuracy(y_hat,targets_indices)

    return loss_t/n_batchs, acc_t/n_batchs

def accuracy(y_hat, targets):

    y_hat_indices = torch.argmax(y_hat,1)
    acc = (targets==y_hat_indices).float()

    return torch.mean(acc)*100.0