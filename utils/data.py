# -*- coding: utf-8 -*-
import torch
import torchvision.datasets as dsets


class DataLoader:
    '''
    Source: https://github.com/leiwu1990/sgd.stability
    '''

    def __init__(self,X,y,batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.n_samples = len(y)
        self.idx = 0

    def __len__(self):
        length = self.n_samples // self.batch_size
        if self.n_samples > length * self.batch_size:
            length += 1
        return length

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_samples:
            self.idx = 0
            rnd_idx = torch.randperm(self.n_samples)
            self.X = self.X[rnd_idx]
            self.y = self.y[rnd_idx]

        idx_end = min(self.idx+self.batch_size, self.n_samples)
        batch_X = self.X[self.idx:idx_end]
        batch_y = self.y[self.idx:idx_end]
        self.idx = idx_end

        return batch_X,batch_y


def load_data(data_size=1000, batch_size=100):

    train_set = dsets.FashionMNIST('fashionmnist', train=True, download=True)

    train_X, train_y = train_set.data[0:data_size].float()/255, \
                     to_one_hot(train_set.targets[0:data_size])

    print("data size : ", train_X.size())

    data_loader = DataLoader(train_X, train_y, batch_size)

    print("batch size : ", data_loader.batch_size)

    print("number of batch : ", len(data_loader))

    return data_loader


def to_one_hot(labels):
    '''
    Source: https://github.com/leiwu1990/sgd.stability
    '''
    if labels.ndimension()==1:
        labels.unsqueeze_(1)
    n_samples = labels.shape[0]
    n_classes = labels.max()+1

    one_hot_labels = torch.FloatTensor(n_samples,n_classes)
    one_hot_labels.zero_()
    one_hot_labels.scatter_(1, labels, 1)

    return one_hot_labels
