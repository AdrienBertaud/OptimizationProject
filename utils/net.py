# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import os


directory = 'saved_net/'


class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.net = nn.Sequential(nn.Linear(784,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,500),
                            nn.ReLU(),
                            nn.Linear(500,10))

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        o = self.net(x)
        return o


def load_net():
    return NN()


def save_net(model, name):

    t = time.localtime()
    timestamp = time.strftime('_%b-%d-%Y_%H%M', t)
    file_name = (name + timestamp)
    save_path = (directory + file_name + '.pkl')

    if not os.path.exists(directory):
        os.makedirs(directory)

    print("save model to {}".format(save_path))

    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)

    return save_path


def reload_net(save_path):
    net = load_net()
    net.load_state_dict(torch.load(save_path))
    return net


# def get_all_stored_net():
