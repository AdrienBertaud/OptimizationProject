# -*- coding: utf-8 -*-
import torch.nn as nn

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