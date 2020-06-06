# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 16:43:05 2020

@author: berta
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    plot_lr(0.01)

def plot_lr(lr):

    if not os.path.exists('results2.csv'):
        return

    df = pd.read_csv('results2.csv', sep = ',')

    plt.scatter(df[df['lr'] == 0.01]['sharpness'], df[df['optimizer'] == 'sgd']['non uniformity'])

    plt.xlabel('sharpness')
    plt.ylabel('non uniformity')

    # plt.legend(loc='best')
    plt.title('$learning learning_rate = {r}$'.format(r=lr))

    # Save the figure as a PNG
    plt.savefig('fig')
    plt.savefig('learning_learning_rate_{r}.png'.format(r=lr))

    plt.show()

if __name__ == '__main__':
    main()