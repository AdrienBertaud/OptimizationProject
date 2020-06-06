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

    plt.legend(loc='best')
    plt.title('$learning learning_rate = {r}$'.format(r=lr))
    plt.xlabel('sharpness')
    plt.ylabel('non_uniformity')

    plt.scatter(df[df['optimizer'] == 'SGD']['sharpness'], df[df['optimizer'] == 'SGD']['non uniformity'])

    plt.scatter(df[df['optimizer'] == 'GD']['sharpness'], df[df['optimizer'] == 'GD']['non uniformity'])

    plt.xlabel('sharpness')
    plt.ylabel('non uniformity')

    # Save the figure as a PNG
    plt.savefig('fig')
    plt.savefig('learning_learning_rate_{r}.png'.format(r=lr))

if __name__ == '__main__':
    main()