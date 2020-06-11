# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:47:52 2020

@author: Danya
"""


def plot_sharpness_vs_lr(df, optimizer = 'sgd', abscissa='lr', ordinate='sharpness train', legend='batch size'):

    df_optimizer = df[(df['optimizer'] == optimizer)]

    df_plot = pd.DataFrame(columns = [legend, abscissa, ordinate])

    for versus, value in df_optimizer.groupby([legend, abscissa]):
        df_plot = df_plot.append({legend: int(versus[0]), abscissa: versus[1], ordinate: round(value.mean()[(ordinate)],1)}, ignore_index=True)
    
    lr_order = [str(i) for i in sorted(list(set(df_plot[abscissa])))]
    df_plot[abscissa] = df_plot[abscissa].astype(str)

    versus_legends = sorted(list(set(df_plot[legend])))

    plt.plot(lr_order, [0 for i in range(len(lr_order))], color = 'white')

    for versus_legend_iter in versus_legends:
        data = df_plot[df_plot[legend] == versus_legend_iter]
        plt.plot(data[abscissa], data[ordinate], marker='o', label = legend+'=' + str(versus_legend_iter))
        scatter_data = df_optimizer[df_optimizer[legend] == versus_legend_iter]
        scatter_data[abscissa] = scatter_data[abscissa].astype(str)
        plt.scatter(scatter_data[abscissa], scatter_data[ordinate], marker='.', alpha=0.3)

    plt.xlabel(abscissa, fontsize = ABSIS_FONT_SIZE)
    plt.ylabel(ordinate, fontsize = ABSIS_FONT_SIZE)
    plt.legend()
    save_and_show(optimizer + ' ' + ordinate + ' vs ' + abscissa)
