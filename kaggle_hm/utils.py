import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import matplotlib.image as mpimg


def init_nb():
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.4f}'.format)
    np.set_printoptions(suppress=True, precision=4)
    plt.rcParams['figure.figsize'] = (10, 5)


def plot_item(c, ax=None):
    if not ax:
        ax = plt.gca()
    path = f'../data/images/{c[:3]}/{c}.jpg'
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set(title=f'{c}')
    except:
        print(f'{path} not found')


def visualize_items(items, rows=8, columns=8):
    items = items[:rows*columns]
    fig, ax = plt.subplots(rows, columns, figsize=(rows * 3, columns * 3))

    for i, c in enumerate(items):
        row, col = i // columns, i % columns
        plot_item(c, ax[row, col])


def init_logging():
    # level
    # format
    # stdout
    pass
