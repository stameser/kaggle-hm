import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
import sys

from kaggle_hm.config import data_root


def init_nb():
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:.4f}'.format)
    np.set_printoptions(suppress=True, precision=4)
    plt.rcParams['figure.figsize'] = (10, 5)
    sns.set_style('white')


def plot_item(c, ax=None):
    if not ax:
        ax = plt.gca()
    path = f'{data_root}/images/{c[:3]}/{c}.jpg'
    try:
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set(title=f'{c}')
    except:
        print(f'{path} not found')


def visualize_items(items, rows=8, columns=8):
    items = items[:rows * columns]
    fig, ax = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))

    for i, c in enumerate(items):
        row, col = i // columns, i % columns
        plot_item(c, ax[row, col])


def init_logging():
    level = 'INFO'
    logger = logging.getLogger('kaggle_hm')
    logger.setLevel(level)
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s'))
    logger.addHandler(h)
