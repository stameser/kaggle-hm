# metric
# predictions customer_id | item_id | rank
# precision(recommended, bought)
# precision
import os
from typing import List
import numpy as np
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from kaggle_hm.config import data_root, train_dates, test_dates
from kaggle_hm.chart_model import filter_data, compute_chart

LOG = logging.getLogger(__file__)


def rel_at_k(rec_items, bought_items, k):
    return int(rec_items[k] in bought_items)


def compute_precision(predictions):
    cols = [f'rel_{i + 1}' for i in range(12)]
    predictions['len_bought'] = predictions['bought'].str.len()
    k = 12
    for i in range(12):
        predictions[f'rel_{i + 1}'] = predictions.apply(lambda _: rel_at_k(_['prediction'], _['bought'], i), axis=1)

    predictions['rel_total'] = predictions[cols].sum(axis=1)

    # vectorized version of precision@k
    predictions['precision'] = np.multiply(
        predictions[cols].cumsum(axis=1) / np.arange(1, k + 1),
        predictions[cols]
    ).sum(axis=1) / np.where(predictions['len_bought'] <= k, predictions['len_bought'], k)

    return predictions


def precision_by_usage(predictions):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    _ = predictions.groupby('hist_size')['precision'].describe()

    _['count'].plot.bar(ax=ax[0])
    _['mean'].plot.bar(ax=ax[1])
    plt.suptitle('avg precision@12 for each usage segment')

    return fig


def plot_precision_at_k(predictions):
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    cols = [f'rel_{i + 1}' for i in range(12)]

    predictions[cols].mean().plot.bar(ax=ax[0])
    predictions[cols].sum().plot.bar(ax=ax[1])
    sns.histplot(data=predictions, x='precision', log_scale=(None, True), ax=ax[2])

    plt.suptitle('avg. precision at K')

    return fig


def plot_precision_at_k_segments(predictions):
    cols = [f'rel_{i + 1}' for i in range(12)]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    predictions['hist_size'].value_counts().sort_index().plot.bar(ax=ax[0, 0])
    predictions.groupby('hist_size')[cols].mean().T.plot.bar(ax=ax[0, 1])

    predictions['age_group'].value_counts().sort_index()[:8].plot.bar(ax=ax[1, 0])
    predictions.groupby('age_group')[cols].mean()[:8].T.plot.bar(ax=ax[1, 1])

    plt.suptitle('avg precision@k by history size / age group')

    return fig


def precision_by_age(predictions):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    _ = predictions.groupby('age_group')['precision'].describe()

    _['count'].plot.bar(ax=ax[0])
    _['mean'].plot.bar(ax=ax[1])
    plt.suptitle('avg precision@12 for each age segment')

    return fig


def enrich_data(data, predictions, customers):
    """
    Add items_total, items_uniq, age_group and hist_size columns to the predictions dataframe
    """

    stats = (
        data
        .groupby('customer_id')
        .agg(
            items_total=('article_id', 'count'),
            items_uniq=('article_id', 'nunique')
        )
    )

    predictions = predictions.join(stats, how='left')
    predictions[['items_total', 'items_uniq']] = predictions[['items_total', 'items_uniq']].fillna(0)
    predictions = predictions.join(customers[['age_group']], how='left')

    bins = [-0.1, 0, 5, 20, 50, 100, 10_000]
    labels = ['0', '1-5', '5-20', '20-50', '50-100', '100+']
    predictions['hist_size'] = pd.cut(predictions['items_total'], bins=bins, labels=labels)

    return predictions


def collect(predictions):
    _ = predictions.query('len_bought <= 12').pivot_table(
        index='len_bought',
        columns='rel_total',
        values='items_uniq',
        aggfunc='count'
    ).fillna(0).astype('int')

    _.to_csv('recall.csv')
    mlflow.log_artifact('recall.csv', 'recall.csv')

    _ = plot_precision_at_k(predictions)
    mlflow.log_figure(_, 'precision.png')

    _ = plot_precision_at_k_segments(predictions)
    mlflow.log_figure(_, 'precision_segments.png')

    _ = precision_by_usage(predictions)
    mlflow.log_figure(_, 'precision_by_usage.png')

    _ = precision_by_age(predictions)
    mlflow.log_figure(_, 'precision_by_segment.png')
    plt.close('all')


# find "cold" users and show they
# explore random users vis. items, highlight relevant items

if __name__ == '__main__':
    # mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment('h-and-m')

    t = pd.read_parquet(data_root / 'clean' / 'transactions.parquet')
    c = pd.read_parquet(data_root / 'clean' / 'customers.parquet').set_index('customer_id')
    c['age_group'] = pd.cut(c['age'], bins=[16, 21, 26, 30, 40, 50, 60, 100])

    train = filter_data(t, train_dates['start'], train_dates['end'])
    top_12 = compute_chart(train)

    test = filter_data(t, test_dates['start'], test_dates['end'])
    results = test.groupby('customer_id', observed=True).agg(bought=('article_id', set))
    results['prediction'] = [top_12 for _ in range(results.shape[0])]

    results = compute_precision(results)
    results = enrich_data(train, results, c)
    test_prec = results['precision'].mean()
    print(f'test: {test_prec:.4f}')
    collect(results)

    latest = filter_data(t, from_date='2020-09-01')
    top_12 = compute_chart(latest)

    submission = pd.read_csv(data_root / 'raw' / 'sample_submission.csv')
    submission['prediction'] = [' '.join(top_12) for _ in range(submission.shape[0])]
    submission.to_csv(data_root / 'output' / 'submission.csv', index=False)
    print('done')

    mlflow.log_metrics({
        'test_map12': test_prec,
    })
    _ = data_root / 'output' / 'submission.csv'
    mlflow.log_artifact(str(_), 'submission.csv')
    mlflow.log_params({
        'train_period': train_dates,
        'test_period': test_dates,
        'train_shape': train.shape,
        'test_shape': test.shape,
        'submission_shape': submission.shape
    })
    # todo: `train` model on full available data before generating predictions for the submission
    # we have more users -> we start with item-to-item similarities; else user-to-user; user-to-user more unstable;
#   kaggle c submit -f .\submission.csv -m "hello world" -c h-and-m-personalized-fashion-recommendations public 0.003
#   0.07
