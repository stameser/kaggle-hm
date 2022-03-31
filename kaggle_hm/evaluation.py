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
import mlflow

LOG = logging.getLogger(__file__)


def precision_at_k(rec_items: List[str], bought_items: List[str], k=12):
    score = 0.0
    num_hits = 0.0

    if not bought_items:
        LOG.warning(f'no ground truth')
        return 0.0

    for i, item in enumerate(rec_items):
        if item in bought_items and item not in rec_items[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(bought_items), k)


def rec_precision(data: pd.DataFrame) -> float:
    k = 12
    precisions = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        bought = data.iloc[i]['bought']
        predicted = data.iloc[i]['predicted']
        precisions[i] = precision_at_k(predicted, bought, k)

    return np.mean(precisions)


if __name__ == '__main__':
    mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment('h-and-m')

    data_root = Path(os.environ['KAGGLE_HM_DATA'])
    train_dates = {
        'start': '2018-09-01',
        'end': '2020-09-08'
    }
    test_dates = {
        'start': '2020-09-09',
        'end': '2020-09-15'
    }
    val_dates = {
        'start': '2020-09-16',
        'end': '2020-09-22'
    }
    t = pd.read_parquet(data_root / 'clean' / 'transactions.parquet')
    c = pd.read_parquet(data_root / 'clean' / 'customers.parquet')

    _ = (
            (t['t_dat'] >= train_dates['start']) &
            (t['t_dat'] <= train_dates['end'])
    )
    train = t[_]
    top_12 = (
        train
            .groupby('article_id')
            .agg(total_count=('customer_id', 'count'))
            .sort_values('total_count', ascending=False)[:12].reset_index()['article_id'].tolist()
    )

    _ = (
            (t['t_dat'] >= test_dates['start']) &
            (t['t_dat'] <= test_dates['end'])
    )
    test = t[_]
    results = test.groupby('customer_id', observed=True).agg(bought=('article_id', list))
    results['predicted'] = [top_12 for _ in range(results.shape[0])]
    test_prec = rec_precision(results)
    print(f'test: {test_prec:.4f}')

    _ = (
            (t['t_dat'] >= val_dates['start']) &
            (t['t_dat'] <= val_dates['end'])
    )
    val = t[_]
    results = val.groupby('customer_id', observed=True).agg(bought=('article_id', list))
    results['predicted'] = [top_12 for _ in range(results.shape[0])]
    val_prec = rec_precision(results)
    print(f'val: {val_prec:.4f}')

    submission = pd.read_csv(data_root / 'raw' / 'sample_submission.csv.zip')
    submission['prediction'] = [' '.join(top_12) for _ in range(submission.shape[0])]
    submission.to_csv(data_root / 'output' / 'submission.csv', index=False)
    print('done')

    mlflow.log_metrics({
        'test_map12': test_prec,
        'val_map12': val_prec
    })
    _ = data_root / 'output' / 'submission.csv'
    mlflow.log_artifact(str(_), 'submission.csv')
    mlflow.log_params({
        'train_period': train_dates,
        'test_period': test_dates,
        'val_period': val_dates,
        'train_shape': train.shape,
        'test_shape': test.shape,
        'val_shape': val.shape,
        'submission_shape': submission.shape
    })
#   kaggle c submit -f .\submission.csv -m "hello world" -c h-and-m-personalized-fashion-recommendations public 0.003
#   0.07
