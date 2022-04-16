import json
import logging
import os

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from kaggle_hm.config import data_root

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('knn_simil')


@click.group()
def cli():
    pass


def cosine_simil(u, v):
    return len(u & v) / np.sqrt(len(u) * len(v))


def calc_item_simil(item_id, cust_item_lookup, item_customer_lookup, min_similar=2):
    """
    Amazon item-to-item similarity search.
    :param item_id:
    :param cust_item_lookup:
    :param item_customer_lookup:
    :param min_similar: default 2: we want pairs that at least 2 customers bought
    :return:
    """
    neighbours = set()
    for aid in item_customer_lookup[item_id]:
        if aid not in cust_item_lookup:
            continue
        neighbours = neighbours | cust_item_lookup[aid]
    rv = item_customer_lookup[item_id]
    similarities = []
    for nid in neighbours:
        if nid == item_id:
            continue
        if nid not in item_customer_lookup:
            continue
        u = item_customer_lookup[nid]
        intersection = len(rv & u)
        if intersection < min_similar:
            continue
        r = {
            'simil': cosine_simil(rv, u),
            'len_b': len(u),
            'len_common': intersection,
            'len_a': len(rv),
            'b': nid,
            'a': item_id
        }
        similarities.append(r)

    return similarities


def similarity_wrapper(item_id, cust_item_lookup, item_customer_lookup, min_similar):
    _ = calc_item_simil(item_id, cust_item_lookup, item_customer_lookup, min_similar=min_similar)
    with open(data_root / 'clean' / 'simil' / f'{item_id}.json', 'w') as f:
        json.dump(_, f)


@cli.command('knn')
@click.option('train_start', '--train-start', help='train start')
@click.option('train_end', '--train-end', help='train end')
@click.option('min_customers', '--min-customers', help='min customers', default=1)
@click.option('min_items', '--min-items', help='min items', default=1)
@click.option('min_similar', '--min-similar', help='min similar', default=1)
def main(train_start, train_end, min_customers, min_items, min_similar):
    """
    compute knn similarities for a given date

    usage:
    python3 knn_simil.py knn --train-start 2020-08-01 --train-end 2020-09-08 --min-customers 2 --min-items 1 --min-similar 2 | tee -a $KAGGLE_HM_DATA/logs/knn.log
    python3 knn_simil.py knn --train-start 2020-09-01 --train-end 2020-09-08 --min-customers 2 --min-items 1 --min-similar 2 | tee $KAGGLE_HM_DATA/logs/knn.log
    """
    LOG.info(
        f"train_start: {train_start}, train_end: {train_end}, min_customers: {min_customers}, min_items: {min_items}")
    df = pd.read_parquet(data_root / 'clean' / 'transactions.parquet')
    df = df[df['t_dat'].between(train_start, train_end)]
    LOG.info(f"number of transactions: {df.shape[0]}")

    # build an index
    item_customer_lookup = (
        df
        .groupby('article_id', observed=True)
        .agg(
            customers=('customer_id', set)
        )
        .assign(set_size=lambda x: x['customers'].str.len())
        .query(f'set_size > {min_customers}')
    )['customers'].to_dict()

    cust_item_lookup = (
        df
        .groupby('customer_id', observed=True)
        .agg(
            items=('article_id', set)
        )
        .assign(
            set_size=lambda x: x['items'].str.len()
        )
        .query(f'set_size > {min_items}')
    )['items'].to_dict()

    LOG.info(
        f'computing knn using {len(item_customer_lookup.keys())} items and {len(cust_item_lookup.keys())} customers')

    for item_id in tqdm(item_customer_lookup.keys()):
        similarity_wrapper(item_id, cust_item_lookup, item_customer_lookup, min_similar=min_similar)

    # export results
    similarity_df = pd.DataFrame()
    for item_id in tqdm(os.listdir(data_root / 'clean' / 'simil')):
        with open(data_root / 'clean' / 'simil' / f'{item_id}') as f:
            data = json.load(f)
            if not data:
                continue
            _ = pd.DataFrame(data).query('simil > .1')  # todo param?
            similarity_df = pd.concat([similarity_df, _])

    similarity_df.to_parquet(data_root / 'clean' / 'similarity_table.parquet')
    with (data_root / 'clean' / 'simil' / 'meta.json').open() as f:
        meta = {
            'train_start': train_start,
            'train_end': train_end,
            'min_customers': min_customers,
            'min_items': min_items,
            'min_similar': min_similar
        }
        json.dump(meta, f)


if __name__ == '__main__':
    cli()
