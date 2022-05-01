import click
import implicit
import mlflow
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from scipy.sparse import coo_matrix

from kaggle_hm.chart_model import filter_data, age_chart
from kaggle_hm.config import data_root, test_dates
from kaggle_hm.evaluation import compute_precision, collect, enrich_data


class Transformer:
    def __init__(self) -> None:
        self.item_code = {}
        self.code_item = {}

    def fit(self, ids):
        self.item_code = {item_id: code for code, item_id in enumerate(ids)}
        self.code_item = {v: k for k, v in self.item_code.items()}

    def transform(self, ids):
        return [self.item_code[_] for _ in ids]

    def inverse(self, codes):
        return [self.code_item[_] for _ in codes]


class MatrixTransformer:
    def __init__(self, cell_value):
        self.cell_values = cell_value
        self.item_transformer: Transformer = None

    def _cell_values(self, X):
        if self.cell_values == 'const':
            return X
        elif self.cell_values == 'tfidf':
            return tfidf_weight(X)
        elif self.cell_values == 'bm25':
            return bm25_weight(X)
        elif self.cell_values == 'time':
            return tfidf_weight(X)
        return X

    def fit(self, transactions: pd.DataFrame):
        self.item_transformer = Transformer()
        self.item_transformer.fit(transactions['article_id'].unique())

    def transform(self, transactions: pd.DataFrame):
        if self.cell_values == 'time':
            data = transactions['delta_weeks']
        else:
            data = np.ones(transactions.shape[0])
        ptrs = (transactions['customer_id'], transactions['article_id'].map(self.item_transformer.item_code).astype('int'))
        matrix = coo_matrix((data, ptrs))

        return self._cell_values(matrix).tocsr()


class MatrixFactorizationPipeline:
    def __init__(self, min_items, min_customers, cell_value, factors, iterations, regularization):
        self.min_items = min_items
        self.min_customers = min_customers
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.vectorizer = MatrixTransformer(cell_value)

    def fit(self, train):
        i_stats = (
            train
            .groupby('article_id', observed=True)
            .agg(
                cust_total=('customer_id', 'count')
            ).query(f'cust_total >= {self.min_customers}').reset_index()
        )
        train_items = set(i_stats['article_id'])
        print('#items', len(train_items))

        c_stats = (
            train[train['article_id'].isin(train_items)]
            .groupby('customer_id', observed=True)
            .agg(
                items_total=('article_id', 'count')
            ).query(f'items_total >= {self.min_items}').reset_index()
        )
        train_customers = set(c_stats['customer_id'])
        print('#customers', len(train_customers))

        cond = (
            train['customer_id'].isin(train_customers) &
            train['article_id'].isin(train_items)
        )
        train_transactions = train[cond].copy()

        transformer = Transformer()
        transformer.fit(train_transactions['customer_id'].unique())
        train_transactions['customer_id'] = train_transactions['customer_id'].map(transformer.item_code).astype('int')

        self.vectorizer.fit(train_transactions)
        X_train = self.vectorizer.transform(train_transactions)

        self.model = implicit.als.AlternatingLeastSquares(factors=self.factors, regularization=self.regularization,
                                                          iterations=self.iterations, calculate_training_loss=True,
                                                          use_gpu=True)
        self.model.fit(X_train)

    def predict(self, transactions, N=12):
        train_items = self.vectorizer.item_transformer.item_code.keys()
        cond = (
            transactions['article_id'].isin(train_items)
        )
        pred_transactions = transactions[cond].copy()

        transformer = Transformer()
        transformer.fit(pred_transactions['customer_id'].unique())
        pred_transactions['customer_id'] = pred_transactions['customer_id'].map(transformer.item_code).astype('int')
        X_pred = self.vectorizer.transform(pred_transactions)

        items, scores = self.model.recommend(np.arange(X_pred.shape[0]), X_pred, N=N,
                                             filter_already_liked_items=False,
                                             recalculate_user=True)

        candidates = pd.DataFrame({'item_id': items.flatten(), 'score': scores.flatten()})
        candidates['customer_id'] = (candidates.index // N).map(transformer.code_item)
        candidates['article_id'] = candidates['item_id'].map(self.vectorizer.item_transformer.code_item)

        return candidates


@click.command()
@click.option('--min-items', default=2, help='Minimal number of items per customer')
@click.option('--min-customers', default=2, help='Minimal number of customers per item')
@click.option('--factors', default=256, help='Number of latent factors')
@click.option('--regularization', default=0.01, help='Regularization parameter')
@click.option('--iterations', default=10, help='Number of iterations')
@click.option('--cell-value', default='tfidf', help='const, tfidf or bm25')
def main(min_items, min_customers, factors, regularization, iterations, cell_value):
    print('Loading data...')
    df = pd.read_parquet(data_root / 'clean' / 'transactions.parquet')
    c = pd.read_parquet(data_root / 'clean' / 'customers.parquet').set_index('customer_id')
    c['age'] = c['age'].fillna(c['age'].mean())
    c['age_group'] = pd.cut(c['age'], bins=[15, 21, 25, 30, 40, 50, 60, 100])
    df = df.merge(c, left_on='customer_id', right_index=True)

    delta = (pd.to_datetime('2020-09-08') - df['t_dat']).dt.days
    delta_weeks = 1 / (1 + np.power(delta // 7, 1))
    df['delta_weeks'] = delta_weeks

    print('Filtering data...')

    full_ds = filter_data(df, to_date='2020-09-08')
    test = filter_data(df, test_dates['start'], test_dates['end'])
    train = filter_data(df, '2020-09-01', '2020-09-08')

    print('Preparing test set...')
    results = test.groupby('customer_id', observed=True).agg(bought=('article_id', set)).reset_index()
    results = results.merge(c, left_on='customer_id', right_index=True)
    cond = ~results['customer_id'].isin(full_ds['customer_id'].unique())
    results.loc[cond, 'segment'] = 'cold'

    cond = results['customer_id'].isin(full_ds['customer_id'].unique())
    results.loc[cond, 'segment'] = 'old'

    cond = results['customer_id'].isin(train['customer_id'].unique())
    results.loc[cond, 'segment'] = 'train'

    # baseline predictions
    print('Baseline predictions...')
    top12_age_pred = age_chart(train)

    with mlflow.start_run() as run:
        pipeline = MatrixFactorizationPipeline(min_items=min_items, min_customers=min_customers, cell_value=cell_value,
                                               factors=factors, iterations=iterations, regularization=regularization)
        pipeline.fit(train)
        rec_df = pipeline.predict(full_ds)

        print('doing evaluation')
        comb = results.merge(rec_df, on='customer_id', how='left')
        comb = comb.merge(top12_age_pred, on='age_group').drop(columns=['age_group'])
        comb['prediction'] = comb['candidates'].combine_first(comb['naive_pred'])

        comb = compute_precision(comb)
        comb = enrich_data(full_ds, comb.set_index('customer_id'), c)
        segment_precision = comb.groupby('segment').agg(avg_p=('precision', 'mean'))['avg_p'].to_dict()

        collect(comb)
        mlflow.log_params({
            'min_items': min_items,
            'min_customers': min_customers,
            'cell_value': cell_value,
            'factors': factors,
            'iterations': iterations,
        })
        mlflow.log_metrics({
            'test_map12': comb['precision'].mean(),
            'als_precision': comb[~comb['candidates'].isna()]['precision'].mean(),
        })
        mlflow.log_metrics(segment_precision)
        print(segment_precision)


if __name__ == '__main__':
    main()
