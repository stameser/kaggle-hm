import implicit
import numpy as np
import pandas as pd
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from scipy.sparse import coo_matrix


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
        return X
        # elif self.cell_values == 'time':
        # dt = dataset.test['t_dat'].min()
        # delta = (dt - dataset.train['t_dat']).dt.days
        # delta_weeks = np.power(delta // 7, 1)
        # norm = coo_matrix(X)
        # norm.data
        # data = np.ones(dataset.test.shape[0])
        # ptrs = (self.test_customers.transform(dataset.test['customer_id']), self.item_transformer.transform(dataset.test['article_id']))
        # return #/(today - df['d_time'])

    def fit(self, transactions: pd.DataFrame):
        self.item_transformer = Transformer()
        self.item_transformer.fit(transactions['article_id'].unique())

    def transform(self, transactions: pd.DataFrame):
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

    def predict(self, transactions):
        train_items = self.vectorizer.item_transformer.item_code.keys()
        cond = (
            transactions['article_id'].isin(train_items)
        )
        pred_transactions = transactions[cond].copy()

        transformer = Transformer()
        transformer.fit(pred_transactions['customer_id'].unique())
        pred_transactions['customer_id'] = pred_transactions['customer_id'].map(transformer.item_code).astype('int')
        X_pred = self.vectorizer.transform(pred_transactions)

        items, scores = self.model.recommend(np.arange(X_pred.shape[0]), X_pred, N=12, filter_already_liked_items=False,
                                             recalculate_user=True)
        rec_df = pd.DataFrame(items)
        rec_cols = np.arange(12)
        rec_df['candidates'] = rec_df[rec_cols].apply(
            lambda item_list: self.vectorizer.item_transformer.inverse(item_list), axis=1)  # todo MAP?
        rec_df['customer_id'] = transformer.inverse(rec_df.index)

        return rec_df


def hello_world():
    print('hi there')


if __name__ == '__main__':
    pipeline = MatrixFactorizationPipeline(min_items=1, min_customers=1, cell_value='tfidf', factors=256, iterations=5, regularization=0.01)
    pipeline.fit()