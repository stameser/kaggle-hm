from typing import List

import pandas as pd


def filter_data(transactions: pd.DataFrame, from_date: str = None, to_date: str = None) -> pd.DataFrame:
    if from_date is None:
        from_date = transactions['t_dat'].min()
    if to_date is None:
        to_date = transactions['t_dat'].max()

    _ = (
        (transactions['t_dat'] >= from_date) &
        (transactions['t_dat'] <= to_date)
    )

    return transactions[_]


def compute_chart(transactions: pd.DataFrame) -> List[str]:
    """
    Calculate top-12 items and return as a list
    :return:
    """
    top_12 = (
        transactions
        .groupby('article_id', observed=True)
        .agg(total_count=('customer_id', 'count'))
        .sort_values('total_count', ascending=False)[:12].reset_index()['article_id'].tolist()
    )

    return top_12
