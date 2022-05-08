import logging

import click
import pandas as pd
from catboost import CatBoostRanker

from kaggle_hm.chart_model import filter_data
from kaggle_hm.config import data_root, train_dates, test_dates
from kaggle_hm.utils import init_logging

init_logging()
LOG = logging.getLogger('kaggle_hm.features')


def cat_feature_prop(transactions, cat_feature):
    beta_a = 2
    beta_b = 5

    stats = (
        transactions.groupby(['customer_id', cat_feature], observed=True, as_index=False)
        .agg(
            total=('article_id', 'count'),
            uniq_total=('article_id', 'nunique'),
            money_spent=('price', 'sum')
        )
    )
    customer_totals = stats.groupby('customer_id', as_index=False, observed=True).agg(items_total=('total', 'sum'))
    stats = stats.merge(customer_totals, on='customer_id')
    stats[f'{cat_feature}_preference'] = (beta_a + stats['total']) / (beta_a + beta_b + stats['items_total'])

    return stats[['customer_id', cat_feature, f'{cat_feature}_preference']]


def calc_features(als_candidates, train: pd.DataFrame, customers: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    LOG.info('Calculating features')
    train_nodup = (
        train
        .groupby(['customer_id', 'article_id'], observed=True, as_index=False)
        .agg(article_totals=('t_dat', 'count'))
    )
    candidates = als_candidates.merge(train_nodup, on=['customer_id', 'article_id'], how='left')

    train_nodup = (
        train
        .groupby(['customer_id', 'product_code'], observed=True, as_index=False)
        .agg(product_totals=('t_dat', 'count'))
    )
    candidates = candidates.merge(train_nodup, on=['customer_id', 'product_code'], how='left')
    # shopping frequency

    candidates = candidates.merge(customers[['customer_id', 'age']], on='customer_id', how='left')

    item_features = ['article_id', 'product_type_name',
                     'colour_group_name', 'department_name', 'section_name', 'garment_group_name']
    candidates = candidates.merge(items[item_features], on='article_id', how='left')

    LOG.info('Done features')
    return candidates


def rank_predictions(pred_df: pd.DataFrame, model: CatBoostRanker) -> pd.DataFrame:
    pred_df['prediction'] = model.predict(pred_df[model.feature_names_])
    pred_df['rank'] = pred_df.groupby('customer_id')['prediction'].rank(ascending=False, method='first')

    relevant_items = pred_df.query('rank <= 12').sort_values(['customer_id', 'rank'], ascending=True)
    return relevant_items.groupby('customer_id', observed=True).agg(candidates=('article_id', list))


def get_item_stats(transactions, prediction_date):
    LOG.info('Getting item stats')
    item_stats = (
        transactions
        .groupby('article_id', observed=True, as_index=False)
        .agg(
            total_cust=('customer_id', 'count'),
            uniq_customers=('customer_id', 'nunique'),
            min_dt=('t_dat', 'min'),
            max_dt=('t_dat', 'max'),
            days=('t_dat', 'nunique'),
            avg_price=('price', 'mean'),
            avg_age=('age', 'mean')
        )
    )
    item_stats['item_age'] = 1 / (1 + (pd.to_datetime(prediction_date) - item_stats['min_dt']).dt.days)
    item_stats['daily_sales'] = item_stats['total_cust'] / item_stats['days']
    item_stats['monday_item'] = (item_stats['min_dt'].dt.dayofweek == 0).astype('int')
    item_stats['avg_per_customer'] = item_stats['total_cust'] / item_stats['uniq_customers']

    LOG.info('Done item stats')
    return item_stats.drop(columns=['min_dt', 'max_dt', 'days'])


def load_candidates(prediction_date):
    # todo try to keep all scores
    candidates = pd.read_parquet(data_root / prediction_date / 'candidates.parquet')
    candidates['method'] = 'als'
    LOG.info(f'Loaded {candidates.shape[0]} ALS candidates')
    return candidates
    semantic_candidates = pd.read_parquet(data_root / prediction_date / 'semantic_candidates.parquet')
    semantic_candidates['method'] = 'tfidf'
    LOG.info(f'Loaded {semantic_candidates.shape[0]} semantic candidates')

    rules_candidates = pd.read_parquet(data_root / prediction_date / 'association_rules.parquet')
    rules_candidates['method'] = 'rules'
    LOG.info(f'Loaded {rules_candidates.shape[0]} rules candidates')

    combined_candidates = pd.concat([candidates,
                                     semantic_candidates[['score', 'customer_id', 'article_id', 'is_relevant', 'method']],
                                     rules_candidates[['score', 'customer_id', 'article_id', 'is_relevant', 'method']]
                                     ])
    combined_candidates = combined_candidates.drop_duplicates(subset=['customer_id', 'article_id'],
                                                              keep='first')  # try last
    LOG.info(f'Deduplicated number of {combined_candidates.shape[0]} candidates')

    candidates['method'] = candidates['method'].astype('category')
    candidates = combined_candidates.sort_values('customer_id')
    return candidates


def cat_feature_wrapper(candidate_df, full_ds, cat_feature):
    LOG.info(f'Started {cat_feature}')
    section_stats = cat_feature_prop(full_ds, cat_feature)
    candidate_df = candidate_df.merge(section_stats, on=['customer_id', cat_feature], how='left')

    na_fill = 2 / (7 + candidate_df['total_items'])
    candidate_df[f'{cat_feature}_clean'] = candidate_df[f'{cat_feature}_preference'].combine_first(na_fill)
    LOG.info(f'Finished {cat_feature}')
    return candidate_df.drop(columns=[f'{cat_feature}_preference'])


@click.command()
@click.option('--prediction-date', required=True, help='Prediction date')
def main(prediction_date):
    LOG.info(f'Started prediction for {prediction_date}')
    c = pd.read_parquet(data_root / 'clean' / 'customers.parquet')
    c['age'] = c['age'].fillna(c['age'].mean())
    c['age_group'] = pd.cut(c['age'], bins=[15, 21, 25, 30, 40, 50, 60, 100])

    items = pd.read_parquet(data_root / 'clean' / 'articles.parquet')

    df = pd.read_parquet(data_root / 'clean' / 'transactions.parquet')
    df['product_code'] = df['article_id'].str[1:7].astype('int')
    df = df.merge(c[['customer_id', 'age', 'age_group']], on='customer_id')
    df = df.merge(items[['article_id', 'product_type_name', 'colour_group_name', 'department_name', 'section_name']],
                  on='article_id')

    to_date = pd.to_datetime(prediction_date)
    from_date = to_date - pd.Timedelta(days=7)

    train = filter_data(df, from_date=from_date, to_date=to_date)
    full_ds = filter_data(df, to_date=to_date)

    t_cust = set(train['customer_id'].unique())

    candidates = load_candidates(prediction_date)
    candidates['product_code'] = candidates['article_id'].str[1:7].astype('int')

    # longer period?
    recent_transactions = filter_data(df, from_date=to_date - pd.Timedelta(days=30), to_date=to_date)
    candidates = calc_features(candidates, recent_transactions, c, items)

    # longer period?
    item_stats = get_item_stats(recent_transactions, to_date)
    candidates = candidates.merge(item_stats, on='article_id', how='left')

    candidates['segment'] = 'old'
    candidates.loc[candidates['customer_id'].isin(t_cust), 'segment'] = 'train'
    candidates['segment'] = candidates['segment'].astype('category')

    # we need this for beta smoothing
    # todo shorter period of time? 30, 60, 90 days?
    customer_totals = full_ds.groupby('customer_id', observed=True, as_index=False).agg(total_items=('article_id', 'count'))
    candidates = candidates.merge(customer_totals, on='customer_id', how='left')

    for cf in ['product_type_name', 'colour_group_name', 'department_name', 'section_name']:
        candidates = cat_feature_wrapper(candidates, full_ds, cf)

    candidates.to_parquet(data_root / prediction_date / 'X.parquet')


if __name__ == '__main__':
    # combine candiates
    # compute features
    main()
