import pandas as pd
from catboost import CatBoostRanker


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
    train_nodup = train.groupby(['customer_id', 'article_id'], observed=True, as_index=False).agg(article_totals=('t_dat', 'count'))
    candidates = als_candidates.merge(train_nodup[['customer_id', 'article_id', 'article_totals']], on=['customer_id', 'article_id'], how='left')

    train_nodup = train.groupby(['customer_id', 'product_code'], observed=True, as_index=False).agg(product_totals=('t_dat', 'count'))
    candidates = candidates.merge(train_nodup[['customer_id', 'product_code', 'product_totals']], on=['customer_id', 'product_code'], how='left')
    # shopping frequency

    candidates = candidates.merge(customers[['customer_id', 'age']], on='customer_id', how='left')
    candidates = candidates.merge(items[['article_id', 'product_type_name', 'colour_group_name', 'department_name', 'section_name', 'garment_group_name']], on='article_id', how='left')

    return candidates


def rank_predictions(pred_df: pd.DataFrame, model: CatBoostRanker) -> pd.DataFrame:
    pred_df['prediction'] = model.predict(pred_df[model.feature_names_])
    pred_df['rank'] = pred_df.groupby('customer_id')['prediction'].rank(ascending=False, method='first')

    relevant_items = pred_df.query('rank <= 12').sort_values(['customer_id', 'rank'], ascending=True)
    return relevant_items.groupby('customer_id', observed=True).agg(candidates=('article_id', list))


def get_item_stats(transactions, prediction_date):
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

    return item_stats.drop(columns=['min_dt', 'max_dt', 'days'])
