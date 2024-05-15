import polars as pl
import pandas as pd


def load_ml_dataset_1m(min_value: float=3.5) -> pl.DataFrame:
    dataset = pl.from_pandas(pd.read_table('datasets/ml-1m/ratings.dat', sep='::', encoding = "latin1", engine = "python", 
              names = ['user', 'item', 'rating', 'timestamp']))
    dataset = dataset.filter(pl.col('rating') > min_value).group_by('user').agg(['item', 'rating', 'timestamp'])
    dataset = dataset.with_columns(len=pl.col('item').list.len()).filter(pl.col('len') > 1).drop('len')
    dataset = dataset.explode(['item', 'rating', 'timestamp'])
    return dataset

def load_ml_dataset_20m(min_value: float=3.5) -> pl.DataFrame:
    dataset = pl.read_csv('datasets/ml-20m/ratings.csv')
    dataset = dataset.rename({'userId': 'user', 'movieId': 'item'})
    dataset = dataset.filter(pl.col('rating') > min_value).group_by('user').agg(['item', 'rating', 'timestamp'])
    dataset = dataset.with_columns(len=pl.col('item').list.len()).filter(pl.col('len') > 1).drop('len')
    dataset = dataset.explode(['item', 'rating', 'timestamp'])
    return dataset


def load_beauty_dataset(min_value: float=3.5) -> pl.DataFrame:
    dataset = pl.read_csv('datasets/Amazon_beauty/ratings_Beauty.csv', has_header=False).rename({'column_1': 'user',
                                                                                           'column_2': 'item',
                                                                                           'column_3': 'rating',
                                                                                           'column_4': 'timestamp'})
    dataset = dataset.filter(pl.col('rating') > min_value).group_by('user').agg(['item', 'rating', 'timestamp'])
    dataset = dataset.with_columns(len=pl.col('item').list.len()).filter(pl.col('len') > 1).drop('len')
    dataset = dataset.explode(['item', 'rating', 'timestamp'])
    return dataset

def load_sport_dataset(min_value: float=3.5) -> pl.DataFrame:
    dataset = pl.read_csv('datasets/Amazon_sports/Sports_and_Outdoors.csv', infer_schema_length=10000, has_header=False).rename({
    "column_1": 'user',
    "column_2": 'item',
    "column_3": 'rating',
    "column_4": 'timestamp'
    })
    dataset = dataset.filter(pl.col('rating') > min_value).group_by('user').agg(['item', 'rating', 'timestamp'])
    dataset = dataset.with_columns(len=pl.col('item').list.len()).filter(pl.col('len') > 1).drop('len')
    dataset = dataset.explode(['item', 'rating', 'timestamp'])
    return dataset


def load_okko_dataset(min_value: float=5.0) -> pl.DataFrame:
    dataset = pl.read_csv('datasets/Okko/ratings.csv')
    dataset = dataset.rename({'user_uid': 'user', 'element_uid': 'item', 'ts':'timestamp'})
    dataset = dataset.filter(pl.col('rating') > min_value).group_by('user').agg(['item', 'rating', 'timestamp'])
    dataset = dataset.with_columns(len=pl.col('item').list.len()).filter(pl.col('len') > 1).drop('len')
    dataset = dataset.explode(['item', 'rating', 'timestamp'])
    return dataset