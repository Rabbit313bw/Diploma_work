import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np



def binarize_rating(df: pl.DataFrame, values: str, min_value: float) -> pl.DataFrame:
    df_bin = df.with_columns((pl.when(pl.col(values) > min_value).then(1).otherwise(0).alias('bin_value'))).drop(values)
    df_bin = df_bin.rename({'bin_value': values})
    return df_bin



def dataset_split(dataset: pl.DataFrame, test_size=0.2):
    train_dataset = dataset.group_by('user').agg(
        pl.col('item').sort_by('timestamp'),
        pl.col('rating').sort_by('timestamp')
    ).with_columns(
        len=pl.col('item').list.len()
    ).with_columns(
        ((1 - test_size) * pl.col('len')).cast(pl.Int32).alias('train_size')
    ).with_columns(
        pl.col('item').list.slice(0, pl.col('train_size')).alias('item_train'),
        pl.col('rating').list.slice(0, pl.col('train_size')).alias('rating_train')
    ).drop(['item', 'rating', 'len', 'train_size']).rename({
        'item_train': 'item',
        'rating_train': 'rating'
    }).explode(['item', 'rating']).join(dataset, on=['user', 'item', 'rating'], how='left')
    test_dataset = dataset.join(train_dataset, on=['user', 'item', 'rating'], how='anti')
    return train_dataset, test_dataset


class OrdinalEncoder:
    def __init__(self, column: str) -> None:
        self.column = column
    def fit(self, df: pl.DataFrame) -> None:
        self._mapper = (
            df[[self.column]].unique()
            .sort(self.column)
            .with_row_index("__index__")
            .with_columns(pl.col("__index__").cast(pl.Int64))
        )
        return self
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.join(self._mapper, on=self.column, how='left')
            .drop(self.column)
            .rename({"__index__": self.column})
        )
        return df
    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.rename({self.column: "__index__"})
            .join(self._mapper, on='__index__', how='left')
        ).drop(f"__index__")
        return df
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.fit(df).transform(df)
    
    
    
class CSRConverter:
    def __init__(self, user: str, item: str, rating: str) -> None:
        self.user = user
        self.item = item
        self.rating = rating
    def fit_transform(self, coo: pl.DataFrame) -> csr_matrix:
        user_idx = coo[self.user].to_numpy()
        item_idx = coo[self.item].to_numpy()
        rating = coo[self.rating].to_numpy()
        
        n_users = user_idx.max() + 1
        n_items = item_idx.max() + 1
        
        
        user_item_coo = coo_matrix(
            (rating,(user_idx, item_idx)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        user_item_coo.sum_duplicates()
        user_item_csr = user_item_coo.tocsr()
        return user_item_csr