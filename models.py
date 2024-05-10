from implicit.als import AlternatingLeastSquares
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from preprocessing_dataset import dataset_split, OrdinalEncoder, CSRConverter, binarize_rating

# from recbole.model.general_recommender.ease import EASE as EASE_recbole
# from recbole.data import create_dataset
# from recbole.config import Config
# from recbole.data.dataset import Dataset
# from recbole.data.interaction import Interaction
from tqdm import tqdm
import torch
    
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet



class ALS:
    def __init__(self, user: str, item: str, rating: str, 
                 n_factors: int, n_iterations: int, reg: float, top_k: int,
                filter_already_liked_items: bool=True,
                binarize: bool=True) -> None:
        self.user = user
        self.item = item
        self.rating = rating
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg = reg
        self.top_k = top_k
        self.filter_already_liked_items = filter_already_liked_items
        self.binarize = binarize
        
    def _preprocessing(self, df: pl.DataFrame) -> csr_matrix:
        self.enc_user = OrdinalEncoder(self.user)
        self.enc_item = OrdinalEncoder(self.item)
        df_ord = self.enc_user.fit_transform(df)
        df_ord = self.enc_item.fit_transform(df_ord)
        if self.binarize:
            df_ord = binarize_rating(df_ord, values=self.rating, min_value=3.5)
        csr_conv = CSRConverter(user=self.user, item=self.item, rating=self.rating)
        user_item = csr_conv.fit_transform(df_ord)
        return user_item
        
    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        user_item = self._preprocessing(df)
        als = AlternatingLeastSquares(factors=self.n_factors,
                                     iterations=self.n_iterations,
                                     alpha=40.0,
                                     regularization=self.reg,
                                    calculate_training_loss=True
                                     )
        als.fit(user_item)
        user_ids = np.arange(user_item.shape[0])
        recommended_item_indices, recommended_scores = als.recommend(
        user_ids,
        user_item,
        N=self.top_k,
        filter_already_liked_items=self.filter_already_liked_items
        )
        scores_df = pl.DataFrame({
            self.user: pl.Series(user_ids, dtype=pl.Int64),
            self.item: pl.Series(recommended_item_indices, dtype=pl.List(pl.Int64)),
            'scores': pl.Series(recommended_scores, dtype=pl.List(pl.Float32))
        })
        scores_df = scores_df.explode(self.item, 'scores')
        scores_df = self.enc_user.inverse_transform(scores_df)
        scores_df = self.enc_item.inverse_transform(scores_df)
        return scores_df
    
    
    
    
    
    
    
# class RecDataset:
#     def __init__(self, user: str, item: str, rating: str,
#                  timestamp: str,
#                  model: str,
#                  config_parametrs: dict,
#                  dataset_name: str) -> None:
#         self.user = user
#         self.item = item
#         self.rating = rating
#         self.dataset_name = dataset_name
#         self.model = model
#         self.timestamp = timestamp
#         self.config_parametrs = config_parametrs
#     def _make_config(self, df: pl.DataFrame) -> Config:
#         self.config_parametrs['data_path'] = 'datasets/' + self.dataset_name + '/'
#         self.config_parametrs['USER_ID_FIELD'] = self.user
#         self.config_parametrs['ITEM_ID_FIELD'] = self.item
#         self.config_parametrs['use_gpu'] = False
#         self.config_parametrs['TIME_FIELD'] = self.timestamp
#         self.config_parametrs['RATING_FIELD'] = self.rating
#         self.config_parametrs['load_col'] = {'inter': df.columns}
#         config = Config(model=self.model, dataset='tmp_train', config_dict=self.config_parametrs)
#         return config
    
#     def _create_inter_file(self, df: pl.DataFrame):
#         df.rename({self.user: self.user + ':token', self.item: self.item + ':token', 
#               self.rating: self.rating + ':float', self.timestamp: self.timestamp + ':float'}).with_columns(
#         pl.col(self.rating + ':float').cast(pl.Float64),
#         pl.col(self.timestamp + ':float').cast(pl.Float64) * 1e9
#         ).to_pandas().reset_index(drop=True).to_csv(
#         'datasets/' + self.dataset_name + '/' + 'tmp_train'+ '/' + 'tmp_train.inter', sep='\t', index=False)
        
#     def fit_transform(self, df: pl.DataFrame) -> (Config, Dataset):
#         config = self._make_config(df)
#         self._create_inter_file(df)
#         dataset = create_dataset(config=config)
#         return config, dataset
        
        
    

# class EASE:
#     def __init__(self, user: str, item:str, rating: str, dataset_name: str,
#                 l2_reg: float=1.0, top_k: int=10, timestamp: str='timestamp', 
#                  filter_already_liked_items: bool=True,
#                 binarize: bool=True) -> None:
#         self.user = user
#         self.item = item
#         self.rating = rating
#         self.l2_reg = l2_reg
#         self.top_k = top_k
#         self.timestamp = timestamp
#         self.dataset_name = dataset_name
#         self.filter_already_liked_items = filter_already_liked_items
#         self.binarize = binarize
#     def _preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
#         self.enc_user = OrdinalEncoder(self.user)
#         self.enc_item = OrdinalEncoder(self.item)
#         df_ord = self.enc_user.fit_transform(df)
#         df_ord = self.enc_item.fit_transform(df_ord)
#         if self.binarize:
#             df_ord = binarize_rating(df_ord, values=self.rating, min_value=3.5)
#         return df_ord
#     def _create_dataset(self, df: pl.DataFrame) -> Dataset:
#         parametrs = {'reg_weight': self.l2_reg}
#         make_dataset = RecDataset(user=self.user, item=self.item, rating=self.rating,
#                                   timestamp=self.timestamp,
#                                   model='EASE',
#                                   config_parametrs=parametrs,
#                                   dataset_name=self.dataset_name)
#         config, dataset = make_dataset.fit_transform(df)
#         self.config = config
#         return dataset
#     def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
#         df_ord = self._preprocessing(df)
#         dataset = self._create_dataset(df_ord)
#         self.ease = EASE_recbole(self.config, dataset)
#         print('EASE is fitted =================>')
#         n_users = df_ord.select(pl.col(self.user).unique()).shape[0]
        
#         pred_scores = torch.from_numpy((self.ease.interaction_matrix[1:, 1:] @ self.ease.item_similarity[1:, 1:]))
#         sorted_scores, sorted_items = torch.sort(pred_scores, 1, True)
#         if self.filter_already_liked_items:
#             result = pl.DataFrame({self.user: list(range(0, n_users)), self.item: sorted_items.numpy().tolist(),
#                                'scores': sorted_scores.numpy().tolist()})
#             result = result.explode([self.item, 'scores'])
#             result = result.join(df_ord, on=[self.user, self.item], how='anti')
#             result = result.sort(pl.col('scores'), descending=True).group_by(self.user).agg(
#                         [self.item, 'scores']).with_columns(
#                 pl.col(self.item).list.slice(0, self.top_k), pl.col('scores').list.slice(0, self.top_k))
#             result = result.explode([self.item, 'scores'])
#         else:
#             result = pl.DataFrame({self.user: list(range(0, n_users)), self.item: sorted_items[:, :self.top_k].numpy().tolist(), 'scores': sorted_scores[:, :self.top_k].numpy().tolist()})
#             result = result.explode([self.item, 'scores'])
#         result = self.enc_user.inverse_transform(result)
#         result = self.enc_item.inverse_transform(result)
#         return result
    
    

class EASE:
    def __init__(self, user: str, item:str, rating: str,
                l2_reg: float=1.0, top_k: int=10, timestamp: str='timestamp', 
                 filter_already_liked_items: bool=True,
                binarize: bool=True, 
                n_bathes = 5) -> None:
        self.user = user
        self.item = item
        self.rating = rating
        self.l2_reg = l2_reg
        self.top_k = top_k
        self.timestamp = timestamp
        self.filter_already_liked_items = filter_already_liked_items
        self.binarize = binarize
        self.n_bathes = n_bathes
    def _preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        self.enc_user = OrdinalEncoder(self.user)
        self.enc_item = OrdinalEncoder(self.item)
        df_ord = self.enc_user.fit_transform(df)
        df_ord = self.enc_item.fit_transform(df_ord)
        if self.binarize:
            df_ord = binarize_rating(df_ord, values=self.rating, min_value=3.5)
        csr_conv = CSRConverter(user=self.user, item=self.item, rating=self.rating)
        user_item = csr_conv.fit_transform(df_ord)
        return user_item
    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        X = self._preprocessing(df)
        print("EASE is fitting =========>")
        G = X.T @ X
        G += self.l2_reg * sp.identity(G.shape[0]).astype(np.float32)
        G = G.todense()
        
        P = np.linalg.inv(G)
        B = np.nan_to_num(P / (-np.diag(P)))
        np.fill_diagonal(B, 0.0)
        self.item_similarity = B
        self.interaction_matrix = X
        print('EASE is fitted =================>')
        n_users = X.shape[0]
        user_in_batch = n_users // self.n_bathes
        rec = pl.DataFrame({'user': [], 'item': [], 'scores':[]})
        for j, i in tqdm(enumerate(range(0, n_users, user_in_batch))):
            last = (j + 1) * user_in_batch
            if j == self.n_bathes:
                last = n_users
            pred_scores = (self.interaction_matrix[i:last] @ self.item_similarity)
            if self.filter_already_liked_items:
                pred_scores = (~np.bool_(X[i:last].toarray())) * np.asarray(pred_scores)
            sorted_scores, sorted_items = torch.sort(torch.from_numpy(pred_scores), 1, True)
            result = pl.DataFrame({self.user: list(range(i, last)), 
                                   self.item: sorted_items[:, :self.top_k].numpy().tolist(), 
                                   'scores': sorted_scores[:, :self.top_k].numpy().tolist()})
            result = result.explode([self.item, 'scores'])
            if i == 0:
                rec = result
                continue
            rec = pl.concat([rec, result])
        rec = self.enc_user.inverse_transform(rec)
        rec = self.enc_item.inverse_transform(rec)
        return rec
    
    
    


class SLIM:
    def __init__(self, user: str, item:str, rating: str,
                 l1_reg: float=1.0,
                l2_reg: float=1.0, top_k: int=10, timestamp: str='timestamp', 
                 filter_already_liked_items: bool=True,
                binarize: bool=True,
                n_bathes: int = 5,
                positive_only: bool=True) -> None:
        self.user = user
        self.item = item
        self.rating = rating
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.top_k = top_k
        self.timestamp = timestamp
        self.filter_already_liked_items = filter_already_liked_items
        self.binarize = binarize
        self.n_bathes = n_bathes
        self.positive_only = positive_only
    def _preprocessing(self, df: pl.DataFrame) -> pl.DataFrame:
        self.enc_user = OrdinalEncoder(self.user)
        self.enc_item = OrdinalEncoder(self.item)
        df_ord = self.enc_user.fit_transform(df)
        df_ord = self.enc_item.fit_transform(df_ord)
        if self.binarize:
            df_ord = binarize_rating(df_ord, values=self.rating, min_value=3.5)
        csr_conv = CSRConverter(user=self.user, item=self.item, rating=self.rating)
        user_item = csr_conv.fit_transform(df_ord)
        return user_item
    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        X = self._preprocessing(df)
        n_items = X.shape[1]
        self.l1_reg = self.l1_reg / (self.l1_reg + self.l2_reg)
        self.model = ElasticNet(alpha=1.0,
                               l1_ratio=self.l1_reg,
                               positive=self.positive_only,
                               fit_intercept=False,
                               copy_X=False)
        values, rows, cols = [], [], []
        
        print("ElasticNet for item =========>")
        for j in tqdm(range(n_items)):
            y = X[:, j].toarray()
            startptr = X.indptr[j]
            endptr = X.indptr[j + 1]
            bak = X.data[startptr:endptr].copy()
            X.data[startptr: endptr] = 0.0
            self.model.fit(X, y)
            nnz_idx = self.model.coef_ > 0.0
            values.extend(self.model.coef_[nnz_idx])
            rows.extend(np.arange(n_items)[nnz_idx])
            cols.extend(np.ones(nnz_idx.sum()) * j)
            
            X.data[startptr:endptr] = bak
        self.item_similarity = sp.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32).toarray()
        self.interaction_matrix = X
        print('SLIM is fitted =================>')
        n_users = X.shape[0]
        user_in_batch = n_users // self.n_bathes
        rec = pl.DataFrame({'user': [], 'item': [], 'scores':[]})
        for j, i in tqdm(enumerate(range(0, n_users, user_in_batch))):
            last = (j + 1) * user_in_batch
            if j == self.n_bathes:
                last = n_users
            pred_scores = (self.interaction_matrix[i:last] @ self.item_similarity)
            if self.filter_already_liked_items:
                pred_scores = ~np.bool_(X[i:last].toarray()) * np.asarray(pred_scores)
            sorted_scores, sorted_items = torch.sort(torch.from_numpy(pred_scores), 1, True)
            result = pl.DataFrame({self.user: list(range(i, last)), 
                                   self.item: sorted_items[:, :self.top_k].numpy().tolist(), 
                                   'scores': sorted_scores[:, :self.top_k].numpy().tolist()})
            result = result.explode([self.item, 'scores'])
            if i == 0:
                rec = result
                continue
            rec = pl.concat([rec, result])
        rec = self.enc_user.inverse_transform(rec)
        rec = self.enc_item.inverse_transform(rec)
        return rec
        
        