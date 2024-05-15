from scipy.sparse import csr_matrix
import polars as pl
import numpy as np
import pandas as pd
from preprocessing_dataset import dataset_split, OrdinalEncoder, CSRConverter, binarize_rating
from models import ALS, EASE, SLIM
from metrics import recall_k, ndcg_k


class GridSearch:
    def __init__(self, model, param: dict, val_size: float=0.2,
                user: str='user',
                item: str='item',
                rating: str='rating',
                timestamp: str='timestamp',
                binirize: bool=False,
                history: bool=False):
        self.param = param 
        self.val_size = val_size
        self.model = model
        self.name_model = self.model.__class__.__name__
        self.user = user
        self.item = item
        self.rating = rating
        self.timestamp = timestamp
        self.model.binirize = binirize
        self.history = history
    def _without_cold_users(self, df: pl.DataFrame) -> pl.DataFrame:
        df_without_cold_users = df.group_by(self.user).agg([self.item, self.rating, self.timestamp]).with_columns(
            len=pl.col(self.item).list.len()
        ).filter(pl.col('len') > 1).drop('len').explode(['item', 'rating', 'timestamp'])
        return df_without_cold_users
    

            
    def _eval_metrics(self, pred: pl.DataFrame, valid: pl.DataFrame) -> dict:
        current_metrics = {}
        for k in [1, 5, 10, 50]:
            current_metrics['recall@{}'.format(k)] = recall_k(pred_df=pred, true_df=valid, k=k,
                                                             user=self.user,
                                                             item=self.item,
                                                             rating='scores'
                                                            )
            current_metrics['ndcg@{}'.format(k)] = ndcg_k(pred_df=pred, true_df=valid, k=k,
                                                         user=self.user,
                                                         item=self.item,
                                                         timestamp=self.timestamp,
                                                         rating='scores')
        return current_metrics
    
    def _check_results(self, current_metrics: dict, par1, str_par1: str, str_par2: str=None, par2=None) -> None:
        for k in [1, 5, 10, 50]:
            if current_metrics['recall@{}'.format(k)] > self.best_scores['recall@{}'.format(k)]:
                self.best_scores['recall@{}'.format(k)] = current_metrics['recall@{}'.format(k)]
                self.best_param['recall@{}'.format(k)] = {str_par1: par1}
                if str_par2 != None:
                    self.best_param['recall@{}'.format(k)][str_par2] = par2
            if current_metrics['ndcg@{}'.format(k)] > self.best_scores['ndcg@{}'.format(k)]:
                self.best_scores['ndcg@{}'.format(k)] = current_metrics['ndcg@{}'.format(k)]
                self.best_param['ndcg@{}'.format(k)] = {str_par1: par1}
                if str_par2 != None:
                    self.best_param['ndcg@{}'.format(k)][str_par2] = par2
            
    
    def _als_fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> None:
        if self.history:
            self.history_param_score['l2'] = []
            self.history_param_score['n_factors'] = []
            self.history_param_score['score'] = []
        for l2 in self.param['l2']:
            for n_factors in self.param['n_factors']:
                self.model.reg = l2
                self.model.n_factors = n_factors
                pred = self.model.fit_predict(train)
                current_metrics = self._eval_metrics(pred=pred, valid=valid)
                if self.history:
                    self.history_param_score['l2'].append(l2)
                    self.history_param_score['n_factors'].append(n_factors)
                    self.history_param_score['score'].append(current_metrics)
                self._check_results(current_metrics=current_metrics, par1=l2, str_par1='l2', par2=n_factors, str_par2='n_factors')
    def _ease_fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> None:
        if self.history:
            self.history_param_score['l2'] = []
            self.history_param_score['score'] = []
        for l2 in self.param['l2']:
            self.model.l2_reg = l2
            pred = self.model.fit_predict(train)
            current_metrics = self._eval_metrics(pred=pred, valid=valid)
            self._check_results(current_metrics=current_metrics, par1=l2, str_par1='l2')
            if self.history:
                self.history_param_score['l2'].append(l2)
                self.history_param_score['score'].append(current_metrics)
    def _slim_fit(self, train: pl.DataFrame, valid: pl.DataFrame) -> None:
        if self.history:
            self.history_param_score['l1'] = []
            self.history_param_score['l2'] = []
            self.history_param_score['score'] = []
        for l1 in self.param['l1']:
            for l2 in self.param['l2']:
                self.model.l1_reg = l1
                self.model.l2_reg = l2
                pred = self.model.fit_predict(train)
                current_metrics = self._eval_metrics(pred=pred, valid=valid)
                if self.history:
                    self.history_param_score['l1'].append(l1)
                    self.history_param_score['l2'].append(l2)
                    self.history_param_score['score'].append(current_metrics)
                self._check_results(current_metrics=current_metrics, par1=l1, str_par1='l1',
                                                    par2=l2, str_par2='l2')
    
    def fit(self, df: pl.DataFrame, history: bool=False):
        df_without_cold_users = self._without_cold_users(df)
        df_train, df_valid = dataset_split(df_without_cold_users, test_size=self.val_size)
        self.best_param = {}
        self.best_scores = {}
        for k in [1, 5, 10, 50]:
            self.best_param['recall@{}'.format(k)] = []
            self.best_param['ndcg@{}'.format(k)] = []
            self.best_scores['recall@{}'.format(k)] = 0.0
            self.best_scores['ndcg@{}'.format(k)] = 0.0
        if self.history:
            self.history_param_score = {}
        if self.name_model == 'ALS':
            self._als_fit(df_train, df_valid)
        if self.name_model == 'EASE':
            self._ease_fit(df_train, df_valid)
        if self.name_model == 'SLIM':
            self._slim_fit(df_train, df_valid)
        
        
        