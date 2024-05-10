import polars as pl
import numpy as np


def recall_k(pred_df: pl.DataFrame, true_df: pl.DataFrame, k: int, user: str='user', 
             item: str='item',
             rating: str='rating') -> float:
    
    
    
    def recall(actual: list, predicted: list) -> float:
        act_set = set(actual)
        pred_set = set(predicted[:k])
        result = len(act_set & pred_set) / float(len(act_set))
        return result
    
    
    
    true_lists = true_df.group_by(user, maintain_order=True).agg(item).rename({item: 'act_item'})
    predict_lists = pred_df.group_by(user).agg(
        pl.col(item).sort_by(pl.col(rating), descending=True)
    ).with_columns(
        pl.col(user).cast(pl.Int64)
    )
    evaluate_data = predict_lists.join(true_lists, how='inner', on=user)
    return evaluate_data.map_rows(lambda t: recall(t[2], t[1])).mean()[0, 0]



def ndcg_k(pred_df: pl.DataFrame, true_df: pl.DataFrame, k: int, user: str='user', 
             item: str='item',
             rating: str='rating', timestamp: str='timestamp') -> float:
    
    
    
    def ndcg(actual: list, predicted: list, k: int) -> float:
        if k > len(actual):
            k = len(actual)
        act_list = np.array([actual[:k]])
        pred_list = np.array([predicted[:k]])
        ndcg_term = np.array([(2 - 1) / np.log2(i + 1) for i in range(1, k + 1)])
        dcg = np.sum(ndcg_term * (act_list == pred_list))
        idcg = np.sum(ndcg_term * np.sort((act_list == pred_list))[::-1])
        if idcg == 0.0:
            return 0.0
        result = dcg / idcg
        return result
    
    
    
    true_lists = true_df.group_by(user, maintain_order=True).agg(
        pl.col(item).sort_by(pl.col(timestamp))).rename({item: 'act_item'})
    predict_lists = pred_df.group_by(user).agg(
        pl.col(item).sort_by(pl.col(rating), descending=True)
    ).with_columns(
        pl.col(user).cast(pl.Int64)
    )
    evaluate_data = predict_lists.join(true_lists, how='inner', on=user)
    return evaluate_data.map_rows(lambda t: ndcg(t[2], t[1], k)).mean()[0, 0]