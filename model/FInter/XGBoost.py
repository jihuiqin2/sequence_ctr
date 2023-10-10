import tensorflow as tf
import numpy as np
from model.BaseModel import BaseModel
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class XGBoost(BaseModel):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq, **params):
        super(XGBoost, self).__init__(statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq)

        pass
