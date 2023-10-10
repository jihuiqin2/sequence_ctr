import logging
import sys
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

sys.path.append('../')
import tensorflow as tf
import random
import numpy as np

import model
from utilization.data_utils import load_config_file, load_csv, train_model, test_model, load_pkl

if __name__ == '__main__':
    # load config
    model_id = 'xgboost_beauty'
    config_model_yaml = '../config/model_config/XGBoost.yaml'
    config_dataset_yaml = '../config/dataset_config/electronics.yaml'

    config_params = load_config_file(model_id, config_model_yaml, config_dataset_yaml)
    tf.set_random_seed(config_params['seed'])
    np.random.seed(config_params['seed'])
    random.seed(config_params['seed'])
    os.environ['TF_DETERMINISTIC_OPS'] = str(config_params['seed'])

    # load dataset
    train_data, valid_data, test_data = load_csv(config_params['train_file']), load_csv(
        config_params['valid_file']), load_csv(config_params['test_file'])
    statical_dict = load_pkl(config_params['statical_dict_file'])
    eval_iter_num = (len(train_data) // config_params['batch_size']) if len(train_data) % config_params[
        'batch_size'] == 0 else (len(train_data) // config_params['batch_size'] + 1)

    logging.info(
        "user_num:{}, item_num:{}, cate_num:{}, sample_num:{}, train_data:{}, valid_data:{}, test_data:{} ".format(
            statical_dict['user_num'],
            statical_dict['item_num'],
            statical_dict['cate_num'],
            statical_dict['sample_num'],
            len(train_data),
            len(valid_data),
            len(test_data)))
    # 2. 定义 XGBoost 参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
    }
    num_round = config_params['epochs']

    X1_train, y1_train = [[row[0], row[1], row[2], row[-1], row[-2]] for row in train_data], [[row[3]] for row in
                                                                                              train_data]

    X1_test, y1_test = [[row[0], row[1], row[2], row[-1], row[-2]] for row in test_data], [[row[3]] for row in
                                                                                           test_data]

    X1_valid, y1_valid = [[row[0], row[1], row[2], row[-1], row[-2]] for row in valid_data], [[row[3]] for row in
                                                                                              valid_data]

    X = X1_train + X1_test + X1_valid
    y = y1_train + y1_test + y1_valid
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 将数据转换为 DMatrix 格式，XGBoost 的特定输入格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(params, dtrain, num_round)

    # 4. 使用模型进行预测
    y_pred_proba = bst.predict(dtest)
    y_pred = np.round(y_pred_proba)  # 将概率转换为二分类预测

    # 5. 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred)

    print('Accuracy:', accuracy)
    print('logloss:', logloss)
    print('auc:', auc)
