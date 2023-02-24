import logging
import sys
import os

sys.path.append('../')
import tensorflow as tf
import random
import numpy as np

import model
from utilization.data_utils import load_config_file, load_csv, train_model, test_model, load_pkl

if __name__ == '__main__':
    # load config
    model_id = 'cubfi_electronics'
    config_model_yaml = '../config/model_config/CUBFI.yaml'
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

    tf.reset_default_graph()
    model_class = getattr(model, config_params['model_name'])
    model = model_class(statical_dict, **config_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 训练
    train_model(model, train_data, valid_data, eval_iter_num, **config_params)

    # 测试
    test_model(model, test_data, **config_params)
