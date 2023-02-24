import os
import logging
import logging.config
import yaml
import glob
import json
from tqdm import tqdm
from collections import OrderedDict
import math
import tensorflow as tf
from sklearn.metrics import *
import pickle as pkl
import random
import pandas as pd

from utilization.data_loader import DataLoader


# 加载配置文件
def load_config_file(model_id, config_model_yaml, config_dataset_yaml):
    config_params = dict()
    params = dict()
    model_configs = glob.glob(os.path.join(config_model_yaml))
    for config in model_configs:
        with open(config, 'rb') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)  # 读取文件，得到字典数据
            if model_id in config_dict:
                config_params[model_id] = config_dict[model_id]
    if model_id not in config_params:
        raise ValueError("expid={} not found in config".format(model_id))

    params.update(config_params.get(model_id))
    dataset_params = load_dataset_config(config_dataset_yaml, params['dataset_id'])
    params.update(dataset_params)

    set_logger(params)  # 打印日志信息
    logging.info('Start the demo...')
    logging.info(print_to_json(params))  # 打印参数

    return params


def load_dataset_config(config_dataset_yaml, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dataset_yaml))
    if not dataset_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(dataset_configs))
    for config in dataset_configs:
        with open(config, 'rb') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))


# 划分数据集
def split_dataset(data_ddf, valid_size, test_size):
    num_samples = len(data_ddf)
    train_size = num_samples
    middle_size = train_size

    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        middle_size = train_size
        test_ddf = data_ddf[train_size:]

    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = data_ddf[train_size:middle_size]

    if valid_size > 0 or test_size > 0:
        train_ddf = data_ddf[:train_size]

    return train_ddf, valid_ddf, test_ddf


# 加载并处理数据集
def data_process(target_data_file, valid_size, test_size):
    target_data = []
    with open(target_data_file, 'r') as f:
        for line in f:
            target_data.append(line[:-1].split(','))
    random.shuffle(target_data)
    train_data, valid_data, test_data = split_dataset(target_data, valid_size, test_size)
    return train_data, valid_data, test_data


# 加载txt
def load_text(hist_sequence_file):
    hist_seq = []
    with open(hist_sequence_file, 'r') as f:
        for line in f:
            hist_seq.append(line[:-1].split(','))
    return hist_seq


# 加载csv
def load_csv(data_file):
    data_df = pd.read_csv(data_file)
    result = data_df.values.tolist()
    random.shuffle(result)
    return result


# 加载pkl
def load_pkl(dict_file):
    with open(dict_file, 'rb') as f:
        dict_data = pkl.load(f)
    return dict_data


def train_model(model, train_data, valid_data, eval_iter_num, batch_size, seq_max_len, model_id, model_root, lr,
                reg_lambda, epochs, seed, **params):
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        training_monitor = {
            'train_loss': [],
            'valid_loss': [],
            'logloss': [],
            'auc': [],
            'accuracy': []
        }

        logging.info("Start training: {} batches/epoch".format(eval_iter_num))

        flag = False
        early_stop = False
        for epoch in range(epochs):
            train_losses_step = []
            if early_stop:
                break
            logging.info("************ Epoch={} start ************".format(epoch + 1))

            step = 0
            data_loader = load_dataloader(model_id, batch_size, seq_max_len, train_data, **params)
            for batch_data in tqdm(data_loader):  # 1次循环batch_size个数
                train_loss, accuracy = model.train(sess, batch_data, lr, reg_lambda)  # 训练集
                step += 1
                if step % (eval_iter_num // 10) == 0:
                    logging.info("train_loss:{:.6f}, accuracy:{:.6f}".format(train_loss, accuracy))
                train_losses_step.append(train_loss)

                # 完成一次迭代
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    valid_loss, logloss, auc, accuracy = eval_model(model, sess, model_id, batch_size, valid_data,
                                                                    seq_max_len, reg_lambda, **params)
                    training_monitor['valid_loss'].append(valid_loss)
                    training_monitor['logloss'].append(logloss)
                    training_monitor['auc'].append(auc)
                    training_monitor['accuracy'].append(accuracy)

                    logging.info(
                        "epoch:{}, train_loss:{:.6f}, valid_loss:{:.6f}, logloss:{:.6f},  auc:{:.6f}, accuracy:{:.6f}".format(
                            epoch + 1, train_loss, valid_loss, logloss, auc, accuracy))

                    before_list_auc = training_monitor['auc']

                    # save model
                    if training_monitor['auc'][-1] >= max(before_list_auc):
                        save_model(model_id, seq_max_len, model_root, seed, sess, model)
                        print("save model...")
                        logging.info("save model...")
                        flag = True

                    print("training_monitor:", training_monitor['valid_loss'], "flag:", flag)

                    # pre ending
                    if len(training_monitor['valid_loss']) > 2 and epoch > 0 and flag:
                        if (training_monitor['valid_loss'][-1] > training_monitor['valid_loss'][-2]) and \
                                (training_monitor['valid_loss'][-2] > training_monitor['valid_loss'][-3]):
                            early_stop = True
                        if (training_monitor['valid_loss'][-2] - training_monitor['valid_loss'][-1]) <= 0.0001 and (
                                training_monitor['valid_loss'][-3] - training_monitor['valid_loss'][-2]) <= 0.0001:
                            early_stop = True

            # generate log
            if not os.path.exists(model_root + 'logs_{}_{}/'.format(model_id, seed)):
                os.makedirs(model_root + 'logs_{}_{}/'.format(model_id, seed))

        logging.info("************ training end ************")


def save_model(model_id, seq_max_len, model_root, seed, sess, model):
    model_name = '{}_{}'.format(model_id, seq_max_len)
    if not os.path.exists(model_root + 'model_{}_{}/'.format(model_id, seed)):
        os.makedirs(model_root + 'model_{}_{}/'.format(model_id, seed))
    save_path = model_root + 'model_{}_{}/{}/ckpt'.format(model_id, seed, model_name)
    model.save(sess, save_path)


def eval_model(model, sess, model_id, batch_size, eval_data, seq_max_len, reg_lambda, **params):
    preds = []
    labels = []
    losses = []
    accuracys = []

    data_loader = load_dataloader(model_id, batch_size, seq_max_len, eval_data, **params)
    for batch_data in data_loader:
        pred, label, loss, accuracy = model.eval(sess, batch_data, reg_lambda)
        preds += pred
        labels += label
        accuracys.append(accuracy)
        losses.append(loss)

    logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)
    accuracy = sum(accuracys) / len(accuracys)
    return loss, logloss, auc, accuracy


def test_model(model, test_data, batch_size, model_root, model_id, seed, seq_max_len, reg_lambda, **params):
    model_name = '{}_{}'.format(model_id, seq_max_len)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, model_root + 'model_{}_{}/{}/ckpt'.format(model_id, seed, model_name))
        logging.info("************ test start ************")
        valid_loss, logloss, auc, accuracy = eval_model(model, sess, model_id, batch_size, test_data, seq_max_len,
                                                        reg_lambda, **params)
        logging.info('test dataset of logloss:{:.6f},  auc:{:.6f}, accuracy:{:.6f}'.format(logloss, auc, accuracy))


def load_dataloader(model_id, batch_size, seq_max_len, eval_data, padding_type, **params):
    data_loader = DataLoader(batch_size, seq_max_len, eval_data, padding_type,
                             item_cate_dict_file=params['item_cate_dict_file'],
                             user_item_dict_file=None, use_neg_seq=params['use_neg_seq'],
                             use_f_num=params['user_f_num'])
    return data_loader


def set_logger(params, log_file=None):
    if log_file is None:
        model_id = params['model_id']
        model_root = params['model_root']
        log_dir = os.path.join(model_root + 'logs_{}_{}/'.format(model_id, params['seed']))
        log_file = os.path.join(log_dir, model_id + '_' + str(params['seq_max_len']) + '.log')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())
