import os
import sys

sys.path.append('../')

import logging
import logging.config
import yaml
import glob
import json
from tqdm import tqdm
from collections import OrderedDict
import tensorflow as tf
from sklearn.metrics import *
import numpy

from utilization.data_loader2 import DataIterator


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


def load_text(hist_sequence_file):
    hist_seq = []
    with open(hist_sequence_file, 'r') as f:
        for line in f:
            hist_seq.append(line[:-1].split(','))
    return hist_seq


def prepare_data(input, target, maxlen=20, return_neg=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x  # 序列真实长度
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = maxlen
    neg_samples = len(noclk_seqs_mid[0][0])
    # print ("maxlen_x",maxlen_x)
    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.  # mask
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = numpy.array([[inp[0]] for inp in input])  # todo
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(
            lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, numpy.array(target), numpy.array(lengths_x)


# 训练数据
def train_model(model, train_data, test_data, eval_iter_num, seq_max_len, model_id, model_root, lr, reg_lambda,
                epochs, seed, **params):
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        training_monitor = {
            'train_loss': [],
            'test_loss': [],
            'logloss': [],
            'auc': [],
            'accuracy': []
        }
        logging.info("Start training: {} batches/epoch".format(eval_iter_num))

        early_stop = False
        flag = False
        for epoch in range(epochs):
            train_losses_step = []
            if early_stop:
                break

            logging.info("************ Epoch={} start ************".format(epoch + 1))
            step = 0
            for src, tgt in tqdm(train_data):
                uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src,
                                                                                                                tgt,
                                                                                                                seq_max_len,
                                                                                                                return_neg=True)
                data = [mid_his, cat_his, sl, uids, mids, cats, target, mid_mask, noclk_mids, noclk_cats]
                train_loss, accuracy = model.train(sess, data, lr, reg_lambda)
                step += 1
                if step % (eval_iter_num // 10) == 0:
                    logging.info("train_loss:{:.6f}, accuracy:{:.6f}".format(train_loss, accuracy))
                train_losses_step.append(train_loss)

                # train完一次，进行test
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    test_loss, logloss, auc, accuracy = eval_model(model, sess, test_data, seq_max_len, reg_lambda)
                    training_monitor['test_loss'].append(test_loss)
                    training_monitor['logloss'].append(logloss)
                    training_monitor['auc'].append(auc)
                    training_monitor['accuracy'].append(accuracy)

                    logging.info(
                        "epoch:{}, train_loss:{:.6f}, test_loss:{:.6f}, logloss:{:.6f}, auc:{:.6f}, accuracy:{:.6f}".format(
                            epoch + 1, train_loss, test_loss, logloss, auc, accuracy))

                    before_list_auc = training_monitor['auc']

                    # save model
                    if training_monitor['auc'][-1] >= max(before_list_auc):
                        save_model(model_id, seq_max_len, model_root, seed, sess, model)
                        print("save model...")
                        flag = True

                    print("training_monitor:", training_monitor['test_loss'], "flag:", flag)

                    # pre ending
                    if len(training_monitor['auc']) > 2 and epoch > 0 and flag:
                        if (training_monitor['auc'][-1] < training_monitor['auc'][-2]) and \
                                (training_monitor['auc'][-2] < training_monitor['auc'][-3]):
                            early_stop = True

            # generate log
            if not os.path.exists(model_root + 'logs_{}_{}/'.format(model_id, seed)):
                os.makedirs(model_root + 'logs_{}_{}/'.format(model_id, seed))

        logging.info("************ training end ************")


def eval_model(model, sess, test_data, seq_max_len, reg_lambda):
    preds = []
    labels = []
    losses = []
    accuracys = []
    for src, tgt in test_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src,
                                                                                                        tgt,
                                                                                                        seq_max_len,
                                                                                                        return_neg=True)
        data = [mid_his, cat_his, sl, uids, mids, cats, target, mid_mask, noclk_mids, noclk_cats]
        pred, label, loss, accuracy = model.eval(sess, data, reg_lambda)  # 训练集
        preds += pred
        labels += label
        accuracys.append(accuracy)
        losses.append(loss)

    logloss = log_loss(labels, preds)  # 交叉熵损失
    auc = roc_auc_score(labels, preds)  # 精确率
    test_loss = sum(losses) / len(losses)
    accuracy = sum(accuracys) / len(accuracys)
    return test_loss, logloss, auc, accuracy


def test_model(model, test_data, model_root, model_id, seed, seq_max_len, reg_lambda, **params):
    model_name = '{}_{}'.format(model_id, seq_max_len)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, model_root + 'model_{}_{}/{}/ckpt'.format(model_id, seed, model_name))
        logging.info("************ test start ************")
        _, logloss, auc, accuracy = eval_model(model, sess, test_data, seq_max_len, reg_lambda)
        logging.info('test dataset of logloss:{:.6f}, auc:{:.6f}, accuracy:{:.6f}'.format(logloss, auc, accuracy))


def load_dataloader(file_data, **params):
    data_loader = DataIterator(file_data, **params)
    return data_loader


def save_model(model_id, seq_max_len, model_root, seed, sess, model):
    model_name = '{}_{}'.format(model_id, seq_max_len)
    if not os.path.exists(model_root + 'model_{}_{}/'.format(model_id, seed)):
        os.makedirs(model_root + 'model_{}_{}/'.format(model_id, seed))
    save_path = model_root + 'model_{}_{}/{}/ckpt'.format(model_id, seed, model_name)
    model.save(sess, save_path)


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
