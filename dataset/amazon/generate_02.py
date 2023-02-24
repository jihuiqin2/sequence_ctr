# -*- coding:utf-8 -*-

import numpy as np
import sys

sys.path.append('../../')
import pandas as pd
import gc
import random
import pickle as pkl
from tqdm import tqdm

from utilization.data_utils import load_dataset_config

random.seed(2020)
save_data_dir = '../../data/beauty_5/'
dataset_id = 'beauty_5'
config_dataset_yaml = '../../config/dataset_config/beauty.yaml'

params = load_dataset_config(config_dataset_yaml, dataset_id)
sample_size = params['sample_size']
seq_max_length = params['seq_max_len']
padding_type = params['padding_type']
valid_size = params['valid_size']
test_size = params['test_size']


def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1, len(key) + 1)))  # todo tip:从1开始编码
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key  # 索引编码


def build_item_cate(df, data, item_cate_dict_file, statical_dict_file):
    item_cate_dict = {}
    statical_dict = {}
    for line in df.values:
        asin, categories, price, brand = line
        if asin not in item_cate_dict:
            item_cate_dict[asin] = [categories]  # todo tip:唯一key索引，item->cate数组[]

    statical_dict['user_num'] = data[0]
    statical_dict['item_num'] = data[1]
    statical_dict['cate_num'] = data[2]
    statical_dict['sample_num'] = data[3]
    statical_dict['feature_total_num'] = data[0] + data[1] + data[2] + 1

    with open(item_cate_dict_file, 'wb') as f:
        pkl.dump(item_cate_dict, f)

    with open(statical_dict_file, 'wb') as f:  # 统计
        pkl.dump(statical_dict, f)


# 分组
def gen_user_item_group(df):
    user_df = df.sort_values(['user_id', 'time']).groupby('user_id')
    user_df = user_df.filter(lambda x: len(x) >= 5).groupby('user_id')  # todo 每个用户至少有5个样本
    print("total_users:", len(user_df))
    item_df = df.sort_values(['item_id', 'time']).groupby('item_id')

    return user_df, item_df


# 正负样本
def generate_sample(user_df, item_cate_dict_file, train_file, valid_file, test_file):
    with open(item_cate_dict_file, 'rb') as f:
        item_cate_dict = pkl.load(f)
    all_items = list(item_cate_dict.keys())  # 获取所有的items

    train_data = []
    valid_data = []
    test_data = []

    # 对每个用户取最后1个数据，其中8:1:1作为训练，验证，测试
    total_data = []
    for uid, hist in tqdm(user_df):
        for index, line in enumerate(reversed(hist.values), start=1):
            if index == sample_size + 1:
                break
            uid, iid, time, rating, price, brand = line
            cid = item_cate_dict[iid][0]
            # 历史序列
            seq_item = hist['item_id'].tolist()[:-index]
            seq_item = cut_seq_list(seq_item)
            seq_item_str = "|".join(str(i) for i in seq_item)
            seq_cate = [item_cate_dict[k][0] for k in seq_item]
            seq_cate_str = "|".join(str(i) for i in seq_cate)
            # 负样本
            neg_item = create_neg_sample(hist['item_id'].tolist(), all_items)
            neg_cate = item_cate_dict[neg_item][0]
            # 正样本和负样本
            total_data.append([uid, iid, cid, 1, seq_item_str, seq_cate_str, rating, price, brand])
            total_data.append([uid, neg_item, neg_cate, 0, seq_item_str, seq_cate_str, rating, price, brand])

    # 划分数据集
    random.shuffle(total_data)
    test = int(len(total_data) * test_size)
    train = int(len(total_data) * (1 - test_size - valid_size))
    test_data += total_data[:test]
    valid_data += total_data[test:-train]
    train_data += total_data[-train:]

    columns = ['user_id', 'item_id', 'cate_id', 'label', 'seq_item', 'seq_cate', 'rating', 'price', 'brand']
    pd.DataFrame(columns=columns, data=valid_data).to_csv(valid_file, index=False)
    pd.DataFrame(columns=columns, data=test_data).to_csv(test_file, index=False)
    pd.DataFrame(columns=columns, data=train_data).to_csv(train_file, index=False)


def create_neg_sample(user_seq, items):
    remain_list = list(set(items).difference(set(user_seq)))
    return random.choice(remain_list)


def cut_seq_list(data):
    if len(data) > seq_max_length:
        data = data[-seq_max_length:]  # 取距离目标item最近的topK个
    return data


# 统计每个用户的历史序列区间
def static_each_user_range_count(user_df):
    users, hists = [], []
    print("start ...")
    for uid, hist in user_df:
        users.append(uid)
        hists.append(len(hist))

    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 2000]
    score = pd.Series(hists)
    se1 = pd.cut(score, bins)  # 排序
    print(se1.value_counts())
    df_count = se1.value_counts().reset_index()
    df_count.to_csv(save_data_dir + "user_hist_count.csv", index=False)


if __name__ == "__main__":
    # 【1】获取两个表中需要的特征数据
    reviews = pd.read_pickle(save_data_dir + 'reviews.pkl')
    meta = pd.read_pickle(save_data_dir + 'meta.pkl')
    print("统计reviews每列中非nan的值：\n", reviews.count())
    print("=============================================================")
    print("overall：", len(reviews['overall'].dropna().unique()))
    print("reviewerID：", len(reviews['reviewerID'].dropna().unique()))
    print("reviewerName", len(reviews['reviewerName'].dropna().unique()))
    print("asin", len(reviews['asin'].dropna().unique()))

    print("统计meta每列中非nan的值：\n", meta.count())
    print("=============================================================")
    print("asin：", len(meta['asin'].dropna().unique()))
    if dataset_id == 'books_5':
        meta['categories'] = meta['categories'].map(lambda x: x[0][-1])  # 只取最后一个类别
    else:
        meta['categories'] = meta['categories'].map(lambda x: x[-1][-1])  # 只取最后一个类别
    print("categories：", len(meta['categories'].dropna().unique()))
    print("price：", len(meta['price'].dropna().unique()))
    print("brand：", len(meta['brand'].dropna().unique()))

    reviews_df = reviews[['reviewerID', 'asin', 'unixReviewTime', 'overall']]
    meta_df = meta[['asin', 'categories', 'price', 'brand']]
    meta_df['brand'] = meta_df['brand'].fillna('brand')
    meta_df['price'] = meta_df['price'].fillna(0).astype(int)
    reviews_df['overall'] = reviews_df['overall'].fillna(0).astype(int)

    # todo 我们在这里做的是将数据集中的连续输入特征变换为一个分类特征
    max = int(meta_df['price'].max())
    min = int(meta_df['price'].min())
    bins = np.linspace(min, max + 1, 100)
    dot_bins = np.digitize(meta_df['price'], bins=bins)
    meta_df['price'] = dot_bins

    del reviews, meta
    gc.collect()
    print("already 01")

    # 【2】索引处理，并转为缩印
    asin_map, asin_key = build_map(meta_df, 'asin')
    brand_map, brand_key = build_map(meta_df, 'brand')
    cate_map, cate_key = build_map(meta_df, 'categories')
    revi_map, revi_key = build_map(reviews_df, 'reviewerID')

    user_count, item_count, cate_count, brand_count, example_count = \
        len(revi_map), len(asin_map), len(cate_map), len(brand_map), reviews_df.shape[0]
    print('user_count: %d\titem_count: %d\tcate_count: %d\t\texample_count: %d' %
          (user_count, item_count, cate_count, example_count))
    print("already 02")

    # 【3】转为索引
    meta_df = meta_df.sort_values('asin')
    meta_df = meta_df.reset_index(drop=True)
    reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])  # 保证reviews_df和meta_df中asin索引一致
    reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
    reviews_df = reviews_df.reset_index(drop=True)
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime', 'overall']]

    data = [len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]]
    build_item_cate(meta_df, data, save_data_dir + 'item_cate_dict.pkl',
                    save_data_dir + 'statical_dict.pkl')  # 构建item->cate对应
    print("already 03")

    df = pd.merge(reviews_df, meta_df, on='asin')
    df = df.drop(columns=['categories'])

    # 【4】转为需要的字段，分组，排序。增加categories
    df.rename(columns={'reviewerID': 'user_id', "asin": "item_id", "unixReviewTime": "time"}, inplace=True)
    user_df, item_df = gen_user_item_group(df)
    print("already 04")
    static_each_user_range_count(user_df)

    # 【5】生成正负样本，划分数据集
    generate_sample(user_df, save_data_dir + 'item_cate_dict.pkl',
                    save_data_dir + 'train.csv', save_data_dir + 'valid.csv', save_data_dir + 'test.csv')
    print("already 05")
