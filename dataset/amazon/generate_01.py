#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import gc
import random
from tqdm import tqdm

random.seed(2020)

raw_data_dir = 'D:\pythonProject\dataset\/books\/'
# raw_data_dir = '../../../dataset/beauty/5/'
save_data_dir = "../../data/beauty_5/"
reviews_file_name = 'reviews_Beauty_5.json'
meta_file_name = 'meta_Beauty.json'


def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in tqdm(fin):
            df[i] = eval(line)  # 直接针对字符串运行
            i += 1

        df = pd.DataFrame.from_dict(df, orient='index')
        return df


def json_to_pkl(raw_data_dir, save_data_dir, reviews_file_name, meta_file_name):
    # 【1】处理reviews_x，并保存为pkl
    reviews_df = to_df(raw_data_dir + reviews_file_name)
    print("reviews总数据量：", reviews_df.shape[0], "reviews列数：", list(reviews_df.columns))

    with open(save_data_dir + 'reviews.pkl', 'wb') as f:
        pickle.dump(reviews_df, f)

    unique_asin = reviews_df['asin'].unique()

    del reviews_df
    gc.collect()

    # 【2】处理meta_x，并保存为pkl，只保存reviews_x中asin有的那些数据
    meta_df = to_df(raw_data_dir + meta_file_name)
    meta_df = meta_df[meta_df['asin'].isin(unique_asin)]
    meta_df = meta_df.reset_index(drop=True)
    print("meta总数据量：", meta_df.shape[0], "meta列数：", list(meta_df.columns))

    with open(save_data_dir + 'meta.pkl', 'wb') as f:
        pickle.dump(meta_df, f)

    del meta_df
    gc.collect()


json_to_pkl(raw_data_dir, save_data_dir, reviews_file_name, meta_file_name)
