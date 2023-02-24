import pickle as pkl
import numpy as np
import random

random.seed(1111)


class DataLoader(object):
    def __init__(self, batch_size, seq_max_len, target_data, padding_type, item_cate_dict_file,
                 user_item_dict_file=None, use_neg_seq=False, label_type=2, use_f_num=0):
        self.batch_size = batch_size
        self.target_data = target_data
        self.use_neg_seq = use_neg_seq
        self.padding_type = padding_type
        self.label_type = label_type
        self.use_f_num = use_f_num

        if user_item_dict_file is not None:
            with open(user_item_dict_file, 'rb') as f:
                self.user_feat_dict = pkl.load(f)
        else:
            self.user_feat_dict = None

        # item has to have multiple feature fields
        with open(item_cate_dict_file, 'rb') as f:
            self.item_cate_dict = pkl.load(f)

        self.seq_max_len = seq_max_len
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        target_user_batch = []
        target_item_batch = []
        target_cate_batch = []
        label_batch = []
        item_seq_batch = []
        cate_seq_batch = []

        item_neg_seq_batch = []
        cate_neg_seq_batch = []
        item_seq_len_batch = []

        item_mask_batch = []  # [None,SL]
        if len(self.target_data) == self.index:
            raise StopIteration

        for i in range(self.batch_size):
            if len(self.target_data) == self.index:
                break

            line = self.target_data[self.index]
            uid, iid, cid, label, item_seq, cate_seq = line[0], line[1], line[2], line[3], \
                                                       line[4].split('|') if not self.isNaNo(line[4]) else [], \
                                                       line[5].split('|') if not self.isNaNo(line[5]) else [],
            if self.use_f_num == 1:
                target_user_batch.append([uid])  # user
            elif self.use_f_num == 2:
                target_user_batch.append([uid, line[7]])  # user, price
            if self.use_f_num == 3:
                target_user_batch.append([uid, line[7], line[8]])  # user, price, brand
            elif self.use_f_num == 4:
                target_user_batch.append([uid, line[6], line[7], line[8]])  # # user, rating, price, brand

            self.index += 1

            if self.label_type == 1:
                label_batch.append([float(label)])
            else:
                label_batch.append([float(label), 1 - float(label)])

            target_item_batch.append(iid)  # item
            target_cate_batch.append(cid)  # cate
            item_seq_len = len(item_seq)  # length

            # 掩码
            item_mask = np.ones(item_seq_len).astype('int32').tolist()
            str_item_mask = [str(i) for i in item_mask]
            if item_seq_len >= self.seq_max_len:
                item_seq_len_batch.append(self.seq_max_len)
                str_item_mask = self.split_seq_data(str_item_mask)
                cate_seq = self.split_seq_data(cate_seq)
                item_seq = self.split_seq_data(item_seq)
            else:
                zero_padding = ['0'] * (self.seq_max_len - item_seq_len)
                if self.padding_type == 'pre':  # 前补0
                    item_seq = zero_padding + item_seq
                    cate_seq = zero_padding + cate_seq
                    str_item_mask = zero_padding + str_item_mask
                    item_seq_len_batch.append(item_seq_len)
                else:  # 后补0
                    item_seq = item_seq + zero_padding
                    cate_seq = cate_seq + zero_padding
                    str_item_mask = str_item_mask + zero_padding
                    item_seq_len_batch.append(item_seq_len)

            item_mask_batch.append(str_item_mask)
            item_seq_batch.append(item_seq)
            cate_seq_batch.append(cate_seq)

            # 所有的item
            all_items = list(self.item_cate_dict.keys())
            if self.use_neg_seq:
                noclk_item_list = []
                noclk_cate_list = []
                for pos_mid in item_seq:
                    noclk_tmp_item = []
                    noclk_tmp_cate = []
                    noclk_index = 0
                    while True:
                        noclk_iid = random.randint(1, len(all_items))
                        if noclk_iid == pos_mid:
                            continue
                        noclk_tmp_item.append(noclk_iid)
                        noclk_tmp_cate.append(self.item_cate_dict[noclk_iid][0])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_item_list.append(noclk_tmp_item)
                    noclk_cate_list.append(noclk_tmp_cate)
                # 负历史序列，item和cate
                item_neg_seq_batch.append(noclk_item_list)
                cate_neg_seq_batch.append(noclk_cate_list)

        data = [item_seq_batch, cate_seq_batch, item_seq_len_batch, target_user_batch, target_item_batch,
                target_cate_batch, label_batch, item_mask_batch]

        if self.use_neg_seq:
            data.append(item_neg_seq_batch)
            data.append(cate_neg_seq_batch)

        return data

    # user对应的item
    def find_cate(self, item_seq):
        cate_list = []

        # 找item对应的cate
        for item in item_seq:
            cate_list.append(self.item_cate_dict[item][0])
        return cate_list

    def isNaNo(self, sth):
        '''
        NaN、None或者空字符串返回True，其他情况返回False
        '''
        if not sth:
            return True
        if isinstance(sth, float):
            if np.isnan(sth):
                return True
        return False

    def split_seq_data(self, data):
        return data[-self.seq_max_len:]


"""
item_seq_batch,cate_seq_batch,[None,SL]  序列
target_item_batch,target_cate_batch,[None,] 序列对应的cate和item
label_batch,item_seq_len_batch, [None,]   label和序列真实长度
target_user_batch,[None,user_f_num] 包括了id和其他的用户信息等
"""
