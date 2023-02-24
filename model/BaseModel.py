import tensorflow as tf

from layer.utils import reduce_mean, prelu, dice


class BaseModel(object):
    def __init__(self, statical_dict, emb_dim, seq_max_len, user_f_num, use_neg_seq=False):
        # reset graph
        tf.reset_default_graph()

        self.use_neg_seq = use_neg_seq

        # input placeholders
        with tf.name_scope('inputs'):
            # item和cate序列
            self.item_seq_ph = tf.placeholder(tf.int32, [None, seq_max_len], name='item_seq_ph')
            self.cate_seq_ph = tf.placeholder(tf.int32, [None, seq_max_len], name='cate_seq_ph')
            # [None,]历史序列真实长度
            self.user_seq_length_ph = tf.placeholder(tf.int32, [None, ], name='user_seq_length_ph')

            # 用户信息[None,user_f_num]，item和cate[None,]
            self.target_user_ph = tf.placeholder(tf.int32, [None, user_f_num], name='target_user_ph')

            self.target_item_ph = tf.placeholder(tf.int32, [None, ], name='target_item_ph')
            self.target_cate_ph = tf.placeholder(tf.int32, [None, ], name='target_cate_ph')

            # [None,]标签
            self.label_ph = tf.placeholder(tf.float32, [None, None], name='label_ph')

            # 掩码
            self.item_mask_ph = tf.placeholder(tf.float32, [None, seq_max_len], name='mask')

            if self.use_neg_seq:
                self.item_neg_seq_ph = tf.placeholder(tf.int32, [None, None, None], name='item_neg_seq_ph')
                self.cate_neg_seq_ph = tf.placeholder(tf.int32, [None, None, None], name='cate_neg_seq_ph')

            # lr
            self.lr = tf.placeholder(tf.float32, [])
            # reg lambda
            self.reg_lambda = tf.placeholder(tf.float32, [])
            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])

        user_num, item_num, cate_num = statical_dict['user_num'] + 1, statical_dict['item_num'] + 1, \
                                       statical_dict['cate_num'] + 1

        # embedding, [1]user_f_num * emb_dim=EU  [2]cate&item 2*emb=EI
        with tf.name_scope('embedding'):
            # item
            self.iid_emb_var = tf.get_variable("iid_emb_var", [item_num, emb_dim])
            self.target_item = tf.nn.embedding_lookup(self.iid_emb_var, self.target_item_ph)  # [None,emb]
            self.item_seq = tf.nn.embedding_lookup(self.iid_emb_var, self.item_seq_ph)  # [None,SL,emb]
            if self.use_neg_seq:
                self.item_neg_seq = tf.nn.embedding_lookup(self.iid_emb_var, self.item_neg_seq_ph)  # [?,?,?,emb]

            # cate
            self.cid_emb_var = tf.get_variable("cid_emb_var", [cate_num, emb_dim])
            self.target_cate = tf.nn.embedding_lookup(self.cid_emb_var, self.target_cate_ph)  # [None,emb]
            self.cate_seq = tf.nn.embedding_lookup(self.cid_emb_var, self.cate_seq_ph)  # [None,SL,emb]
            if self.use_neg_seq:
                self.cate_neg_seq = tf.nn.embedding_lookup(self.cid_emb_var, self.cate_neg_seq_ph)  # [?,?,?,emb]

            # user
            self.uid_emb_var = tf.get_variable("uid_emb_var", [user_num, emb_dim])
            self.target_user = tf.nn.embedding_lookup(self.uid_emb_var, self.target_user_ph)  # [None,1,emb]
            self.user_emb = tf.reshape(self.target_user, [-1, user_f_num * emb_dim])  # [None,user_f_num * emb_dim] need

            # 候选集cate和item
            self.item_emb = tf.concat([self.target_item, self.target_cate], 1)  # [None,2*emb]  need
            # 序列cate和item
            self.item_his_emb = tf.concat([self.item_seq, self.cate_seq], 2)  # [None,SL,2*emb] need
            self.item_his_emb_sum = tf.reduce_sum(self.item_his_emb, 1)  # todo [None,2*emb]  need
            if self.use_neg_seq:
                self.item_neg_emb = tf.concat([self.item_neg_seq[:, :, 0, :], self.cate_neg_seq[:, :, 0, :]], -1)
                neg_shape = tf.shape(self.item_neg_seq)[1]  # todo tip 获取维度
                hist_shape = self.item_his_emb.get_shape().as_list()[-1]  # 获取维度
                self.item_neg_emb = tf.reshape(self.item_neg_emb, [-1, neg_shape, hist_shape])  # [?,?,2*emb] need

    def build_fc_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')

        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_pred = tf.nn.softmax(dnn3) + 0.00000001

    def build_logloss(self):
        self.loss = tf.losses.log_loss(self.label_ph, self.y_pred)
        # self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)  # todo
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

        if self.use_neg_seq:
            self.loss += self.aux_loss

        # optimizer and training step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)
        # tf.metrics.accuracy()  精确率
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_pred), self.label_ph), tf.float32))

    # 辅助损失函数
    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, batch_data, lr, reg_lambda):
        input_params = self.train_eval(batch_data)
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.train_step], feed_dict={
            **input_params,
            self.lr: lr,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 0.8
        })
        return loss, accuracy

    def eval(self, sess, batch_data, reg_lambda):
        input_params = self.train_eval(batch_data)
        pred, label, loss, accuracy = sess.run([self.y_pred, self.label_ph, self.loss, self.accuracy], feed_dict={
            **input_params,
            self.reg_lambda: reg_lambda,
            self.keep_prob: 1.
        })
        return pred.reshape([-1, ]).tolist(), label.reshape([-1, ]).tolist(), loss, accuracy

    def train_eval(self, batch_data):
        params = {
            self.item_seq_ph: batch_data[0],
            self.cate_seq_ph: batch_data[1],
            self.user_seq_length_ph: batch_data[2],
            self.target_user_ph: batch_data[3],
            self.target_item_ph: batch_data[4],
            self.target_cate_ph: batch_data[5],
            self.label_ph: batch_data[6],
            self.item_mask_ph: batch_data[7],
        }

        if self.use_neg_seq:
            params[self.item_neg_seq_ph] = batch_data[8]
            params[self.cate_neg_seq_ph] = batch_data[9]

        return params

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from {}'.format(path))
