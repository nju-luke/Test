# -*- coding: utf-8 -*-
# @Time    : 16-8-31 上午11:36
# @Author  : Luke
# @Software: PyCharm
import numpy
import tensorflow as tf
import collections
import numpy as np


class config:
    vocab_size = 20000
    embedding_size = 128
    batch_size = 64
    window_size = 5
    num_sampled = 8
    lr = 1
    # l2 = 0.0001


class word2vec:
    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32,
                                                 [self.config.batch_size],
                                                 name="inputs_placeholder"
                                                 )
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 [self.config.batch_size, self.config.vocab_size],
                                                 name="labels_placeholder"
                                                 )

    def creat_feedDict(self, batch):
        data = batch[0]
        labels = batch[1]
        one_hot_label = np.zeros((self.config.batch_size, self.config.vocab_size), dtype="int32")
        for i in range(len(labels)):
            one_hot_label[i][labels[i]] = 1
        feed_dict = {self.inputs_placeholder: data, self.labels_placeholder: one_hot_label}
        return feed_dict

    def add_embedding(self):
        with tf.variable_scope("embeddings"):
            self.embeddings = tf.get_variable("embeddings",
                                              initializer=tf.random_uniform(
                                                  [self.config.vocab_size, self.config.embedding_size], -1, 1)
                                              )
            # self.embeddings = tf.get_variable("embeddings",
            #                                   initializer=tf.zeros(
            #                                       [self.config.vocab_size, self.config.embedding_size])
            #                                   )
        window = tf.nn.embedding_lookup(self.embeddings, self.inputs_placeholder)
        return window

    def add_weights(self):
        with tf.variable_scope("Weights"):
            # self.W = tf.get_variable("W", initializer=tf.truncated_normal(
            #     [self.config.vocab_size, self.config.embedding_size],
            #     stddev=1.0 / tf.sqrt(self.config.embedding_size)))
            self.W = tf.get_variable("W", shape=[self.config.vocab_size, self.config.embedding_size])
            self.b = tf.get_variable("b", shape=[self.config.vocab_size],
                                     initializer=tf.constant_initializer(np.zeros(self.config.vocab_size)))

    def add_loss_op(self, window):
        logits = tf.matmul(window, self.W, transpose_b=True) + self.b
        loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.labels_placeholder, name="loss")
        loss = tf.reduce_mean(loss)
        loss += tf.nn.l2_loss(self.W, "l2loss")
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def generator(self, data, batch_size, window_size):
        assert window_size % 2 == 1
        assert batch_size % ((window_size - 1) / 2) == 0
        mid = (window_size - 1) / 2
        words_num_per_batch = batch_size / (window_size - 1)
        for window_start in xrange(0, (len(data) - 2) - words_num_per_batch, words_num_per_batch):
            yield self.generateBatch(data, window_size, mid, window_start, words_num_per_batch)

    def generateBatch(self, data, window_size, mid, window_start, words_num_per_batch):
        data_batch = []
        label_batch = []
        for j in xrange(words_num_per_batch):
            index = [ind for ind in range(window_size) if ind != mid]
            for i in index:
                data_batch.append(data[window_start + j + mid])
                label_batch.append(data[window_start + j + i])
        return data_batch, label_batch

    def loadFile(self, path):

        words = []
        with open(path, 'r') as file:
            for line in file:
                line = line.strip().split()
                words += line
        if len(set(words)) < self.config.vocab_size:
            print "Words numbers is less than vocab_size!!! \nProgram stopped!"
            exit()
        count = collections.Counter(words).most_common(self.config.vocab_size - 1)
        for word, _ in count:
            self.vocab.append(word)
        del count
        data = []
        for word in words:
            if word in self.vocab:
                index = self.vocab.index(word)
            else:
                index = 0
            data.append(index)
        return data

    def __init__(self, path):
        self.config = config
        self.vocab = ["UNK"]
        self.data = self.loadFile(path)
        self.batches = self.generator(self.data, self.config.batch_size, self.config.window_size)
        self.add_placeholders()
        window = self.add_embedding()
        self.add_weights()
        self.loss = self.add_loss_op(window)
        self.train_op = self.add_training_op(self.loss)

    def train(self, epoches):
        ini = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(ini)
            epoch = 1
            # for epoch in xrange(epoches):
            for batch in self.batches:
                # batch = self.batches.next()
                feed = self.creat_feedDict(batch)
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
                if epoch % 1000 == 0:
                    print "\nepoch {} loss={}".format(epoch, loss)
                    self.similar(sess, 10)
                epoch += 1

    def similar(self, sess, k=1):
        # wordList = ["one", "good", "job","they"]
        wordList = ["人民币", "中国", "杭州","大海"]

        #todo embedding归一化

        for word in wordList:
            ind = self.vocab.index(word)
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),1,keep_dims=True))
            embed = self.embeddings/norm
            vec = tf.nn.embedding_lookup(embed, ind)
            sim = sess.run(tf.matmul(embed, tf.reshape(vec, (-1, 1))))
            sim_inds = (-sim).argsort(axis=0)[1:10]
            sim_words = []
            for ind in sim_inds:
                sim_words.append(self.vocab[ind])
                # print self.vocab[ind]
            print word,":",sim_words


if __name__ == '__main__':
    # word2v = word2vec("data/data.txt")
    # word2v = word2vec("data/text8")
    word2v = word2vec('/media/luke/工作/Wiki/wiki00_chs_cut')
    word2v.train(10000)
    print "down"
