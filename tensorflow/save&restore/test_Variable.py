# -*- coding: utf-8 -*-
# @Time    : 16-8-31 下午2:57
# @Author  : Luke
# @Software: PyCharm


import tensorflow as tf
import numpy as np


def save():
    #save.py

    import tensorflow as tf

    l = 'abcdefg'
    vocabulary = [cha for cha in l]
    index = range(len(vocabulary))

    aList = range(10)
    with tf.variable_scope("embeedings"):
        embeedings = tf.Variable(
            tf.random_uniform([len(vocabulary),5],-1,1),name="embeedings")
        a = tf.Variable(aList,dtype=tf.float32,name = "a")
    embed = tf.nn.embedding_lookup(embeedings,index)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    embed = sess.run(embed)

    saver = tf.train.Saver([embeedings,a])
    print embeedings
    print a
    saver.save(sess,'./embeedings.ckpt')

    print sess.run(a)
    print embed


def restore():
    #restore.py

    l = 'abcdefg'
    vocabulary = [cha for cha in l]
    index = range(len(vocabulary))
    np.random.shuffle(index)

    with tf.variable_scope("embeedings"):
        embeedings = tf.Variable(
            tf.random_uniform([len(vocabulary),5],-1,1),name="embeedings")
        a = tf.Variable([0]*10, dtype=tf.float32, name="a")

    sess = tf.Session()

    saver = tf.train.Saver([embeedings,a])
    saver.restore(sess,'embeedings')

    print index
    embed = tf.nn.embedding_lookup(embeedings,index)

    embed = sess.run(embed)

    print sess.run(a)
    print embed

def restore1():
    with tf.variable_scope("abc"):
        saver = tf.train.import_meta_graph("embeedings.meta")
        sess = tf.Session()
        saver.restore(sess, "embeedings")
        embed = sess.graph.get_tensor_by_name("model/embeedings/embeedings:0")
        # print sess.run(embed)

    with tf.variable_scope("abc1"):
        saver = tf.train.import_meta_graph("embeedings.meta")
        sess = tf.Session()
        saver.restore(sess, "embeedings1")
        embed1 = sess.graph.get_tensor_by_name("model1/embeedings/embeedings:0")

    print sess.run(embed)
    print sess.run(embed1)

if __name__ == '__main__':
    # with tf.variable_scope("model"):
    #     save()
    # restore()
    restore1()