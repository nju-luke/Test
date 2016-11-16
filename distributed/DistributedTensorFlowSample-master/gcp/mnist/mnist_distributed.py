#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from matplotlib import pyplot as plt
import numpy as np
import math
import time

import sys

import input_data

import tensorflow as tf

argvs = sys.argv
if len(argvs)!=2:
  print("usage: python mnist_distributed.py grpc_host_name")
  exit(-1)
grpc_host= argvs[1]

num_workers=6
num_batches=600

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# model utitlity functions

def weight_variable(name,shape):
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  return tf.get_variable(name,shape,initializer=initializer)

def bias_variable(name,shape):
  initializer = tf.constant_initializer(0.1)
  return tf.get_variable(name,shape,initializer=initializer)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# model definition

def myModel(x,keep_prob):

  W_conv1 = weight_variable("W_conv1",[5, 5, 1, 8])
  b_conv1 = bias_variable("b_conv1",[8])

  x_image = tf.reshape(x, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable("W_conv2",[5, 5, 8, 16])
  b_conv2 = bias_variable("b_conv2",[16])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable("W_fc1",[7 * 7 *16, 1024])
  b_fc1 = bias_variable("b_fc1",[1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable("W_fc2",[1024, 10])
  b_fc2 = bias_variable("b_fc2",[10])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  return y_conv

def loss(x,t,keep_prob=1.0):
  y_conv=myModel(x,keep_prob)
  cross_entropy = -tf.reduce_sum(t*tf.log(y_conv))
  return cross_entropy

def loss_and_accuracy(x,t,keep_prob=1.0):
  y_conv=myModel(x,keep_prob)
  cross_entropy = -tf.reduce_sum(t*tf.log(y_conv))
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(t,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  return cross_entropy ,accuracy

def average_gradients(tower_grads):
    """
    集まった微分を平均する。
    tensorflow/models/image/cifar10 のコードそのまま。
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# master内のpsスコープでパラメータ（variable)を作成する。
with tf.device("/job:master/task:0"):
  with tf.variable_scope("ps") as scope:
    x = tf.placeholder("float", shape=[None, 784])
    t = tf.placeholder("float", shape=[None, 10])
    keep_prob = tf.placeholder("float")

    total_loss,accuracy =loss_and_accuracy(x,t,keep_prob)
    # あとで参照できるようにコレクションしておく
    tf.add_to_collection("master_x",x)
    tf.add_to_collection("master_t",t)
    tf.add_to_collection("keep_prob",keep_prob)
    tf.add_to_collection("total_loss",total_loss)
    tf.add_to_collection("accuracy",accuracy)

# worker は、psのパラメータをreuseして微分を計算するところまでの演算を行う。
for i in range(num_workers):
    with tf.device("/job:worker%d/task:0" % i):
        with tf.variable_scope("ps",reuse=True):
            # psのパラメータをreuseして微分を演算
            x = tf.placeholder("float", shape=[None, 784])
            t = tf.placeholder("float", shape=[None, 10])
            e = loss(x,t,keep_prob=0.5)
            opt = tf.train.AdamOptimizer(1e-4)
            gv = opt.compute_gradients(e)

            # あとで参照できるようにコレクションしておく
            tf.add_to_collection("x",x)
            tf.add_to_collection("t",t)
            tf.add_to_collection("e",e)
            tf.add_to_collection("gv",gv)

# masterは、workerの計算した微分の平均をとって、パラメータを更新するところまでを行う。
with tf.device("/job:master/task:0"):
    with tf.variable_scope("ps",reuse=True) as scope:
        grads=[]
        for gv in tf.get_collection("gv"):
            grads.append(gv)
        grad= average_gradients(grads)
        opt = tf.train.AdamOptimizer()
        ag  = opt.apply_gradients(grad)



# train loop

sess = tf.Session("grpc://%s:2222" % grpc_host)
sess.run(tf.initialize_all_variables())

elapsed_time10=0.0

with sess.as_default():
    for i in range(1000):
      start = time.time()

      # バッチ作成
      x,t = mnist.train.next_batch(num_batches)
      x = x.astype(np.float32).reshape(num_batches,28*28)
      t = t.astype(np.float32).reshape(num_batches,10)

      # workerの数だけ分割する
      xs= np.split(x,num_workers)
      ts= np.split(t,num_workers)

      # x,tの組をdictionaryの形にする

      keep_prob=tf.get_collection("keep_prob")[0]
      feed_dict={keep_prob:0.5}
      for j in range(num_workers):
          feed_dict[tf.get_collection("x")[j]] = xs[j]
          feed_dict[tf.get_collection("t")[j]] = ts[j]

      # パラメータ更新
      sess.run(ag, feed_dict=feed_dict)

      elapsed_time = time.time() - start
      elapsed_time10+=elapsed_time

      if i%10 == 0:
        # トータルコスト計算
        mx = tf.get_collection("master_x")[0]
        mt = tf.get_collection("master_t")[0]
        ac = tf.get_collection("accuracy")[0]
        tl = tf.get_collection("total_loss")[0]
        train_accuracy = sess.run(ac,feed_dict={
          mx: x, mt:t, keep_prob:1.0})
        total_loss = sess.run(tl,feed_dict={
          mx: x, mt:t, keep_prob:1.0})
        print "step %05d, training accuracy %2.3f, loss %5.2f,"%(i, train_accuracy, total_loss),
        print "time %2.3f [sec/step]" % elapsed_time
        elapsed_time10=0.0

    mx = tf.get_collection("master_x")[0]
    mt = tf.get_collection("master_t")[0]
    ac = tf.get_collection("accuracy")[0]
    test_accuracy = sess.run(ac,feed_dict={mx: mnist.test.images, mt:mnist.test.labels, keep_prob:1.0})
    print "test accuracy %g"%test_accuracy
