#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import time

np.random.seed(0)
tf.set_random_seed(0)

def get_batch(n):
    x = np.random.random(n)
    y = np.exp(x)
    return x,y
def leaky_relu(x,alpha=0.2):
    return tf.maximum(alpha*x,x)

x_ = tf.placeholder(tf.float32, shape=[None, 1])
t_ = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_uniform([1,16], 0.0, 1.0))
W2 = tf.Variable(tf.random_uniform([16,32], 0.0, 1.0))
W3 = tf.Variable(tf.random_uniform([32,1], 0.0, 1.0))

b1  = tf.Variable(tf.zeros([16]))
b2  = tf.Variable(tf.zeros([32]))
b3  =  tf.Variable(tf.zeros([1]))

h1 = leaky_relu(tf.matmul(x_,W1)+b1)
h2 = leaky_relu(tf.matmul(h1,W2)+b2)
y   = leaky_relu(tf.matmul(h2,W3)+b3)
e   =tf.nn.l2_loss(y-t_)


train_step = tf.train.AdamOptimizer().minimize(e)

losses =[]

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    start = time.time()

    for i in range(1000):
        x,t = get_batch(100)
        x = x.astype(np.float32).reshape(100,1)
        t = t.astype(np.float32).reshape(100,1)
        loss= e.eval(feed_dict={x_: x, t_:t})
        losses.append(loss)

        train_step.run(feed_dict={x_: x, t_:t})

        if i%100==0:
            print loss

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

np.save('single_cpu.npy', losses)

