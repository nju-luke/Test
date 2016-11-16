#from matplotlib import pyplot as plt
import numpy as np
import math
import time

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf


num_batches=600


# model utitlity functions

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# model definition

x = tf.placeholder("float", shape=[None, 784])
t = tf.placeholder("float", shape=[None, 10])


W_conv1 = weight_variable([5, 5, 1, 8])
b_conv1 = bias_variable([8])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 8, 16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 *16, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# loss, optimizer & output definitions
cross_entropy = -tf.reduce_sum(t*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train loop

sess = tf.Session()
sess.run(tf.initialize_all_variables())

elapsed_time10=0.0

with sess.as_default():
    for i in range(1000):
      start = time.time()
      batch = mnist.train.next_batch(num_batches)

      train_step.run(feed_dict={x: batch[0], t: batch[1], keep_prob: 0.5})
      elapsed_time = time.time() - start
      elapsed_time10+=elapsed_time
      if i%10 == 0:
        train_accuracy = sess.run(accuracy,feed_dict={
            x:batch[0], t: batch[1], keep_prob: 1.0})
        loss = sess.run(cross_entropy,feed_dict={
            x:batch[0], t: batch[1], keep_prob: 1.0})
        print "step %05d, training accuracy %2.3f, loss %5.2f,"%(i, train_accuracy, loss),
        print "time %2.3f [sec/step]" % elapsed_time
        elapsed_time10=0.0

    print "test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, t: mnist.test.labels, keep_prob: 1.0})
