# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

tf.app.flags.DEFINE_string("ps_hosts", "172.16.122.1:13240",
                    "List of hostname:port for ps jobs."
                    "This string should be the same on every host!!")
tf.app.flags.DEFINE_string("worker_hosts", "172.16.122.1:13241,172.16.122.130:13242",
                    "List of hostname:port for worker jobs."
                    "This string should be the same on every host!!")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 1, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start1 a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:0" ,
                cluster=cluster)):

            # Build model...
            network = input_data(shape=[None, 28, 28, 1], name='input')
            network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)
            network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
            network = max_pool_2d(network, 2)
            network = local_response_normalization(network)

        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:1",
                cluster=cluster)):

            network = fully_connected(network, 128, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.8)
            network = fully_connected(network, 10, activation='softmax')
            network = regression(network, optimizer='adam', learning_rate=0.01,
                                 loss='categorical_crossentropy', name='target')


        X, Y, testX, testY = mnist.load_data(one_hot=True)
        X = X.reshape([-1, 28, 28, 1])
        testX = testX.reshape([-1, 28, 28, 1])

        sv = tf.train.Supervisor()
        sess = sv.managed_session(server.target)
        # Training
        model = tflearn.DNN(network, tensorboard_verbose=0)
        model.fit({'input': X}, {'target': Y}, n_epoch=20,
                   validation_set=({'input': testX}, {'target': testY}),
                   snapshot_step=100, show_metric=True, run_id='convnet_mnist')

if __name__ == "__main__":
    tf.app.run()