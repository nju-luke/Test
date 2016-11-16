# -*- coding: utf-8 -*-
# @Time    : 10/12/16 17:32
# @Author  : Luke
# @Software: PyCharm

import os
import sys
from pyspark import SparkContext
import tensorflow as tf
from pyspark import SparkConf
import numpy as np

conf = SparkConf()
conf.setAppName("test_spark_tensorflow").setMaster("spark://172.16.122.128:7077")#.setMaster("spark://120.132.3.69:7077")

sc = SparkContext(conf=conf)


def map_fun(i):
  with tf.Graph().as_default() as g:
    hello = tf.constant('Hello, TensorFlow! %s'%i, name="hello_constant")
    with tf.Session() as sess:
      return sess.run(hello)
'''
spark集群也需要安装所需要的库
'''

def map_fun1(i):
    # return np.power(i,2)
    return i

rdd = sc.parallelize(range(10))
print rdd.map(map_fun1).collect()
