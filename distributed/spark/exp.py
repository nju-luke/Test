# -*- coding: utf-8 -*-
# @Time    : 11/4/16 15:40
# @Author  : Luke
# @Software: PyCharm




from pyspark import SparkContext
from pyspark import SparkConf


conf = SparkConf()
conf.setAppName("test_remote").setMaster("spark://172.16.122.139:7077")
# conf.setAppName("test_remote").setMaster("spark://127.0.0.1:7077")#
#
sc = SparkContext(conf=conf)

import tensorflow as tf

abc = tf.constant(3)

sess = tf.Session()
res = sess.run(abc)
sess.close()


print res