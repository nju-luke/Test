# -*- coding: utf-8 -*-
# @Time    : 22/11/2016 09:23
# @Author  : Luke
# @Software: PyCharm

from pyspark import SparkContext
from pyspark import SparkConf
import cv2

from splearn.rdd import ArrayRDD

conf = SparkConf()
conf.setAppName("test").setMaster("local[*]")

sc = SparkContext(conf=conf)

def load_img(file_name):
    img = cv2.imread(file_name)
    return img
A = sc.textFile(".")
file_list = sc.parallelize(['2.jpg','3.jpg'])

file_list_array = ArrayRDD(file_list)
print file_list_array[0].collect()

results = file_list.map(load_img).collect()
X = sc.parallelize(results)
X = ArrayRDD(X)
print X
print X[0]
print X[0].collect()[0][0]