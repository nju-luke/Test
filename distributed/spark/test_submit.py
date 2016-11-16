# -*- coding: utf-8 -*-
# @Time    : 10/20/16 11:05
# @Author  : Luke
# @Software: PyCharm

from pyspark import SparkContext,SparkConf
import numpy as np
from func_t import  func

sc = SparkContext()
#
#
# conf = SparkConf()
# conf.setAppName("test_spark_tensorflow").setMaster("spark://HZQB-Lukes-MacBook-Pro.local:7077")
#
# # sc = SparkContext(conf=conf)


def map_fun1(i):
    return np.power(i,2)

rdd = sc.parallelize(range(10))
print rdd.map(map_fun1).collect()


'''
单机spark运行尝试结论：
在每一个node安装全部所需要的库，否则需要将所有需要的文件利用 --py-file 的方式全部提交上去。
'''