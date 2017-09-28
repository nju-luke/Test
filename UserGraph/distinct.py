# -*- coding: utf-8 -*-
# @Time    : 2017/9/21
# @Author  : Luke

import os
# from pyspark import SparkContext
# from pyspark import SparkConf
#
# conf = SparkConf()
# conf.setAppName("Distinct").setMaster("local[*]")
# sc = SparkContext(conf=conf)

def is_test_data(line):
    return "000000" not in line

parent_path = "/Users/hzqb_luke/Desktop/Test/UserGraph/20170923"
for secP in os.listdir(parent_path):
    # if secP != "yangbo_5941_2900": continue
    if secP.endswith("new"):continue
    print secP
    if secP.startswith("."): continue
    ori_path = os.path.join(parent_path,secP,"all_records")
    # new_path = ori_path+"_new"
    # if os.path.exists(new_path):
    #     continue
    new_path = ori_path+"_new"
    secP_file = open(ori_path)
    secP_file_new = open(new_path,"w")
    lines = set(secP_file.readlines())

    for line in lines:
        secP_file_new.writelines(line)
    secP_file_new.close()


    # file_path = os.path.join(ori_path,"all_records")
    # rdd = sc.textFile("file://"+file_path).distinct()
    # rdd_distincted = rdd.filter(is_test_data)
    # rdd_distincted.saveAsTextFile("file://"+new_path)

