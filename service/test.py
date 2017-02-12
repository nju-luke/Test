# -*- coding: utf-8 -*-
# @Time    : 19/12/2016 18:15
# @Author  : Luke
# @Software: PyCharm

import urllib2
from multiprocessing.pool import Pool

import time
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers

# 在 urllib2 上注册 http 流处理句柄
register_openers()

# 开始对文件 "DSC0001.jpg" 的 multiart/form-data 编码
# "image1" 是参数的名字，一般通过 HTML 中的 <input> 标签的 name 参数设置

# headers 包含必须的 Content-Type 和 Content-Length
# datagen 是一个生成器对象，返回编码过后的参数
datagen, headers = multipart_encode({"img": open("in.jpg", "rb")})

t1 =  time.asctime()
# def urlre(j):
for i in range(300):

    request = urllib2.Request("http://cr.u51pay.com/upload", datagen, headers)
    # 创建请求对象
    # 实际执行请求并取得返回
    result = urllib2.urlopen(request).read()
    # print result
    print i
# pool = Pool(4)
times = range(12)
# pool.map(urlre,times)
# pool.close()
# pool.join()
print t1
print time.asctime()
