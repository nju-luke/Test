# -*- coding: utf-8 -*-
# @Time    : 2017/9/7 09:16
# @Author  : Luke
# @Software: PyCharm

import pymongo

client = pymongo.MongoClient("192.168.3.231", 27017)


def get_collection(collection_name,db_name="coupon"):
    db = client[db_name]
    collection = db[collection_name]
    return collection
