# -*- coding: utf-8 -*-
from pprint import pprint

import pymongo
import json


client = pymongo.MongoClient("192.168.3.231",27017)
db = client.coupon
collection = db.CategoryDictionaryNew

items = collection.find()
dicList = [it for it in items]

shopName = []
for item in dicList:
    try:
        shopName.append(item['shopName'])
    except KeyError:
        pass

with open("shopName.txt",'w') as file_new:
    for item in shopName:
        file_new.writelines(item.encode('utf-8')+'\n')
        

for dic in dicList:
    dic.pop('_id')
    for key in dic:
        try:
            dic[key].encode('utf-8')
        except:
            pass


key = u"topLevel"
key1 = u"category"
key2 = u"realCategory"

category = {}
category1 = {}
for dic in dicList:
    try:
        category[dic[key]].append(dic[key1])
        category1[dic[key]].append(dic[key2])
    except KeyError:
        category[dic[key]]=[dic[key1]]
        category1[dic[key]]=[dic[key2]]

pprint(category)




