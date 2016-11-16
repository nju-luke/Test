# -*- coding: utf-8 -*-

import pymongo
import json


client = pymongo.MongoClient("localhost",27017)
db = client.test
collection = db.mycollection

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

dicString = json.dumps(dicList)