# -*- 'utf-8': utf-8 -*-
# @Time    : 2017/9/7 09:17
# @Author  : Luke
# @Software: PyCharm
from bson import ObjectId
from pandas import DataFrame

from utils import get_collection

collection_relation = get_collection("UpShopToMainShop")
collection_UP = get_collection("unionpayShop")
collection_DP = get_collection("MainShop5")


shops = {}

UP_keys = {"MCHNT_NM", "MCHNT_ADDR", "MCHNT_PHONE"}
DP_keys = {'shopName', 'address', 'telephones'}

for pair in collection_relation.find():
    shop_UP = collection_UP.find_one({'MCHNT_CD': pair[u'_id']},
                                     {"MCHNT_NM": 1, "MCHNT_ADDR": 1, "MCHNT_PHONE": 1})
    shop_DP = collection_DP.find_one({"_id": ObjectId("{}".format(pair[u'desId']))},
                                     {'shopName': 1, 'address': 1, 'telephones': 1})

    if shop_DP == None or shop_DP == None:
        print "No record:",pair
        continue

    shop_UP.pop(u'_id')
    shop_DP.pop(u'_id')

    if not set(shop_DP.keys()) == DP_keys:
        print "Not contain all keys!",shop_DP
        continue
    if not set(shop_UP.keys()) == UP_keys:
        print "Not contain all keys!",shop_UP
        continue

    for shop in [shop_UP,shop_DP]:
        for key in shop.keys():
            if isinstance(shop[key],list):
                try:
                    shop[key] = shop[key][0]
                except IndexError:
                    shop[key] = ""
            try:
                shops[key].append(shop[key])
            except KeyError:
                shops[key] = [shop[key]]


df = DataFrame(shops,columns=[u'shopName', u'MCHNT_NM',  u'address', u'MCHNT_ADDR', u'telephones',u'MCHNT_PHONE'])

df.to_excel("shops.xlsx",sheet_name="Sheet1")

