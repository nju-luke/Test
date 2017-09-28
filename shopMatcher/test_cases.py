# -*- coding: utf-8 -*-
# @Time    : 2017/9/7 09:22
# @Author  : Luke
# @Software: PyCharm
from unittest import TestCase
from utils import get_collection



class TestGet_collection(TestCase):
    def test_get_collection(self):
        collection = get_collection("UpShopToMainShop")
        result = collection.find_one()
        if not result == None:
            print result
        else:
            self.fail()
