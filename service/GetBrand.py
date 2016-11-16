# -*- coding: utf-8 -*-

import time

class GetBrand():
    def __init__(self):
        self.a = range(10)
        self.b = 'name'
        self.brand = self.loadFile("/home/luke/Projects/PycharmProjects/StatiscOfTheShops/Brands (复件).txt")

    def get(self,intN):
        return sum(self.a)+intN

    def inBrands(self,name):
        if name in self.brand:
            return name
        else:
            return "NoBrands"

    def loadFile(self,path):
        brand = {}
        file = open(path, 'r')
        for name in file.readlines():
            brand[name.strip()] = 1
        file.close()
        return brand

if __name__ == '__main__':

    print time.asctime()
    test = GetBrand()
    print time.asctime()
    print test.get(10)
    print time.asctime()
    print test.inBrands("重庆小面")
    print time.asctime()