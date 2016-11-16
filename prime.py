# -*- coding: utf-8 -*-
# @Time    : 10/16/16 10:53
# @Author  : Luke
# @Software: PyCharm

import numpy as np

class Prime():
    def __init__(self):
        self.W = 0
        self.train()

    def is_prime(self,n):
        k = int(np.sqrt(n))
        for i in range(2,k+1):
            if n%i == 0:
                return False
        return True

    def get(self,n):
        return np.dot(self.W, n)

    def train(self):
        self.W = np.random.rand(2,2)
        print self.W
w = np.array([[1,0],[0,1]])
print w
Prime.W = w
model = Prime()
print "Finish"