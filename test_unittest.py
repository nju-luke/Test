# -*- coding: utf-8 -*-
# @Time    : 11/7/16 09:04
# @Author  : Luke
# @Software: PyCharm

import unittest
import numpy as np


class Prime(object):
    def __init__(self,n):
        print "Creat Class Prime Done!"

    def print_prime(self, n):
        for i in xrange(2,n):
            if self.is_prime(i):
                print i

    def is_prime(self,n):
        if n == 2: return True
        return True if reduce(lambda x,y:x*y,[n%i for i in xrange(2,int(np.ceil(np.sqrt(n)))+1)]) \
            else False


class testPrime(unittest.TestCase):
    def test_is_prime(self):
        p = Prime(17)
        print p.is_prime(17)

    def test_print_prime(self):
        n = 100
        p = Prime(n)
        p.print_prime(100)

if __name__ == '__main__':
    unittest.main()