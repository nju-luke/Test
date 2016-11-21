# -*- coding: utf-8 -*-
# @Time    : 21/11/2016 21:54
# @Author  : Luke
# @Software: PyCharm
import time
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf()
conf.setAppName("test_remote").setMaster("spark://172.16.122.128:7077")

sc = SparkContext(conf=conf)

class is_prime():
    def __init__(self,n):
        nums = sc.parallelize(xrange(1, n))
        nums.filter(self.isprime).collect()

    def isprime(self,n):
        """
        check if integer n is a prime
        """
        # make sure n is a positive integer
        n = abs(int(n))
        # 0 and 1 are not primes
        if n < 2:
            return False
        # 2 is the only even prime number
        if n == 2:
            return True
        # all other even numbers are not primes
        if not n & 1:
            return False
        # range starts with 3 and only needs to go up the square root of n
        # for all odd numbers
        for x in range(3, int(n ** 0.5) + 1, 2):
            if n % x == 0:
                return False
        return True


# Create an RDD of numbers from 0 to 1,000,000


# Compute the number of primes in the RDD
print time.asctime()
is_prime = is_prime(100)
print time.asctime()

