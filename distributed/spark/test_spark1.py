# -*- coding: utf-8 -*-
# @Time    : 11/4/16 15:40
# @Author  : Luke
# @Software: PyCharm


from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf()
# conf.setAppName("test_remote").setMaster("local[*]")
conf.setAppName("test_remote").setMaster("spark://localhost:7077")#
#
sc = SparkContext(conf=conf)






def isprime(n):
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
nums = sc.parallelize(xrange(1,200))

# # Compute the number of primes in the RDD
print nums.filter(isprime).collect()
#
# # nums.map(lambda x: x/2).collect()
