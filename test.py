# -*- coding: utf-8 -*-
# @Time    : 10/13/16 09:51
# @Author  : Luke
# @Software: PyCharm

import urllib2
from multiprocessing.dummy import Pool as ThreadPool

urls = [
    'http://www.python.org',
    'http://www.python.org/about/',
    'http://www.onlamp.com/pub/a/python/2003/04/17/metaclasses.html',
    'http://www.python.org/doc/',
    'http://www.python.org/download/',
    'http://www.python.org/getit/',
    'http://www.python.org/community/',
    'https://wiki.python.org/moin/',
    'http://planet.python.org/',
    'https://wiki.python.org/moin/LocalUserGroups',
    'http://www.python.org/psf/',
    'http://docs.python.org/devguide/',
    'http://www.python.org/community/awards/'
    # etc..
    ]

# Make the Pool of workers
class Main():
    def double(self,n):
        return 2*n

    def main(self):
        def double(n):
            return 2*n

        pool = ThreadPool(4)
        # Open the urls in their own threads
        # and return the results
        results = pool.map(double, urls)
        #close the pool and wait for the work to finish
        pool.close()
        pool.join()

if __name__ == '__main__':
    m = Main()
    m.main()