# -*- coding: utf-8 -*-

import logging
import os

'''
凯哥用的是fileconfig
'''


# logging.basicConfig(level=logging.DEBUG,
#                 format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='/Users/hzqb_luke/Desktop/Test/myapp.log',
#                 filemode='a')
#
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-8s: %(levelname)-8s ：%(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)
#
# logging.debug('This is debug message')
# logging.info('This is info message')
# logging.warning('This is warning message')

def logger(log_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_dir,
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-8s: %(levelname)-8s ：%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging
# logger = logging

if __name__ == '__main__':
    logger = logger('/Users/hzqb_luke/Desktop/Test/myapp.log')
    logger.debug('This is debug message')
    logger.info('This is info message')
    logger.warning('This is warning message')



