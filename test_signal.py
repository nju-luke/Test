# -*- coding: utf-8 -*-
# @Time    : 06/12/2016 16:06
# @Author  : Luke
# @Software: PyCharm
# coding=utf-8
import signal
import time


class TimeOutException(Exception):
    pass


def setTimeout(num, callback):
    def wrape(func):
        def handle(signum, frame):
            raise TimeOutException("运行超时！")

        def toDo(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)  # 开启闹钟信号
                rs = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟信号
                return rs
            except TimeOutException, e:
                raise TimeOutException

        return toDo

    return wrape


if __name__ == '__main__':
    def doSome():
        print "lij"


    @setTimeout(2, doSome)
    def getName():
        time.sleep(3)
        return ["A", "B", "C"]


    s = getName()
    print s