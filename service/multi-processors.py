# -*- coding: utf-8 -*-
# @Time    : 13/12/2016 17:32
# @Author  : Luke
# @Software: PyCharm
import os

import tornado.web
import tornado.httpserver
import tornado.options
import tornado.ioloop


import time



class Fib:
    def fib(self,n):
        a,b = 0,1
        while n > 0:
            a,b = b,a+b
            n -= 1
        return a


class SleepHandler(tornado.web.RequestHandler):
    fib = Fib()
    def get(self):
        self.write(str(os.getpid()))
        self.write("<br>")
        self.write(str(self.fib.fib(os.getpid())))
        time.sleep(5)
        self.write("<br>")
        self.write(time.asctime())
        self.write("this is SleepHandler...")

class DirectHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("this is DirectHandler...")

tornado.options.parse_command_line()
app = tornado.web.Application(
    handlers = [
        (r"/d",DirectHandler),
        (r"/s",SleepHandler),
    ]
)
if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(8888)
    http_server.start(0)
    # [I 150610 10:42:05 process:115] Starting 4 processes
    tornado.ioloop.IOLoop.instance().start()