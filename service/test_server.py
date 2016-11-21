# -*- coding: utf-8 -*-
# @Time    : 21/11/2016 16:54
# @Author  : Luke
# @Software: PyCharm

import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

application = tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()