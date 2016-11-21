# -*- coding: utf-8 -*-
# @Time    : 21/11/2016 17:59
# @Author  : Luke
# @Software: PyCharm

import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class UploadHandler(tornado.web.RequestHandler):
    def post(self):
        if self.request.files:
            myfile = self.request.files['myfile'][0]
            fin = open("in.jpg","w")
            print "success to open file"
            fin.write(myfile["body"])
            fin.close()
            self.write("Save file done!")

application=tornado.web.Application([(r'/',MainHandler),(r'/upload', UploadHandler) ]
        )

if __name__=='__main__':
    application.listen(2033)
    tornado.ioloop.IOLoop.instance().start()