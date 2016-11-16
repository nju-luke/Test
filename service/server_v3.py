# coding:utf-8
import textwrap

import tornado.ioloop
import tornado.httpserver
import tornado.web
import GetBrand

RequestHandler = tornado.web.RequestHandler

# test1 = GetBrand.GetBrand()

class MainHandler(RequestHandler):
    # def getBrand(self):
    #     return GetBrand.GetBrand().inBrands

    getBrand = GetBrand.GetBrand().inBrands

    def get(self,name):              ##post？？？？？？
        name = self.getBrand(name)
        self.write(name)

class WrapHandler(tornado.web.RequestHandler):
    def post(self):
        text = self.get_argument("text")
        width = self.get_argument("width", 40)
        self.write(textwrap.fill(text, width))

if __name__ == "__main__":
    app = tornado.web.Application([tornado.web.url(r"/getBrands/(\w+)", MainHandler),
                                   tornado.web.url(r"/wrap", WrapHandler)
                                    ])
    http_sever = tornado.httpserver.HTTPServer(app)
    http_sever.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
    # handle_request()
