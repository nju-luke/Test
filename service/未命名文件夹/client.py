# -*- coding: utf-8 -*-

import socket

print "Creating socket...",
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print "done."

print "Looking up port number...",
port = socket.getservbyname('http', 'tcp')
print "done."

print "Connecting to remote host on port %d..." % port,
s.connect(("0.0.0.0", 1060))
print "done."

#获取本身的IP和端口号
print "Connected from", s.getsockname()
#获取远程的IP和端口号
print "Connected to", s.getpeername()