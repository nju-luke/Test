# -*- coding: utf-8 -*-
# @Time    : 06/12/2016 17:26
# @Author  : Luke
# @Software: PyCharm

import Tkinter as tk
from Tkinter import *

import tkMessageBox

# class App(Frame):
#     def __init__(self, master=None):
#         if not hasattr(self,'i'):
#             self.i = 0
#         Frame.__init__(self, master)
#         self.pack()
#
#         self.entrythingy = Entry()
#         self.entrythingy.pack()
#
#         self.img = PhotoImage(file="4.gif")
#         self.label = Label(self,image=self.img)
#         self.label.pack()
#
#         # here is the application variable
#         self.contents = StringVar()
#         # set it to some value
#         self.contents.set("this is a variable %s")
#         # tell the entry widget to watch this variable
#         self.entrythingy["textvariable"] = self.contents
#
#         # and here we get a callback when the user hits return.
#         # we will have the program print out the value of the
#         # application variable when the user hits return
#         self.entrythingy.bind('<Key-Return>',
#                               self.print_contents)
#
#     def print_contents(self, event):
#         print "hi. contents of entry is now ---->", \
#               self.contents.get()
#
# app = App()

#
self = tk.Tk()
img = tk.PhotoImage(file='/Users/hzqb_luke/Desktop/Test/4.gif')
self.label = tk.Label(self,image=img)
self.label.pack()

self.nameInput = tk.Entry(self)
self.nameInput.pack()
#
def hello():
    name = self.nameInput.get()
    print "number is %s" %name
    img = tk.PhotoImage(file='/Users/hzqb_luke/Desktop/Test/3.gif')
    self.label = tk.Label(self,image=img)
    self.label.pack()


self.alertButton = tk.Button(self, text='Hello',command=hello)
self.alertButton.pack()


self.mainloop()
