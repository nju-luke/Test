# -*- coding: utf-8 -*-
# @Time    : 06/12/2016 17:26
# @Author  : Luke
# @Software: PyCharm

from easygui import buttonbox

image = "3.gif"
msg = "Do you like this picture?"
# choices = ["Yes","No","No opinion"]
choices = [str(i) for i in range(11)]
reply = buttonbox(msg, image=image, choices=choices)
print reply
