# -*- coding: utf-8 -*-
# @Time    : 10/13/16 09:51
# @Author  : Luke
# @Software: PyCharm
import os
import shutil


path = '/Users/hzqb_luke/Downloads/pukeImage'
names = [name for name in os.listdir(path) if not name.startswith(".")]

C_oir = "S"
C_new = "s"

for name in names:
    if name.startswith(C_oir):
        path_1 = os.path.join(path,name)
        path_2 = os.path.join(path,name.replace(C_oir,C_new))
        shutil.move(path_1,path_2)
