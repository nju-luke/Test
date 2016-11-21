# -*- coding: utf-8 -*-
# @Time    : 14/11/2016 09:08
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def pre_processing(img,n=3):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)[:,:,0]
    img_gray = np.array(img_gray, 'float') / 255
    return np.array(img_gray**n*255,'uint8')

def load_templates(path,shape=(39,62)):
    names = [name for name in os.listdir(path) if not name.startswith(".")]
    mat_templates = []
    for name in names:
        template = cv2.imread(os.path.join(path,name))
        template = cv2.resize(template,shape)
        template = pre_processing(template)
        mat_templates.append(template)

    return names,mat_templates


if __name__ == '__main__':
    path = "numbers"
    _,templates = load_templates(path)
    for template in templates:
        t = np.array(template,'float')/255
        cv2.imshow('t',t**2)
        cv2.waitKey(0)
