# -*- coding: utf-8 -*-
# @Time    : 11/11/2016 19:13
# @Author  : Luke
# @Software: PyCharm
import cv2
from perspective import Perspective
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
perspective = Perspective(img)

plt.imshow(img)
