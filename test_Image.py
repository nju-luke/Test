# -*- coding: utf-8 -*-
# @Time    : 16-9-26 下午3:52
# @Author  : Luke
# @Software: PyCharm

import os
import numpy as np
import cv2
from PIL import Image

from test_logging import logger

path = "兴业QQ VIP VISA金卡（可爱版）.jpg"
im = Image.open(path).convert("L")          #convert to gray image
im.save(path+".bmp")

img = cv2.imread(path+".bmp")               #Image读取图像以后的格式为函数，cv2读取后为矩阵
cv2.imshow("BG",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

path1 = "兴业QQ VIP VISA卡（可爱版）.jpg.bmp"
imag1 = cv2.imread(path1)

print "done"

