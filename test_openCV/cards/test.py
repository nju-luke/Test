# -*- coding: utf-8 -*-
# @Time    : 11/1/16 15:43
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('广发DIY信用卡.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

plt.imshow(img)
plt.show()

# corners = cv2.goodFeaturesToTrack(gray,25,0.01,10) # 返回的结果是 [[ 311., 250.]] 两层括号的数组。
# corners = np.int0(corners)
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# plt.imshow(img),plt.show()