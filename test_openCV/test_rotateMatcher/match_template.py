# -*- coding: utf-8 -*-
# @Time    : 16/02/2017 15:28
# @Author  : Luke
# @Software: PyCharm

import cv2


img = cv2.imread("landscape.jpg",0)

template = cv2.imread("unionpay.jpg",0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)

cv2.imshow("match result",res)
cv2.waitKey(0)