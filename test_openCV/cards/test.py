# -*- coding: utf-8 -*-
# @Time    : 11/1/16 15:43
# @Author  : Luke
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Take each img

img = cv2.imread("测试集3交通银行金卡3.jpg")
# img = remove_logo_u(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)[:, :, 0]

hist1 = cv2.calcHist([img[:, :]], [0], None, [256], [0.0, 255.0])
hist1 = [i for i in np.squeeze(hist1)]

img = cv2.imread("VISA白金卡_.jpg")
# img = remove_logo_u(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)[:, :, 0]

hist2 = cv2.calcHist([img[:, :]], [0], None, [256], [0.0, 255.0])
hist2 = [i for i in np.squeeze(hist2)]

width = 50
hist1.extend(hist1[:width])
plt.subplot(211), plt.plot(hist1)
hist2.extend(hist2[:width])
plt.subplot(212), plt.plot(hist2)

def var(hist):
    h1 = np.array(hist, 'float')
    p1 = h1 / sum(h1)
    id1 = np.arange(len(h1))
    mean1 = sum(p1 * id1)
    return sum((id1 - mean1) ** 2 * p1)

kernal = np.ones(width)
print "Local weight center:"
weights1 = np.convolve(hist1, kernal, 'same')
center_l1 = np.argmax(weights1)
print center_l1
weights2 = np.convolve(hist2, kernal, 'same')
center_l2 = np.argmax(weights2)
print center_l2

if abs(center_l1 - 255) > width:
    hist1 = hist1[:256]
    start1 = 0
else:
    hist1 = hist1[width:]
    start1 = width
if abs(center_l2 - 255) > width:
    hist2 = hist2[:256]
    start2 = 0
else:
    hist2 = hist2[width:]
    start2 = width

def similarity(param1, param2, bandwidth_1 = 20, bandwidth_2 = 800):
    sim = 0.
    for i in range(3):
        sim += np.exp(-((param1[i] - param2[i]) / bandwidth_1)**2)
    sim += np.exp(-((param1[3] - param2[3]) / bandwidth_2)**2)
    return sim/4

print "Global weight center:"
id = np.arange(len(hist1))
G1 =  (float(sum(id * hist1)) / sum(hist1)) + start1
G2 =  (float(sum(id * hist2)) / sum(hist2)) + start2

print "Most hue position:"
M1 =  np.argmax(hist1) + start1
M2 =  np.argmax(hist2) + start2

print "Var:"
var1 =  var(hist1)
var2 =  var(hist2)

plt.show()

param1 = [G1,center_l1,M1,var1]
param2 = [G2,center_l2,M2,var2]

print similarity(param1,param2)



'''
max, G_weight, L_weight, var
当颜色偏差大一点的时候这样一些判断都存在一定程度的误差
（方案1：当var大时，认为图片为杂色，全部计算）
'''
