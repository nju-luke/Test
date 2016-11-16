# -*- coding: utf-8 -*-
# @Time    : 11/2/16 09:38
# @Author  : Luke
# @Software: PyCharm

from numpy import *
import cv2
import os
import math
import matplotlib.pyplot as plt


# 为了让LBP具有旋转不变性，将二进制串进行旋转。
# 假设一开始得到的LBP特征为10010000，那么将这个二进制特征，
# 按照顺时针方向旋转，可以转化为00001001的形式，这样得到的LBP值是最小的。
# 无论图像怎么旋转，对点提取的二进制特征的最小值是不变的，
# 用最小值作为提取的LBP特征，这样LBP就是旋转不变的了。
def minBinary(pixel):
    length = len(pixel)
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] == '0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero + pixel
    if len(pixel) == 0:
        return '0'

        # 加载图像

def loadImageSet(add):
    FaceMat = []
    j = 0
    names = []
    for name in os.listdir(add):
        try:
            img = cv2.imread(add + name, 0)
            img = cv2.resize(img,(203,128))
            # cv2.imwrite(str(i)+'.jpg',img)
        except:
            print 'load %s failed' % name
        FaceMat.append(img)
        j += 1
        names.append(name)
    return array(FaceMat),names

def core_LBP(face, R=2, P=8):
    pi = math.pi
    W, H = shape(face)
    tempface = mat(zeros((W, H)))
    for x in xrange(R, W - R):
        for y in xrange(R, H - R):
            repixel = ''
            pixel = int(face[x, y])
            # 　圆形LBP算子
            for p in [2, 1, 0, 7, 6, 5, 4, 3]:
                p = float(p)
                xp = x + R * cos(2 * pi * (p / P))
                yp = y - R * sin(2 * pi * (p / P))
                if face[xp, yp] > pixel:
                    repixel += '1'
                else:
                    repixel += '0'
                    # minBinary保持LBP算子旋转不变
            tempface[x, y] = int(minBinary(repixel), base=2)
    return tempface

# 算法主过程
def LBP(FaceMat):
    LBPoperator = array(zeros(shape(FaceMat)))
    for i in range(shape(FaceMat)[0]):
        # 对每一个图像进行处理
        face = FaceMat[i]
        LBPoperator[i,:,:] = core_LBP(face)
        # cv2.imwrite(str(i)+'hh.jpg',array(tempface,uint8))
    return LBPoperator

    # judgeImg:未知判断图像
    # LBPoperator:实验图像的LBP算子
    # exHistograms:实验图像的直方图分布

# 统计直方图
def calHistogram(Img):
    W, H = shape(Img)
    # 把图片分为7*4份
    Histogram = []
    maskx, masky = W / 6, H / 10
    for i in range(4):
        for j in range(7):
            # 使用掩膜opencv来获得子矩阵直方图
            mask = zeros(shape(Img), uint8)
            mask[i * maskx: (i + 1) * maskx, j * masky:(j + 1) * masky] = 255
            hist = cv2.calcHist([array(Img, uint8)], [0], mask, [256], [0, 256])
            Histogram.extend(hist)
    return squeeze(Histogram)


def dis(hit1,hist2):
    diff = ((hit1 - hist2) ** 2).sum()
    return diff

def runLBP():
    # 加载图像
    FaceMat,names = loadImageSet('cards/')
    nb_samples = len(names)

    LBPoperator = LBP(FaceMat)  # 获得实验图像LBP算子

    # 获得实验图像的直方图分布，这里计算是为了可以多次使用
    exHistograms = []
    for i in range(nb_samples):
        exHistogram = calHistogram(LBPoperator[i])
        exHistograms.append(exHistogram)

    for i in range(nb_samples):
        print names[i]
        for j in range(nb_samples):
            if i == j: continue
            print "\t",names[j],"\t",dis(exHistograms[i],exHistograms[j])

if __name__ == '__main__':
    # 测试这个算法的运行时间
    from timeit import Timer

    t1 = Timer("runLBP()", "from __main__ import runLBP")
    print t1.timeit(1)

#plt.subplot(411),plt.imshow(LBPoperator[0],'gray')
# plt.subplot(412),plt.imshow(LBPoperator[6],'gray')
# plt.subplot(413),plt.imshow(LBPoperator[7],'gray')
# plt.subplot(414),plt.imshow(LBPoperator[8],'gray')
# plt.show()

'''
LBP 可以去除光照的影响，但是对于纹理复杂的图片应该怎么体现纹理的不同呢？
'''