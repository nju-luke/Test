# -*- coding: utf-8 -*-
# @Time    : 14/11/2016 09:07
# @Author  : Luke
# @Software: PyCharm

from util import *

img = cv2.imread("test.jpg")
img = img[340:413, :, :]
rows, cols, _ = np.shape(img)

threshold = 0.2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
# th,gray = cv2.threshold(gray,np.mean(gray),255,cv2.THRESH_BINARY)
canny_g = cv2.Canny(gray, 0, threshold)
plt.subplot(221), plt.imshow(canny_g, 'gray')

##YUV 边缘检测
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img1_Y = img1[:, :, 0]
th, img1_Y = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)
# th,img1_Y = cv2.threshold(img1_Y,np.mean(gray),255,cv2.THRESH_BINARY)
canny_img1 = cv2.Canny(img1_Y, 0, threshold)
tmp1 = np.array(canny_img1, 'float')
plt.subplot(220 + 2), plt.imshow(canny_img1, 'gray')
for i in range(1, 2):
    img1_Y = img1[:, :, i]
    th, img1_Y = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)
    canny_img1 = cv2.Canny(img1_Y, 0, threshold)
    tmp2 = np.array(cv2.dilate(canny_img1, np.array([[1, 1], [1, 1]]), iterations=1), 'float')
    # tmp2 = np.array(canny_img1, 'float')
    tmp = (tmp1 - tmp2) > 0  #
    # tmp =  tmp2>0#
    plt.subplot(220 + i + 2), plt.imshow(tmp, 'gray')

plt.show()
pixels = np.sum(tmp, axis=0) > 0


def cat(line, val):
    '''
    通过长度对比将噪声对应的少数点位置去掉
    :param val:
    :return:
    '''
    line = [li for li in line]
    start = 0
    val1 = (val == 0) * 1
    for li in xrange(1, len(line)):
        if line[li] == val:
            if line[start] != val:
                start = li
            else:
                continue
        else:
            if line[start] == val:
                length = li - start
                # print li, length
                if length < 10:
                    line[start:li] = [val1 for i in range(start, li)]
                start = li
    return line


pixels1 = pixels[:]
pixels1 = cat(pixels1, 0)
pixels1 = cat(pixels1, 1)
plt.subplot(211), plt.imshow(img)
plt.subplot(212), plt.plot(pixels1), plt.ylim([0, 2])
plt.show()

'''
通过以上处理以后，字符之间的间隔0的长度可以用来作为一个字符位置判断的依据
（利用间隔位置，以及单字符长度，求出较长连接字符的位置）
'''
