# -*- coding: utf-8 -*-
# @Time    : 12/11/2016 17:42
# @Author  : Luke
# @Software: PyCharm

from util import *

img = cv2.imread('test4.jpg')
img = img[340:413, :, :]

threshold = 1

for i in range(3):
    plt.subplot(221+i)
    img1 = img[:,:,i]
    # th, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_OTSU)
    # canny_img1 = cv2.Canny(img1, 0, threshold)
    # if i == 0:
    #     tmp1 = canny_img1
    plt.imshow(img1,'gray')
# plt.show()

img2 = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

img1_Y = img2[:, :, 0]
th, img12 = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)
canny_img1 = cv2.Canny(img12, 0, threshold)
tmp1 = np.array(canny_img1, 'float')
plt.subplot(221 + 0), plt.imshow(tmp1, 'gray')

for i in range(1,3):
    img1_Y = img2[:, :, i]
    th, img12 = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)
    canny_img1 = cv2.Canny(img12, 0, threshold)
    tmp2 = np.array(cv2.dilate(canny_img1, np.array([[1, 1], [1, 1]]), iterations=1), 'float')
    # tmp2 = np.array(canny_img1, 'float')
    tmp = (tmp1 - tmp2) > 0  #
    # tmp =  tmp2>0#
    plt.subplot(221+i), plt.imshow(tmp, 'gray')
plt.show()