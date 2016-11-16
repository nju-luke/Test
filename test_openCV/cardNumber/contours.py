# -*- coding: utf-8 -*-
# @Time    : 14/11/2016 17:17
# @Author  : Luke
# @Software: PyCharm

from util import *

img = cv2.imread("test4.jpg")
rows, cols, _ = np.shape(img)

r = 1
img = img[340/r:413/r, :, :]

img2 = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

img1_Y = img2[:, :, 0]
blank  = np.zeros(img1_Y.shape,'uint8')
th, img12 = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img12,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours))[::-1]:
    if cv2.contourArea(contours[i]) < 20:
        contours.pop(i)
cv2.drawContours(blank, contours, -1, (255),1)

plt.imshow(blank,'gray')
plt.show()

# plt.subplot(221 + 0), plt.imshow(tmp1, 'gray')
#
# for i in range(1,3):
#     img1_Y = img2[:, :, i]
#     th, img12 = cv2.threshold(img1_Y, 127, 255, cv2.THRESH_OTSU)
#     canny_img1 = cv2.Canny(img12, 0, threshold)
#     tmp2 = np.array(cv2.dilate(canny_img1, np.array([[1, 1], [1, 1]]), iterations=1), 'float')
#     # tmp2 = np.array(canny_img1, 'float')
#     tmp = (tmp1 - tmp2) > 0  #
#     # tmp =  tmp2>0#
#     plt.subplot(221+i), plt.imshow(tmp, 'gray')
# plt.show()
