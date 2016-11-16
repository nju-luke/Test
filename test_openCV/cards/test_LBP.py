# -*- coding: utf-8 -*-
# @Time    : 11/2/16 09:38
# @Author  : Luke
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt

ids = [0,1,2,5,8,7,6,3]

def core_LBP(img):
    size = 3
    rows,cols = img.shape
    lbp_mat = []
    for i in range(1, rows - 1):
        row_mat = []
        for j in range(1, cols - 1):
            tmp_img = img[i - 1:i + size - 1, j - 1:j + size - 1]
            lbp = (tmp_img > tmp_img[(size - 1) / 2, (size - 1) / 2]) * 1
            lbp = np.squeeze(np.reshape(lbp, (1, -1)))

            lbp = int("".join([str(lbp[id]) for id in ids]), base=2)

            row_mat.append(lbp)
        lbp_mat.append(np.array(row_mat))
    return np.array(lbp_mat)


def LBP(img, rows_batch = 3, cols_batch = 3):
    rows,cols = img.shape
    nb_row = int(np.floor(float(rows)/rows_batch))
    nb_col = int(np.floor(float(cols)/cols_batch))
    lbp_img = []
    for i in range(nb_row):
        lbp_row = []
        for j in range(nb_col):
            tmp_img = img[i*rows_batch:(i+1)*rows_batch,j*cols_batch:(j+1)*cols_batch]
            lbp_tmp = core_LBP(tmp_img)
            lbp_row.append(lbp_tmp)
        lbp_img.append(np.concatenate(lbp_row,axis=1))
    lbp_img = np.concatenate(lbp_img,axis = 0)
    # plt.subplot(121),plt.imshow(img,'gray')
    # plt.subplot(122),plt.imshow(lbp_img,'gray')
    # plt.show()
    return lbp_img



img1 = cv2.imread("lena.jpg",0)
img1_lbp = LBP(img1)
img2 = cv2.imread("测试集1交通沃尔玛联名普卡2.jpg",0)
img2_lbp = LBP(img2)
plt.subplot(221),plt.imshow(img1,'gray')
plt.subplot(222),plt.imshow(img1_lbp,'gray')
# plt.subplot(223),plt.imshow(img2,'gray')
# plt.subplot(224),plt.imshow(img2_lbp,'gray')
plt.show()