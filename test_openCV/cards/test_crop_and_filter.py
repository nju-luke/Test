# -*- coding: utf-8 -*-
# @Time    : 10/27/16 12:42
# @Author  : Luke
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mtimage


def header(img):
    rows, cols, _ = np.shape(img)
    new_rows = int(float(rows) / 5)
    img1 = img[:new_rows]
    plt.figure()
    plt.imshow(img1)

def remove_logo_u(img, is_gray = False):
    row_ratio = 0.656
    col_ratio = 0.694
    if not is_gray:
        rows, cols, _ = np.shape(img)
    else:
        rows, cols = np.shape(img)
    pad = np.asarray(np.ones(np.shape(img)),'uint8')
    pad[int(rows * row_ratio):, int(cols * col_ratio):] = 0
    img *= pad
    # plt.figure()
    # plt.imshow(img2)
    return img

if __name__ == '__main__':

    img1 = cv2.imread("北京银行北京公务卡（金卡）.jpg")
    img = mtimage.imread("北京银行北京公务卡（金卡）.jpg")

    plt.imshow(img)

    header(img)
    remove_logo_u(img)


    plt.waitforbuttonpress()

    print sum(img)