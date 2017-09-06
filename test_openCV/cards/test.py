# -*- coding: utf-8 -*-
# @Time    : 11/1/16 15:43
# @Author  : Luke
# @Software: PyCharm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

path_ori = "/Users/hzqb_luke/Desktop/cardRecognizer_server/cardsForRecognizer_all_0606/02/上海银行"
path_new = "/Users/hzqb_luke/Desktop/test/上海银行"
if not os.path.exists(path_new):
    os.makedirs(path_new)

for card in os.listdir(path_ori):
    if card.startswith("."): continue
    img_path = os.path.join(path_ori, card)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]

    # for i in range(3):
    #     ch = str(int(np.mean(img[:,:,i])))
    #     while len(ch) < 3:
    #         ch = "0"+ch
    #     prefix += ch

    hist = np.histogram(img, 8, (0,256))[0]
    hist = hist/( np.max(hist)+1.)
    hist *= 100
    hist = np.array(hist,"int32")
    vec = [str(mag) if mag > 9 else "0" + str(mag) for mag in hist]

    prefix = "".join(vec) + "_"
    print prefix

    img_path_new = os.path.join(path_new, prefix + card)
    shutil.copy(img_path, img_path_new)
