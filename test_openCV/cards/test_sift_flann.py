# -*- coding: utf-8 -*-
# @Time    : 10/24/16 14:41
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
import matplotlib.pyplot as plt

from test_openCV.cards.test_crop_and_filter import remove_logo_u
from test_openCV.test_orb import matched_dis, drawMatches

img1 = cv2.imread('1华夏精英白金卡.jpg',0)          # queryImage
img2 = cv2.imread('测试集1华夏精英白金卡2.jpg',0) # trainImage

img1 = remove_logo_u(img1)
img2 = remove_logo_u(img2)

MIN_MATCH_COUNT = 10


# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if 0.5*n.distance<m.distance < 0.8*n.distance:
        good.append(m)
print len(good)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    # img21 = cv2.polylines(img2,[np.int32(dst)],True,255,3)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = drawMatches(img1,kp1,img2,kp2,good)

# plt.imshow(img3, 'gray'),plt.show()