# -*- coding: utf-8 -*-
# @Time    : 10/30/16 23:24
# @Author  : Luke
# @Software: PyCharm
import cv2
import sys

img1_path  = '广发标准双金卡.jpg'
img2_path  = '普通光广发DIY信用卡.jpg'

img1 = cv2.imread(img1_path,0) # queryImage
img2 = cv2.imread(img2_path,0) # trainImage

orb = cv2.ORB()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6

index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m_n in matches:
  if len(m_n) != 2:
    continue
  (m,n) = m_n
  if m.distance < 0.69*n.distance:
    good.append(m)

if len(matches)>0:
    print "%d total matches found" % (len(good))
else:
    print "No matches were found - %d" % (len(good))
    sys.exit()

