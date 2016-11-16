# -*- coding: utf-8 -*-
# @Time    : 14/11/2016 09:07
# @Author  : Luke
# @Software: PyCharm

from util import *
from itertools import groupby

img_rgb = cv2.imread('test.jpg')
r = 1
img_rgb = img_rgb[340/r:433/r, :, :]


def cluster(loc):
    """
    聚类去除干扰点
    """
    if isinstance(loc, tuple):
        loc = [pt for pt in zip(*loc)]

    def remove_nearby(cols):
        cols = cols[:]
        for i in range(1, len(cols))[::-1]:
            distance = lambda pt: np.sqrt((pt[0] - cols[i][0]) ** 2
                                          + (pt[1] - cols[i][1]) ** 2)
            if min([distance(ci) for ci in cols[:i]]) < 1000/50:             #todo 距离阈值
                cols.pop(i)
        return cols
    return remove_nearby(loc)


def match(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #todo  增强图片，利用幂次,(局部直方图均衡）---增加对比度
    path = "numbers"
    numbers = os.listdir(path)
    pts = []    #(pt[0],pt[1],label)
    # labels = []

    for number in numbers:
        template = cv2.imread(os.path.join(path,number),0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.4
        #umpy.where(condition[, x, y])
        #Return elements, either from x or y, depending on condition. #If only condition is given, return condition.nonzero().
        loc = np.where( res >= threshold)
        loc = cluster(loc)
        pts.extend([(l[0],l[1],number[0],res[l[0],l[1]]) for l in loc])

    pts = cluster(pts)


    for pt in pts:
        pt = pt[:2][::-1]
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    pts = sorted(pts, key=lambda x: x[0])
    group = groupby(pts, key=lambda x: x[0])
    for key, vals in group:
        print key,list(vals)
        # print key, len(list(vals))

    #todo 判断最多点

    plt.imshow(img_rgb)
    plt.show()

match(img_rgb)