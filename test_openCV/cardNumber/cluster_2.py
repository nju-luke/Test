# -*- coding: utf-8 -*-
# @Time    : 15/11/2016 16:40
# @Author  : Luke
# @Software: PyCharm

from util import *
from itertools import groupby

distance = lambda pt1,pt2:np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def means(point_list,k = 2):
    x_ = np.mean([pt[0] for pt in point_list])
    y_ = np.mean([pt[1] for pt in point_list])
    if k == 1:
        return np.ceil(x_)
    return np.ceil(x_),np.ceil(y_)

def cluster_rows(pts):
    '''
    保留匹配位置最多的行对应匹配点，其他的删除
    :param pts:
    :return:
    '''
    pts = sorted(pts,key=lambda x:x[0])
    group = groupby(pts,lambda x:x[0])
    dic = {}
    for key, val in group:
        dic[key] = [va for va in val]
    length = dict([(key, len(dic[key])) for key in dic.keys()])
    length = sorted(length.iteritems(), key=lambda x: x[1], reverse=True)
    key_m = length[0][0]
    for key in dic.keys():
        if key == key_m:
            continue
        if key != key_m and abs(key - key_m) < 4:
            dic[key_m].extend(dic.pop(key))
        else:
            dic.pop(key)
    return dic[key_m]


def cluster1(pts):
    '''
    通过聚类，只保留定位到相同位置的所有数字中相似度最大的
    :param pts:
    :return:
    '''
    dic = {}
    for pt in pts:
        loc = pt[0],pt[1]
        if len(dic) == 0:
            dic[loc] = [pt]
            continue
        flag = False
        for key in dic.keys():
            if distance(loc,key) < 1000/50:
                dic[key].append(pt)
                center = means(dic[key])
                dic[center] = dic.pop(key)
                flag = True
                break
        if not flag:
            dic[loc] = [pt]

    # todo 这里做一步knn投票
    for key in dic:
        dic[key] = max(dic[key],key = lambda x:x[3])
    return dic.values()


def match(img_rgb):
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2YUV)[:,:,0]
    #todo  增强图片，利用幂次,(局部直方图均衡）---增加对比度

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    # img_gray = clahe.apply(img_gray)
    # plt.subplot(211), plt.imshow(img_gray, 'gray')
    # plt.subplot(212), plt.imshow(img, 'gray')

    path = "numbers_1"
    numbers = os.listdir(path)
    pts = []    #(pt[0],pt[1],label)
    # labels = []

    for number in numbers:
        template = cv2.imread(os.path.join(path,number))
        template = cv2.resize(template,(42,64))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2YUV)[:, :, 0]
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.35                                                  #todo 阈值设定
        #umpy.where(condition[, x, y])
        #Return elements, either from x or y, depending on condition. #If only condition is given, return condition.nonzero().
        loc = np.where( res >= threshold)
        loc = zip(*loc)
        pts.extend([(l[0],l[1],number[0],res[l[0],l[1]]) for l in loc])

    pts = cluster_rows(pts)
    pts = cluster1(pts)
    print len(pts)

    #todo 如果数字个数不符合规范，通过距离删除最不合适的

    pts = sorted(pts,key=lambda x:x[1])
    for pt in pts:
        print pt
        pt = pt[:2][::-1]
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    #todo：①校验 ②对数字做进一步识别

    plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    img_rgb = cv2.imread('cards/test6.jpg')
    r = 1
    img_rgb = img_rgb[340 / r:433 / r, :, :]

    match(img_rgb)