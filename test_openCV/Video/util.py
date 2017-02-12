# -*- coding: utf-8 -*-
# @Time    : 14/11/2016 09:08
# @Author  : Luke
# @Software: PyCharm
import logging
import numpy as np
import cv2
import os
import signal

# from perspective_alignment_1 import dperspective_alignment
import matplotlib.pyplot as plt



def pre_processing(img, shape = None, ch=0, align = True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, ch]
    if align:
        img_gray = dperspective_alignment(img_gray)

    if shape:
        img_gray = cv2.resize(img_gray, shape)

    # plt.imshow(img_gray)
    # plt.show()
    img_gray = np.array(img_gray, 'float') / 255
    hist1 = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])
    weight_global = np.sum(np.array(np.squeeze(hist1)) * np.array(range(256)) / sum(hist1))
    ch = 2 ** ((weight_global - 127.) / 127)
    return np.array(img_gray ** ch * 255, 'uint8')


def load_templates(path, shape):
    names = [name for name in os.listdir(path) if not name.startswith(".")]
    mat_templates = []
    for name in names:
        template = cv2.imread(os.path.join(path, name))
        template = cv2.resize(template, shape)
        template = pre_processing(template,align=False)
        mat_templates.append(template)

    return names, mat_templates


distance = lambda pt1, pt2: np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def means(point_list, k=2):
    x_ = np.mean([pt[0] for pt in point_list])
    y_ = np.mean([pt[1] for pt in point_list])
    if k == 1:
        return np.ceil(x_)
    return np.ceil(x_), np.ceil(y_)


def validation_luhn(numbers):
    numbers = [int(char) for char in numbers[::-1]]
    numbers_odd = numbers[::2]
    numbers_even = np.array(numbers[1::2]) * 2
    numbers_even = numbers_even % (9 * (2 - (numbers_even > 9)))
    if (sum(numbers_even) + sum(numbers_odd)) % 10 == 0:
        return True
    else:
        return False


def compression_core(img_path, col=1000, row=None):
    img = cv2.imread(img_path)
    if row == None:
        row = col * 636 / 1000
    return cv2.resize(img, (col, row))


def compression(folder, new_path, col=1000):
    prefix = folder.split("/")[-1]
    filenames = [fi for fi in os.listdir(folder) if not fi.startswith(".")]
    for fi in filenames:
        path = os.path.join(folder, fi)
        if os.path.isdir(path):
            compression(os.path.join(folder, path), new_path)
            continue
        img = compression_core(path, col)
        # cv2.imshow('img',img)
        cv2.imwrite(os.path.join(new_path, prefix + fi), img)


class TimeOutException(Exception):
    pass


def WaitTime(num):
    def wrape(func):
        def handle(signum, frame):
            raise TimeOutException("运行超时！")

        def toDo(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)  # 开启闹钟信号
                rs = func(*args, **kwargs)
                signal.alarm(0)  # 关闭闹钟信号
                return rs
            except TimeOutException, e:
                raise TimeOutException
        return toDo
    return wrape


def logger(log_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=log_dir,
                        filemode='a')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-8s: %(levelname)-8s ：%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

def cosine_similarity(vec1,vec2):
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
    return np.sum(vec1 * vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

if __name__ == '__main__':
    path = "/Users/hzqb_luke/Documents/data/cards/test_v2_hz/val"
    new_path = "cards"
    compression(path, new_path, col=500)
