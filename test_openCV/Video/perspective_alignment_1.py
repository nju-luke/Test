# -*- coding: utf-8 -*-
# @Time    : 08/12/2016 14:06
# @Author  : Luke
# @Software: PyCharm

from util import *


def longest_line_P(img, length):
    nb_sec = 0.7
    line = cv2.HoughLinesP(img, 1, np.pi / 180, int(length * nb_sec))
    while line is None:
        nb_sec -= .1
        line = cv2.HoughLinesP(img, 1, np.pi / 180, int(length * nb_sec))
    if len(line) > 1:
        distances = [distance((x1, y1), (x2, y2)) for x1, y1, x2, y2 in line[0]]
        id = np.argmax(distances)
    else:
        id = 0
    return line[0][id]  # todo 优化用并行加速


def point2line(pt, lpt1, lpt2):
    A = lpt2[1] - lpt1[1]
    B = lpt1[0] - lpt2[0]
    C = lpt2[0] * lpt1[1] - lpt1[0] * lpt2[1]

    dis = abs(A * pt[0] + B * pt[1] + C) / np.sqrt(A ** 2 + B ** 2)
    return dis


def longest_line(img, length):
    nb_sec = 0.7
    lines = cv2.HoughLines(img, 1, np.pi / 180, int(length * nb_sec))
    while lines is None:
        if nb_sec <= 0.4:
            nb_sec -= 0.05
        else:
            nb_sec -= .1
        lines = cv2.HoughLines(img, 1, np.pi / 180, int(length * nb_sec))

    # blank = np.zeros(img.shape)
    lines_point = []
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines_point.append([x1, y1, x2, y2])
        # cv2.line(blank, (x1, y1), (x2, y2), 255, 2)
    # plt.subplot(211),plt.imshow(blank, 'gray')
    # plt.subplot(212),plt.imshow(img, 'gray')
    # plt.show()
    id = 0
    if len(lines_point) > 1:
        pt = np.array(img.shape) / 2
        distances = [point2line(pt, (x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines_point]
        id = np.argmax(distances)
    return lines_point[id]  # todo 优化用并行加速


def find_edge(img, direction):
    row, col = img.shape
    if direction is 'x':
        length = row
        width = col
        derivative_x, derivative_y = 1, 0
    else:
        length = col
        width = row
        derivative_x, derivative_y = 0, 1
    sobel = abs(cv2.Sobel(img, cv2.CV_64F, derivative_x, derivative_y, ksize=5))
    bin = sobel / np.max(sobel)
    bin = np.array(bin * 255, 'uint8')
    th, bin = cv2.threshold(bin, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if direction is 'x':
        line1 = longest_line(bin[:, :int(width / 2)], length)
        line2 = longest_line(bin[:, int(width   / 2):], length)
        line2[0] += width /2
        line2[2] += width /2
    else:
        line1 = longest_line(bin[:int(width / 2), :], length)
        line2 = longest_line(bin[int(width /2):, :], length)
        line2[1] += width /2
        line2[3] += width /2
    return [line1, line2]


def dperspective_alignment(img):
    line_x = find_edge(img, 'x')
    line_y = find_edge(img, 'y')

    pts1 = []

    def intersection(line1):
        x1, y1, x2, y2 = line1
        for line2 in line_y:
            x3, y3, x4, y4 = line2
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - x4 * y3)) / \
                ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - x4 * y3)) / \
                ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
            # print x, y
            pts1.append([x, y])

    intersection(line_x[0])
    intersection(line_x[1])

    pts1[1], pts1[2] = pts1[2], pts1[1]

    pts1 = np.array(pts1, 'float32')
    width = max(pts1[:, 0]) - pts1[0][0]
    height = max(pts1[:, 1]) - pts1[0][1]

    pts2 = np.zeros((4, 2), 'float32')
    pts2[0] = pts1[0]
    pts2[1] = pts1[0] + np.array([width, 0])
    pts2[2] = pts1[0] + np.array([0, height])
    pts2[3] = pts1[0] + np.array([width, height])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_img = cv2.warpPerspective(img, M, img.shape[::-1])

    new_img = new_img[int(pts2[0, 1]):int(pts2[3, 1]), int(pts2[0, 0]):int(pts2[3, 0])]

    # plt.imshow(new_img, 'gray')
    # plt.show()
    return new_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = cv2.imread('data/Test/5.jpg', 0)
    new_img = dperspective_alignment(img)
    new_img = cv2.resize(new_img, (500, 325))
    img = cv2.resize(img, (500, 325))
    #
    plt.subplot(121), plt.imshow(img, 'gray')
    plt.subplot(122), plt.imshow(new_img, 'gray')
    plt.show()
    # cv2.imwrite('/Users/hzqb_luke/Desktop/1.jpg',new_img)
