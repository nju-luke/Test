# -*- coding: utf-8 -*-
# @Time    : 24/11/2016 15:26
# @Author  : Luke
# @Software: PyCharm
from sklearn.metrics.pairwise import cosine_similarity

from util import *
from itertools import groupby


class Config:
    def __init__(self, width):
        self.ratio = 1000 / col
        self.height = int(969 * width / 1536)

        self.threshold_1 = 300

        self.template_w = int(np.ceil(39 * width / 1000))  # 39,62
        self.template_h = int(np.ceil(62 * width / 1000))
        margin_nb = 1.5
        self.margin_left = int(margin_nb * self.template_w)
        self.margin_right = width - int(margin_nb * self.template_w)

        self.row_top = 340 * width / 1000
        self.row_bottom = 433 * width / 1000

        self.kernal = np.exp(-((np.array(range(10 / self.ratio)) - 4.5 / self.ratio) / 3 / self.ratio) ** 2)
        self.kernal_d = np.array([1, -1])

        self.location_vectors = [self.string2vectors('11110111101111011110'),  # 4-4-4-4
                                 self.string2vectors('11111101111111111111'),  # 6-13
                                 self.string2vectors('11111101111110111111'),  # 6-6-6
                                 self.string2vectors('11110111111011111000'),  # 4-6-5
                                ]

    def string2vectors(self, str):
        return np.array([int(char) for char in str])



filenames = [fi for fi in os.listdir("cards") if not fi.startswith(".")]
path = "cards" + "/" + filenames[17]  # 27
img_rgb = cv2.imread(path)

col = 500
config = Config(col)
img_rgb = compression_core(path, col=col,row = config.height)
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_gray = pre_processing(img_rgb, 0)

row, col = img_gray.shape

img = img_gray[config.row_top:config.row_bottom, :]

# sobel,laplacian 边缘检测#
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
laplacian = np.sqrt(sobelx ** 2 + sobely ** 2)
img_e = np.array(laplacian / np.max(laplacian) * 255, 'uint8')

img_g = cv2.GaussianBlur(img_e, (3, 3), 2)
# img_e = img_g

th, img_b = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

Hy = np.sum(img_b, 1)
# kernal = np.ones(10, 'float')

# kernal = np.exp(-((np.array(range(15)) - 7) / 7) ** 2)

Hy = np.convolve(Hy, config.kernal, 'same') / len(config.kernal)
Hy *= Hy > np.max(Hy) / 3
# plt.subplot(211),plt.plot(Hy)
Hy1 = Hy

Hy = np.convolve(Hy, config.kernal_d, 'same')

y_index = [id for id in range(len(Hy) - 1) if Hy[id] > 0 and Hy[id + 1] < 0]
distance = [y_index[i] - y_index[i - 1] for i in range(1, len(y_index))]
id = np.argmin([abs(distance[i] - distance[i + 1]) if distance[i] > 15 else 20 for i in range(len(distance) - 1)])
half_length = (distance[id] + distance[id + 1]) / 2 / 2

row_1 = y_index[id] + config.row_top - half_length
row_2 = y_index[id + 2] + config.row_top + half_length

img_new = img_gray[row_1:row_2, :]

img_b = img_b[max(0, y_index[id] - half_length):min(config.row_bottom - config.row_top, y_index[id + 2] + half_length), :]
#
Hx = np.sum(img_b, 0)
Hx[:int(0.08 * col)] = 0
Hx[col - int(0.08 * col):] = 0

Hx = np.convolve(Hx, config.kernal, 'same') / len(config.kernal)
# plt.subplot(211),plt.plot(Hy)
Hx = Hx * (Hx < max(Hx) / 3) + max(Hx) * (1 - (Hx < max(Hx) / 3))

Hx_d = np.convolve(Hx, config.kernal_d, 'same')

x_index = [id for id in range(len(Hx_d) - 1) if Hx_d[id] * Hx_d[id + 1] <= 0
           and not (Hx_d[id] == 0 and Hx_d[id + 1] == 0)]

# x_index_v --> valley
x_index_v = [i for i in x_index if Hx[i - 1] >= Hx[i] and Hx[i + 1] >= Hx[i]]

##合并数字间隔出现的两个点
for i in range(len(x_index_v) - 1)[::-1]:
    if np.mean(Hx[x_index_v[i]:x_index_v[i + 1]]) < 10 \
            and x_index_v[i + 1] - x_index_v[i] < 15 / config.ratio:  # todo 调整参数
        x_index_v[i] = (x_index_v[i] + x_index_v[i + 1]) / 2
        x_index_v.pop(i + 1)

print x_index_v

## 切分数字
def post_processing(numbers_coord):
    '''
    1. 对切分多余的数字（不包括多个数字）进行截断，剩余数字部分
    2. 对比较长的数字重新投影
    '''
    distances = [coord[1] - coord[0] for coord in numbers_coord]
    for cod, dis in zip(numbers_coord, distances):
        print cod, dis

    if 50 / config.ratio < distances[0] < 76 / config.ratio \
            and abs(distances[1] - 44/config.ratio) < 2:
        numbers_coord[0] = [numbers_coord[0][1] - 42/config.ratio,numbers_coord[0][1]]

    for i in range(2, len(distances) - 2):
        if 50 / config.ratio < distances[i] < 80 / config.ratio:
            if distances[i - 1] < 45/config.ratio and distances[i - 2] < 45/config.ratio \
                    and numbers_coord[i][0] == numbers_coord[i - 1][1]\
                    and numbers_coord[i-1][0] == numbers_coord[i - 2][1]:
                numbers_coord[i][1] = numbers_coord[i][0] + 21
            if distances[i + 1] < 45/config.ratio and distances[i + 2] < 45/config.ratio \
                    and numbers_coord[i][1] == numbers_coord[i + 1][0]\
                    and numbers_coord[i+1][1] == numbers_coord[i + 2][0]:
                numbers_coord[i][0] = numbers_coord[i][1] - 42/config.ratio
    print numbers_coord

    for index in range(len(distances))[::-1]:
        if distances[index] > 126/config.ratio:
            img_tmp = img_new[:, numbers_coord[index][0]:numbers_coord[index][1]]
            mean_th = np.mean(img_tmp)
            th, tmp = cv2.threshold(img_tmp, mean_th, 255, cv2.THRESH_OTSU)
            if mean_th > 128:
                tmp = 255 - tmp
            # plt.plot(np.sum(tmp, 0), 'gray')
            Hx = np.sum(tmp, 0)
            Hx = np.convolve(Hx, config.kernal, 'same') / len(config.kernal)
            # plt.subplot(211),plt.plot(Hy)
            Hx = Hx * (Hx < max(Hx) / 3) + max(Hx) * (1 - (Hx < max(Hx) / 3))
            #
            Hx_d = np.convolve(Hx, config.kernal_d, 'same')

            x_index = [id for id in range(len(Hx_d) - 1) if Hx_d[id] * Hx_d[id + 1] <= 0
                       and not (Hx_d[id] == 0 and Hx_d[id + 1] == 0)]

            # x_index_v --> valley
            x_index_v = [i+numbers_coord[index][0] for i in x_index if Hx[i - 1] >= Hx[i] and Hx[i + 1] >= Hx[i]]

            ##合并数字间隔出现的两个点
            for i in range(len(x_index_v) - 1)[::-1]:
                if np.mean(Hx[x_index_v[i]:x_index_v[i + 1]]) < config.threshold_1 \
                        and x_index_v[i + 1] - x_index_v[i] < config.template_w * 0.7 / config.ratio:
                    x_index_v[i] = (x_index_v[i] + x_index_v[i + 1]) / 2
                    x_index_v.pop(i + 1)

            # plt.plot(Hx)
            # plt.plot(x_index_v, np.ones(len(x_index_v)) * 1000, '*')
            # plt.show()

            left_index = x_index_v[0]
            numbers_coord_1 = []
            for i in range(1, len(x_index_v)):

                if x_index_v[i] - left_index > 35 / config.ratio:
                    if np.mean(Hx[left_index:x_index_v[i]]) < config.threshold_1:
                        left_index = x_index_v[i]
                        continue
                    # print left_index, x_index_v[i], x_index_v[i] - left_index, np.mean(Hx[left_index:x_index_v[i]])
                    if 80 / config.ratio <= x_index_v[i] - left_index <= 96 / config.ratio:
                        numbers_coord_1.append([left_index, int(left_index + x_index_v[i]) / 2])
                        numbers_coord_1.append([int(left_index + x_index_v[i]) / 2, x_index_v[i]])
                    else:
                        numbers_coord_1.append([left_index, x_index_v[i]])
                    left_index = x_index_v[i]
                    continue
                if np.mean(Hx[left_index:x_index_v[i]]) < config.threshold_1:
                    left_index = x_index_v[i]
                    continue
            print numbers_coord_1
            numbers_coord.pop(index)
            for j in range(len(numbers_coord_1))[::-1]:
                numbers_coord.insert(index,numbers_coord_1[j])
    print numbers_coord

def vector_loc(numbers_coord):
    locations = [numbers_coord[0]]
    vector = [1]
    for i in range(1,len(numbers_coord)):
        if numbers_coord[i][0] == numbers_coord[i - 1][1]:
            if numbers_coord[i][1] - numbers_coord[i][0] < config.template_w*1.5:
                vector.append(1)
                locations.append(numbers_coord[i])
            else:
                nb_insert = int(np.ceil(
                    float(numbers_coord[i][1] - numbers_coord[i][0]) / 21) - 1)
                vector.extend([0]*nb_insert)
                locations.extend([None]*nb_insert)
                vector.append(1)
                locations.append(numbers_coord[i])
        else:
            left_coord = numbers_coord[i-1][1]
            while numbers_coord[i][1] - left_coord > 30:
                vector.append(0)
                locations.append(None)
                left_coord += config.template_w
            vector.append(1)
            locations.append(numbers_coord[i])

    while len(vector)<20:
        vector.append(0)
        locations.append(True)
    return vector,locations

def processing_vector(numbers_coord):
    vector,locations = vector_loc(numbers_coord)

    if len(vector) > 20 or len(vector) < 10:
        return numbers_coord

    vector = np.array(vector)
    sims = []
    for vec in config.location_vectors:
        sims.append(cosine_similarity(vec.reshape(1, -1), vector.reshape(1, -1)))
    type_nb = np.argmax(sims)
    vec = config.location_vectors[type_nb]
    print vector,type_nb
    print vec

    while sum(vector == vec) < len(vec):
        for i in range(1, len(vec)):
            if vec[i] == vector[i] == 0:
                locations[i] = True
                continue
            if vec[i] == 0 and vector[i] == 1:
                if vector[i-1] == 0:
                    locations[i-1] = [locations[i][0],locations[i][1] - 21]
                    vector[i-1] = 1
                locations[i] = True
                vector[i] = 0
            if vec[i] == vector[i] == 1 \
                    and vector[i - 1] == 0 and not locations[i - 1] \
                    and locations[i]:
                n = i
                while not locations[n-1] or vector[n-1] == 0:
                    locations[n - 1] = [locations[n][0],locations[n][1] - 21]
                    locations[n] = [locations[n][1] - 21, locations[n][1]]
                    vector[n - 1] = 1
                    n -= 1

    locations = [loc for loc in locations if loc is not True and loc is not None]
    for i in range(len(locations))[::-1]:
        if locations[i][1] - locations[i][0] < 10:
            locations.pop(i)
    return locations


def column_crop_l2r():
    left_index = x_index_v[0]
    numbers_coord = []
    for i in range(1, len(x_index_v)):
        #
        # todo 开头的地方判断

        if x_index_v[i] - left_index > 35 / config.ratio:
            if np.mean(Hx[left_index:x_index_v[i]]) < config.threshold_1:
                left_index = x_index_v[i]
                continue
            # print left_index, x_index_v[i], x_index_v[i] - left_index, np.mean(Hx[left_index:x_index_v[i]])
            if 80 / config.ratio <= x_index_v[i] - left_index <= 96 / config.ratio:
                numbers_coord.append([left_index, int(left_index + x_index_v[i]) / 2])
                numbers_coord.append([int(left_index + x_index_v[i]) / 2, x_index_v[i]])
            else:
                numbers_coord.append([left_index, x_index_v[i]])
            left_index = x_index_v[i]
            continue
        if np.mean(Hx[left_index:x_index_v[i]]) < config.threshold_1:
            left_index = x_index_v[i]
            continue

    # todo 根据长度与左右的关系，调整、删除
    post_processing(numbers_coord)
    numbers_coord = processing_vector(numbers_coord)

    print len(numbers_coord)
    print numbers_coord
    plt.subplot(211), plt.imshow(img_new, 'gray')
    for i in range(len(numbers_coord)):
        plt.subplot(2, len(numbers_coord), len(numbers_coord) + i + 1)
        plt.imshow(img_new[:, numbers_coord[i][0]:numbers_coord[i][1]], 'gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()


column_crop_l2r()


def column_crop_r2l():
    right_index = x_index_v[-1]
    numbers_coord = []
    for i in range(len(x_index_v) - 1)[::-1]:
        if right_index - x_index_v[i] > 35 / config.ratio:
            if np.mean(Hx[x_index_v[i]:right_index]) < config.threshold_1:
                right_index = x_index_v[i]
                continue
            # print x_index_v[i], right_index, right_index - x_index_v[i], np.mean(Hx[x_index_v[i]:right_index])
            if 80 / config.ratio <= right_index - x_index_v[i] <= 90 / config.ratio:
                numbers_coord.append([int(right_index + x_index_v[i]) / 2, right_index])
                numbers_coord.append([x_index_v[i], int(right_index + x_index_v[i]) / 2])
            else:
                numbers_coord.append([x_index_v[i], right_index])
            right_index = x_index_v[i]
            continue
        if np.mean(Hx[x_index_v[i]:right_index]) < config.threshold_1:
            right_index = x_index_v[i]
            continue

    numbers_coord = numbers_coord[::-1]
    post_processing(numbers_coord)
    numbers_coord = processing_vector(numbers_coord)

    print len(numbers_coord)
    print numbers_coord
    plt.subplot(211), plt.imshow(img_new, 'gray')
    for i in range(len(numbers_coord)):
        plt.subplot(2, len(numbers_coord), len(numbers_coord) + i + 1)
        plt.imshow(img_new[:, numbers_coord[i][0]:numbers_coord[i][1]], 'gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()


column_crop_r2l()

# x_index = [id for id in range(len(Hx_d) - 1) if Hx_d[id] <= 0 and Hx_d[id + 1] > 0]
yy = np.array(np.ones(len(x_index_v))) * 1000

x_dif = [x_index[i + 1] - x_index[i] for i in range(len(x_index) - 1)]

# todo 聚类
# todo or 不用聚类，利用规则，在边缘向前向后0.04*col范围如果出现边缘，则当做一个数字（前提是在两个边缘之间没有出现另一个边缘）
# todo 如何判断切分部分为数字的可信度？


plt.figure()
plt.subplot(211), plt.imshow(img_new, 'gray')
plt.subplot(212)
plt.plot(Hx, '-')
plt.plot(x_index_v, yy, "*")
plt.show()
