# -*- coding: utf-8 -*-
# @Time    : 06/12/2016 10:49
# @Author  : Luke
# @Software: PyCharm

# from sklearn.metrics.pairwise import cosine_similarity
from util import *
from matplotlib import pyplot as plt

class Config:
    def __init__(self, width):
        self.width = width
        self.ratio = 1000 / width
        self.height = int(969 * width / 1536)

        self.template_w = int(np.ceil(44 * width / 1000))  # 39,62
        self.template_h = int(np.ceil(66 * width / 1000))
        margin_nb = 1.5
        self.margin_left = int(margin_nb * self.template_w)
        self.margin_right = width - int(margin_nb * self.template_w)

        self.row_top = 350 * width / 1000
        self.row_bottom = 450 * width / 1000

        self.threshold_1 = 300
        self.kernal = np.exp(-((np.array(range(10 / self.ratio)) - 4.5 / self.ratio) / 3 / self.ratio) ** 2)
        self.kernal_d = np.array([1, -1])

        self.location_vectors = [self.string2vectors('11110111101111011110'),  # 4-4-4-4
                                 self.string2vectors('11111101111111111111'),  # 6-13
                                 self.string2vectors('11111101111110111111'),  # 6-6-6
                                 self.string2vectors('11110111111011111000'),  # 4-6-5
                                 ]

    def string2vectors(self, str):
        return np.array([int(char) for char in str])


class Segment():
    def __init__(self, col):
        self.col = col
        self.config = Config(col)

    def y_index(self, Hy):
        Hy = Hy*(Hy > np.max(Hy) / 3)
        for i in range(len(Hy)):
            if Hy[i] > 0:
                id_up = i
                break
        for i in range(id_up+1,len(Hy)):
            if Hy[i] == 0:
                id_bottom = i
                break
        if id_bottom - id_up < self.config.template_h:
            mid = (id_up + id_bottom) / 2
            id_up = mid - self.config.template_h / 2
            id_bottom = mid + self.config.template_h / 2
        return id_up, id_bottom

    def binaryzation(self, sobel):
        img = np.array(sobel / np.max(sobel) * 255, 'uint8')
        img = cv2.GaussianBlur(img, (3, 3), 2)      #高斯模糊
        th, img_b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_b

    def crop_y(self, img_gray):
        '''
        横向切分数字所在的行。
        方法：利用二值化梯度的横向投影，找到三段数字出现最多的位置
        '''
        img = img_gray[self.config.row_top:self.config.row_bottom, :]

        # sobel,laplacian 边缘检测#
        sobelx = abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5))
        img_b = self.binaryzation(sobelx)
        Hy = np.sum(img_b, 1)
        row_1, row_2 = self.y_index(Hy)
        self.img_new = img[row_1:row_2, :]

    def enforcement(self,img):
        # sobel,laplacian 边缘检测#
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        # laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=5)
        laplacian = np.sqrt(sobely ** 2)
        img_e = np.array(laplacian / np.max(laplacian) * 255, 'uint8')
        img_g = cv2.GaussianBlur(img_e, (3, 3), 2)

        th, img_b = cv2.threshold(img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 用 GaussianBlur的结果比直接的laplacian边界效果好一些

        S = np.zeros(img_b.shape, 'uint8')
        row,col = img_b.shape

        nb_in = 5
        nb_out = 9
        l = nb_in/2
        k = nb_out/2

        start = nb_out / 2
        row_end = row - start - 1
        col_end = col - start - 1
        #
        for i in range(start, row_end):
            for j in range(start, col_end):
                if img_b[i, j] == 255:
                    img_tmp = img_e[i - l:i + l + 1, j - l:j + l + 1]
                    P_min = int(img_tmp.min())
                    P_max = int(img_tmp.max())
                    t = (P_max + P_min) / 2
                else:
                    continue
                for i_ in range(i - k, i + k + 1):
                    for j_ in range(j - k, j + k + 1):
                        if img_e[i_, j_] >= t:
                            S[i_, j_] += 1
        return S

    def pre_processing_x(self,img, left_index=0):
        '''
        对纵向投影的二值化梯度进行处理
        只用sobely的效果比sobely和Laplacian效果好
        '''
        sobely = abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5))
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = np.sqrt(sobelx**2+sobely**2)
        img_b = self.binaryzation(sobely)
        Hx = np.sum(img_b, 0)

        # S = self.enforcement(img)
        # Hx = np.sum(S,0)

        if len(Hx) == self.col:
            Hx[:self.config.template_w] = 0
            Hx[self.col - self.config.template_w:] = 0

        # 原理与Hy处理方式相似
        Hx = np.convolve(Hx, self.config.kernal, 'same') / len(self.config.kernal)
        Hx = Hx * (Hx < max(Hx) / 3) + max(Hx) * (1 - (Hx < max(Hx) / 3))
        Hx_d = np.convolve(Hx, self.config.kernal_d, 'same')

        x_index = [id for id in range(len(Hx_d) - 1) if Hx_d[id] * Hx_d[id + 1] <= 0
                   and not (Hx_d[id] == 0 and Hx_d[id + 1] == 0)]
        # x_index_v --> valley
        x_index_v = [i + left_index for i in x_index if Hx[i - 1] >= Hx[i] and Hx[i + 1] >= Hx[i]]

        ##合并数字间隔出现的两个点
        for i in range(len(x_index_v) - 1)[::-1]:
            if np.mean(Hx[x_index_v[i]:x_index_v[i + 1]]) < self.config.threshold_1 \
                    and x_index_v[i + 1] - x_index_v[i] < self.config.template_w * 0.7 / self.config.ratio:
                x_index_v[i] = (x_index_v[i] + x_index_v[i + 1]) / 2
                x_index_v.pop(i + 1)
        return Hx, x_index_v

    def gen_vector(self, numbers_coord):
        locations = [numbers_coord[0]]
        vector = [1]
        for i in range(1, len(numbers_coord)):
            if numbers_coord[i][0] == numbers_coord[i - 1][1]:
                if numbers_coord[i][1] - numbers_coord[i][0] < self.config.template_w * 1.5:
                    vector.append(1)
                    locations.append(numbers_coord[i])
                else:
                    nb_insert = int(np.round(
                        float(numbers_coord[i][1] - numbers_coord[i][0]) / (44 / self.config.ratio)) - 1)
                    vector.extend([0] * nb_insert)
                    locations.extend([None] * nb_insert)
                    vector.append(1)
                    locations.append(numbers_coord[i])
            else:
                left_coord = numbers_coord[i - 1][1]
                while numbers_coord[i][1] - left_coord > 30:
                    vector.append(0)
                    locations.append(None)
                    left_coord += self.config.template_w * 1.2
                vector.append(1)
                locations.append(numbers_coord[i])

        while len(vector) < 20:
            vector.append(0)
            locations.append(True)
        return vector, locations

    def vector_processing(self, numbers_coord):
        vector, locations = self.gen_vector(numbers_coord)

        if len(vector) > 20 or len(vector) < 10:
            return numbers_coord

        vector = np.array(vector)
        sims = []
        for vec in self.config.location_vectors:
            sims.append(cosine_similarity(vec,vector))
            # sims.append(np.sum(vec * vector) / np.linalg.norm(vec) / np.linalg.norm(vector))
            # sims.append(cosine_similarity(vec.reshape(1, -1), vector.reshape(1, -1)))
        type_nb = np.argmax(sims)
        vec = self.config.location_vectors[type_nb]

        while sum(vector == vec) < len(vec):
            for i in range(1, len(vec)):
                if vec[i] == vector[i] == 0:
                    locations[i] = True
                    continue
                if vec[i] == 0 and vector[i] == 1:
                    if vector[i - 1] == 0:
                        locations[i - 1] = [locations[i][0], locations[i][1] - 21]
                        vector[i - 1] = 1
                    locations[i] = True
                    vector[i] = 0
                if vec[i] == vector[i] == 1 \
                        and vector[i - 1] == 0 and not locations[i - 1] \
                        and locations[i]:
                    n = i
                    while not locations[n - 1] or vector[n - 1] == 0:
                        locations[n - 1] = [locations[n][0], locations[n][1] - 21]
                        locations[n] = [locations[n][1] - 21, locations[n][1]]
                        vector[n - 1] = 1
                        n -= 1
                        # todo 最后一位调整

        locations = [loc for loc in locations if loc is not True and loc is not None]
        for i in range(len(locations))[::-1]:
            if locations[i][1] - locations[i][0] < 10:
                locations.pop(i)
                try:
                    if vector[i] == 1 and vector[i - 1] == 1 and vector[i + 1] == 1:
                        locations.insert(i, [locations[i - 1][1], locations[i][0]])
                        continue
                except IndexError:
                    continue
        return locations

    def crop_x_part(self, numbers_coord):
        '''
        切分调整及细分
        1. 对切分多余的数字（不包括多个数字）进行截断，剩余数字部分
        2. 对比较长的数字重新投影,对于较长未能切分开的部分直接利用位置切分容易出错
        '''
        distances = [coord[1] - coord[0] for coord in numbers_coord]
        # 调整第一个数字的宽度
        if 50 / self.config.ratio < distances[0] < 76 / self.config.ratio \
                and abs(distances[1] - 44 / self.config.ratio) < 2:
            numbers_coord[0] = [numbers_coord[0][1] - int(42 / self.config.ratio), numbers_coord[0][1]]
        # 调整间隔处数字的宽度（依据前后数字做调整）
        for i in range(2, len(distances) - 2):
            if 50 / self.config.ratio < distances[i] < 80 / self.config.ratio:
                if distances[i - 1] < 45 / self.config.ratio and distances[i - 2] < 45 / self.config.ratio \
                        and numbers_coord[i][0] == numbers_coord[i - 1][1] \
                        and numbers_coord[i - 1][0] == numbers_coord[i - 2][1]:
                    numbers_coord[i][1] = numbers_coord[i][0] + 42 / self.config.ratio
                if distances[i + 1] < 45 / self.config.ratio and distances[i + 2] < 45 / self.config.ratio \
                        and numbers_coord[i][1] == numbers_coord[i + 1][0] \
                        and numbers_coord[i + 1][1] == numbers_coord[i + 2][0]:
                    numbers_coord[i][0] = numbers_coord[i][1] - 42 / self.config.ratio

        for index in range(len(distances))[::-1]:
            if distances[index] > 123 / self.config.ratio:
                img_tmp = self.img_new[:, numbers_coord[index][0]:numbers_coord[index][1]]
                # mean_th = np.mean(img_tmp)
                # th, tmp = cv2.threshold(img_tmp, mean_th, 255, cv2.THRESH_OTSU)
                # if mean_th > 128:
                #     tmp = 255 - tmp
                Hx, x_index_v = self.pre_processing_x(img_tmp, numbers_coord[index][0])
                numbers_coord_tmp = self.crop_x_l2r(Hx, x_index_v)
                numbers_coord.pop(index)
                for j in range(len(numbers_coord_tmp))[::-1]:
                    numbers_coord.insert(index, numbers_coord_tmp[j])
        return numbers_coord

    def crop_x_l2r(self, Hx, x_index_v):
        '''
        依据求导得到的数字边缘进行一次粗分，对符合两个数字的部分直接分为两个数字
        '''
        left_index = x_index_v[0]
        numbers_coord = []
        for i in range(1, len(x_index_v)):
            if x_index_v[i] - left_index > 35 / self.config.ratio:
                # todo 调整
                if np.mean(Hx[left_index:x_index_v[i]]) < self.config.threshold_1:
                    left_index = x_index_v[i]
                    continue
                # print left_index, x_index_v[i], x_index_v[i] - left_index, np.mean(Hx[left_index:x_index_v[i]])
                # todo 调整此处算法
                if 80 / self.config.ratio <= x_index_v[i] - left_index <= 96 / self.config.ratio:
                    numbers_coord.append([left_index, int(left_index + x_index_v[i]) / 2])
                    numbers_coord.append([int(left_index + x_index_v[i]) / 2, x_index_v[i]])
                else:
                    numbers_coord.append([left_index, x_index_v[i]])
                left_index = x_index_v[i]
                continue
            if np.mean(Hx[left_index:x_index_v[i]]) < self.config.threshold_1:
                left_index = x_index_v[i]
                continue
        return numbers_coord

    def crop_x_r2l(self, Hx, x_index_v):
        right_index = x_index_v[-1]
        numbers_coord = []
        for i in range(len(x_index_v) - 1)[::-1]:
            if right_index - x_index_v[i] > 35 / self.config.ratio:
                if np.mean(Hx[x_index_v[i]:right_index]) < self.config.threshold_1:
                    right_index = x_index_v[i]
                    continue
                # print x_index_v[i], right_index, right_index - x_index_v[i], np.mean(Hx[x_index_v[i]:right_index])
                if 80 / self.config.ratio <= right_index - x_index_v[i] <= 90 / self.config.ratio:
                    numbers_coord.append([int(right_index + x_index_v[i]) / 2, right_index])
                    numbers_coord.append([x_index_v[i], int(right_index + x_index_v[i]) / 2])
                else:
                    numbers_coord.append([x_index_v[i], right_index])
                right_index = x_index_v[i]
                continue
            if np.mean(Hx[x_index_v[i]:right_index]) < self.config.threshold_1:
                right_index = x_index_v[i]
                continue

        numbers_coord = numbers_coord[::-1]
        return numbers_coord

    def main(self, path):
        img_rgb = compression_core(path, col=self.col, row=self.config.height)
        img_gray = pre_processing(img_rgb, (self.config.width, self.config.height), 0)
        self.crop_y(img_gray)
        Hx, x_index_v = self.pre_processing_x(self.img_new)

        # segment from left to right
        @WaitTime(3)  # todo 设置等待时间
        def segment(func):

            #todo
            #todo 从中间某两个满足长度的位置去向两边扩展，而不从两边开始

            numbers_coord = func(Hx, x_index_v)
            numbers_coord = self.crop_x_part(numbers_coord)
            numbers_coord = self.vector_processing(numbers_coord)

            return numbers_coord

        def map_crop(func):
            try:
                return segment(func)
            except TimeOutException:
                return 'TimeOut'  # todo logger
            except IndexError:
                return "IndexError"

        funcs = [self.crop_x_l2r, self.crop_x_r2l]  #self.crop_x_l2r,
        numbers_coord_list = map(map_crop, funcs)

        numbers_list = []
        for j in range(len(numbers_coord_list)):
            if numbers_coord_list[j] is 'TimeOut':
                print "TimeOut"
                continue
            if numbers_coord_list[j] is 'IndexError':
                print "IndexError"
                continue
            numbers_coord = numbers_coord_list[j]
            numbers = []
            plt.subplot(2, 2, j + 1), plt.imshow(self.img_new, 'gray')
            for i in range(len(numbers_coord)):
                plt.subplot(2, 2 * len(numbers_coord), (2 + j) * len(numbers_coord) + i + 1)
                plt.imshow(self.img_new[:, numbers_coord[i][0]:numbers_coord[i][1]], 'gray')
                numbers.append(self.img_new[:, numbers_coord[i][0]:numbers_coord[i][1]])
                plt.xticks([])
                plt.yticks([])
            numbers_list.append(numbers)

        return numbers_list


if __name__ == '__main__':
    segment = Segment(500)
    numbers_list = segment.main('../Test/4.jpg')

    #
    # for filename in filenames:
    #     # if not filename == "3普通光浦发加速积分白金卡.JPG":
    #     #     continue
    #     # filename = filenames[27]
    #     path = "cards" + "/" + filename  # 27
    #     print path
    #     numbers_list = segment.main(path)
    #     for i in range(len(numbers_list)):
    #         for j in range(len(numbers_list[i])):
    #             number = numbers_list[i][j]
    #             number = cv2.resize(number,(21,32))
    #             new_name = os.path.join(unlabeled_path,filename.split(".")[0])+"_%s%s"%(i,j)+".jpg"
    #             cv2.imwrite(new_name,number)
    plt.show()
    # todo plot移出来
