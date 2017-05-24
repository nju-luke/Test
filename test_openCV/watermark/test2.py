# -*- coding: utf-8 -*-
# @Time    : 09/02/2017 17:45
# @Author  : Luke
# @Software: PyCharm

import numpy as np
import cv2
import matplotlib.pyplot as plt

row = 512
col = 323

shape = (row, col)
union = "watermark_2.jpg"
watermark_1 = cv2.resize(cv2.imread(union,0), shape)
watermark_1 = (watermark_1 > 128) * 1

np.random.seed(128)
watermark_1 = ((watermark_1 + (np.random.randint(0,9,(col,row))<3))>0) * 2 - 1

# plt.imshow(watermark_1,'gray')
# plt.show()


# 打乱水印
coordinates = []
for x in range(col):
    for y in range(row):
        coordinates.append((x, y))

coordinates_backup = coordinates[:]

np.random.seed(13241)
np.random.shuffle(coordinates)

watermark_2 = np.zeros_like(watermark_1)
n = 0
for x2 in range(col):
    for y2 in range(row):
        x1, y1 = coordinates[n]
        watermark_2[x2, y2] = watermark_1[x1, y1]
        n += 1

# 重组水印
# img = np.zeros_like(watermark_1)
# n = 0
# for x2 in range(col):
#     for y2 in range(row):
#         x1,y1 = coordinates[n]
#         img[x1,y1] = watermark_2[x2,y2]
#         n+=1

# 限制中心对称


x_center = col / 2
y_center = row / 2
R_outer = min(row, col) / 2 * 0.75
R_inner = min(row, col) / 2 * 0.25
concentric_annulus = np.zeros_like(watermark_1)
for x2 in range(col):
    for y2 in range(row):
        if R_inner < np.sqrt((x2 - x_center) ** 2 + (y2 - y_center) ** 2) < R_outer:
            concentric_annulus[x2, y2] = 1

embed = watermark_2 * concentric_annulus
plt.imshow(embed)
plt.show()


#
# # 打水印
img1 = "3ca77a5d-d89e-4234-8cc9-9801388ba95e.jpg"
img1 = np.asfarray(cv2.imread(img1))
# img1/=255
img2 = np.zeros_like(img1)

def watermarking(Z):
    mag = np.abs(Z)
    angle = np.angle(Z)
    mag = mag + mag * 1 * embed
    Z_watermarked = mag * np.cos(angle) + 1j * mag * np.sin(angle)
    return Z_watermarked


for ch in range(3):
    img1_1 = img1[:, :, ch]
    fft_1 = np.fft.fft2(img1_1)
    fft_1 = watermarking(fft_1)
    img1_f1 = abs(np.fft.ifft2(fft_1))
    img2[:, :, ch] = img1_f1
cv2.imwrite("test.jpg", img2)
#
img2 = cv2.imread("test.jpg")



plt.imshow(img2[:, :, ::-1])
plt.show()
#
# # 检测
img_R = img2[:, :, 0]
mag_new = np.abs(np.fft.fft2(img_R))

def find_nonzeros_sequence(mark, magnitude):
    seq_mark = []
    seq_mag = []
    row, col = mark.shape
    for i in range(row):
        for j in range(col):
            if mark[i, j] == 0:
                continue
            seq_mag.append(magnitude[i, j])
            seq_mark.append(mark[i, j])
    return seq_mark,seq_mag


x = range(13230,13250)
result1 = []
result2 = []

for seed in x:
    coordinates = coordinates_backup[:]
    np.random.seed(seed)
    np.random.shuffle(coordinates)
    watermark_2 = np.zeros_like(watermark_1)
    n = 0
    for x2 in range(col):
        for y2 in range(row):
            x1, y1 = coordinates[n]
            watermark_2[x2, y2] = watermark_1[x1, y1]
            n += 1
    embed1 = watermark_2 * concentric_annulus

# 第一种方法
    seq_mark, seq_mag = find_nonzeros_sequence(embed1,mag_new)
    corr = np.corrcoef(seq_mark,seq_mag)[0,1]
    result1 .append(corr)

# 第二种方法
    corr = np.sum(embed1*mag_new)
    result2.append(corr)

print np.pi*(R_outer**2-R_inner**2)

plt.subplot(211)
plt.plot(x,result1)
plt.subplot(212)
plt.plot(x,result2)
plt.show()



'''
1. 内外界大小设置
2. 正负值大小可不一样
3.
'''