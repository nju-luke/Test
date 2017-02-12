# -*- coding: utf-8 -*-
# @Time    : 04/01/2017 15:39
# @Author  : Luke
# @Software: PyCharm

import matplotlib.pyplot as plt
import cv2
import numpy as np

# 获得视频的格式
from perspective_alignment_1 import dperspective_alignment

videoCapture = cv2.VideoCapture('cardVideo.mp4')
# videoCapture = cv2.VideoCapture(0)

# 获得码率及尺寸
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

# 读帧
success, frame = videoCapture.read()

frames = []
i = 0

rows,cols,depth = np.shape(frame)

new_vidio = cv2.VideoWriter("tester.avi", cv2.cv.CV_FOURCC('M','J','P','G'), 10.0,(rows,cols),True)



while success:
    # cv2.imshow("Video", frame)  # 显示
    # cv2.waitKey(1)#1000 / int(fps))  # 延迟
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2YUV)
    frame = cv2.resize(frame,(cols/2,rows/2))

    frame = np.flipud(np.transpose(frame,(1,0,2)))

    new_vidio.write(frame)
    # frame = dperspective_alignment(frame)

    frames.append(frame)
    # cv2.imshow("frame",frame)
    # cv2.waitKey(0)
    success, frame = videoCapture.read()  # 获取下一帧
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

new_vidio.release()
videoCapture.release()

#
# frames = np.asfarray(frames)
#
# frames_10 = np.sum(frames[:10,:,:],axis=0)
# tmp = frames_10/np.max(frames_10)
#
# # plt.imshow(tmp)
#
# print "Done"
