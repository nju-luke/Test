# -*- coding: utf-8 -*-
# @Time    : 10/24/16 14:41
# @Author  : Luke
# @Software: PyCharm


import numpy as np
import cv2

# from matplotlib import pyplot as plt




def drawMatches(img1, kp1, img2, kp2, matches, is_gray = True):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    if is_gray:
        out[:rows1, :cols1] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2, cols1:] = np.dstack([img2, img2, img2])
    else:
        out[:rows1, :cols1] = img1
        out[:rows2, cols1:] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def matched_dis(img1, kp1, img2, kp2, matches):
    dis = []
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        di = np.sqrt((x1-x2)**2+(y1-y2)**2)
        dis.append(di)
    return dis


if __name__ == '__main__':

    img1 = cv2.imread('普通光广发DIY信用卡.jpg', 0)  # queryImage
    img2 = cv2.imread('广发DIY信用卡.jpg', 0)  # trainImage
    img3 = cv2.imread('广发标准双金卡.jpg', 0)  # trainImage


    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    kp3, des3 = orb.detectAndCompute(img3, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    matches1 = bf.match(des1,des3)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    matches1 = sorted(matches1, key=lambda x: x.distance)

    # Draw first 10 matches.
    img11 = drawMatches(img1, kp1, img2, kp2, matches[:10])
    img12 = drawMatches(img1, kp1, img3, kp3, matches1[:10])

    dis12 = matched_dis(img1, kp1, img2, kp2, matches)
    dis13 = matched_dis(img1, kp1, img3, kp3, matches1)

    for di12,di13 in zip(dis12,dis13):
        print di12,di13

    cv2.imshow('dst', img3)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    # plt.imshow(img3),plt.show()
