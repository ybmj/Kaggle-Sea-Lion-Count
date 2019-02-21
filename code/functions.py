import os
import numpy as np
import cv2
import skimage.feature
import matplotlib.pyplot as plt


r = 0.4  # scale down
width = 300  # patch size


def shrink(img):
    h, w = img.shape[:2]
    size = (int(w * 0.3), int(h*0.3))
    res = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.namedWindow("fuck")
    cv2.imshow("fuck", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res


def data_process(path1, path2):
    '''
    :param path1 : img1的路径
    :param path2 : img2
    :return: trainX, trainY (np.array)
    '''
    origin = cv2.imread(path1)  # 原始图像
    originDotted = cv2.imread(path2)  # 标记后的图像
    img1 = cv2.GaussianBlur(origin, (5, 5), 0)  # 高斯滤波，用于去除高斯噪声

    # 取原始图像和标记后图像之差的绝对值
    absdiff = cv2.absdiff(origin, originDotted)
    # 将背景置为黑色，只保留标记点
    mask_1 = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    dotImg = cv2.bitwise_or(absdiff, absdiff, mask=mask_1)

    # 转换成灰度图，进行斑点检测convert to grayscale to be accepted by skimage.feature.blob_log
    grayImg = np.max(dotImg, axis=2)
    blobs = skimage.feature.blob_log(
        grayImg, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)
    # blob 或者叫斑点，就是在一幅图像上，暗背景上的亮区域。或者亮背景上的暗区域，都能够称为blob
    h, w, d = originDotted.shape
    res = np.zeros(
        (int((w*r)//width)+1, int((h*r)//width)+1, 5), dtype='int16')

    for blob in blobs:
        # 斑点的坐标
        y, x, s = blob
        # 得到斑点中心的rgb的值（颜色）
        b, g, R = img1[int(y)][int(x)][:]
        x1 = int((x*r)//width)
        y1 = int((y*r)//width)
        # 根据rgb值判断颜色
        if R > 225 and b < 25 and g < 25:  # RED
            res[x1, y1, 0] += 1
        elif R > 225 and b > 225 and g < 25:  # MAGENTA
            res[x1, y1, 1] += 1
        elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
            res[x1, y1, 4] += 1
        elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
            res[x1, y1, 3] += 1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1, y1, 2] += 1

    ma = cv2.cvtColor((1*(np.sum(origin, axis=2) > 20)
                       ).astype('uint8'), cv2.COLOR_GRAY2BGR)

    img = cv2.resize(originDotted * ma, (int(w*r), int(h*r)))
    return cutFigure(img, res, h, w)


def cutFigure(image, res, h, w):
    # 将图片重置大小，保证平均分割
    img = cv2.resize(image, (int(w*r), int(h*r)))
    h1, w1, d = img.shape
    trainX = []
    trainY = []
    for i in range(int(w1//width)):
        for j in range(int(h1//width)):
            # 分割好的图片放入trainX中
            trainX.append(img[j*width:j*width+width, i*width:i*width+width, :])
            # 分割好的图片中，各种标记点的数量放到trainY中
            trainY.append(res[i, j, :])
    return np.array(trainX), np.array(trainY)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
