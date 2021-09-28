import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


def nothing(x):
    pass


# 伽马变换
def track(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('parameter', 'image', 0, 200, nothing)

    while True:
        # 伽马变换
        parameter = float(cv2.getTrackbarPos('parameter', 'image') / 100)
        res = filter_choose(img, 4)
        cv2.imshow("image", res)
        if cv2.waitKey(1) == ord('q'):
            break


# 伽马变换
def gamma(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    fI = gray / 255.0
    res = np.power(fI, 1.6)
    return res


# 全局直方图均衡化
def contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_equalize = cv2.equalizeHist(gray)
    return img_equalize


# 对比度限制自适应直方图均衡
def clahe(img):
    B, G, R = cv2.split(img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    clahe = cv2.createCLAHE(clipLimit=60.0, tileGridSize=(1, 1))
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    clahe_test = cv2.merge((clahe_B, clahe_G, clahe_R))
    return clahe_test


# 线性变换
def line(img):
    res = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    return res


# 阈值处理
def threshold_track(img):
    """
    带有拖动条的阈值分割
    :param img:
    :param ori:
    :return:
    """
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while True:
        threshold = cv2.getTrackbarPos('threshold', 'image')
        ret, res = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        # ret, res = cv2.threshold(img, 255 - threshold, 255, cv2.THRESH_BINARY_INV)
        # compare = image_join(1, [img, res])
        cv2.imshow("image", res)
        if cv2.waitKey(1) == ord('q'):
            break


def filter_choose(img, type=1):
    """
    滤波器
    :param img: 初始图像
    :param type: 滤波器选择
    :return: 滤波后的图像
    """
    if type == 1:  # 高斯滤波
        res = cv2.GaussianBlur(img, (3, 3), 0)
    elif type == 2:  # 中值滤波
        res = cv2.medianBlur(img, 5)
    elif type == 3:  # 双边滤波
        res = cv2.bilateralFilter(img, 7, 75, 75)
    elif type == 4:  # 拉普拉斯算子
        res = cv2.Laplacian(img, cv2.CV_64F)
    elif type == 5:  # sobel算子
        res = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    else:
        res = img
    return res


def slic_seg(image_path):
    """
    slic超像素分割
    :param image_path: 图片路径
    :return:
    """
    image = img_as_float(io.imread(image_path))
    for numSegments in (30, 50):
        segments = slic(image, n_segments=numSegments, sigma=5)
        fig = plt.figure("Superpixels -- %d segments" % (numSegments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.axis("off")
    plt.show()


def image_compare(scale, imgArray):
    """
    图像拼接
    :param scale: 大小
    :param imgArray: 图像数组
    :return: 分别放置的图像
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def image_join(img1, img2):
    """
    图像拼接
    :param img1: 图像1
    :param img2: 图像2
    :return: 拼接后的图像
    """
    (h1, w1, c1) = img1.shape
    (h2, w2, c2) = img2.shape
    width = w1 + w2
    height = max(h1, h2)
    # 初始化一个黑色图像，方便后续填充
    result = np.zeros((height, width, 3), 'uint8')
    # 开始图像拼接
    result[0:h1, 0:w1] = img1   # 拼接左图
    result[0:h2, w1:] = img2    # 拼接右图
    return result


def frequency_show(img_path):
    """
    展示图片的频谱
    :param img_path: 图像路径
    :return:
    """
    img = cv2.imread(img_path, 0)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dftshift = np.fft.fftshift(dft)
    result = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('ori_img'), plt.axis('off')
    plt.subplot(122), plt.imshow(result, cmap='gray')
    plt.title('frequency'), plt.axis('off')
    plt.show()
