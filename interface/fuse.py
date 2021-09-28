import os
import time
from pathlib import Path

import numpy as np

from utils.panorama import Stitcher

import imutils

import cv2


def is_integrated(file_names, fold_path):
    """
    从集合中挑选出可以配对的
    :param file_names: 名字
    :param fold_path: 文件的路径
    :return: 可配对的和不可配对的
    """
    false = set()
    for name in file_names.copy():
        if len(name.split("-")) == 1:
            file1 = os.path.join(fold_path, name + "-1.jpg")
            file2 = os.path.join(fold_path, name + "-2.jpg")
        else:
            file1 = os.path.join(fold_path, name.split("-")[0] + "-1-fx.jpg")
            file2 = os.path.join(fold_path, name.split("-")[0] + "-2-fx.jpg")

        if not Path(file1).exists() or not Path(file2).exists():
            file_names.remove(name)
            false.add(name)
    print("=" * 75)
    print("找到配对的图片：" + str(len(file_names)) + "对")
    return file_names, false


def get_unique_name(files):
    """
    将文件名放入标签，使用集合进行重复排除
    :param files: 文件名数组
    :return: 独一无二的文件名集合
    """
    name = set()
    for file in files:
        file_name = os.path.splitext(file)[0]
        labels = file_name.split("-")
        if len(labels) == 2:
            name.add(labels[0])
        else:
            name.add(labels[0] + "-fx")
    return name


def remove_back(img):
    """
    去除图像融合后的黑边
    :param img: 初始图像
    :return: res_img 去除黑边的图像
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    data = gray[499]
    data_reverse = data[::-1]
    location = w - (data_reverse != 0).argmax(axis=0)
    img_res = img[:, :location]     # 应当截取的中点位置
    return img_res


def img_fuse(files, fold_path, save_path):
    """
    图像融合
    :param files: 文件名
    :param fold_path: 文件夹路径
    :param save_path: 存储路径
    :return:
    """
    file_names = get_unique_name(files)
    file_names, false = is_integrated(file_names, fold_path)
    # 判断文件夹是否存在
    """paths = fold_path.split("\\")
    fold_name = paths[len(paths) - 1]
    save_path = os.path.join(save_path, fold_name)
    if not Path(save_path).exists():
        os.mkdir(save_path)
        print(fold_name)"""
    # 路径拼接
    for file_name in file_names:
        # 拼接正确的图片路径
        if len(file_name.split("-")) == 1:
            file1_path = os.path.join(fold_path, file_name + "-1.jpg")
            file2_path = os.path.join(fold_path, file_name + "-2.jpg")
        else:
            file1_path = os.path.join(fold_path, file_name.split("-")[0] + "-1-fx.jpg")
            file2_path = os.path.join(fold_path, file_name.split("-")[0] + "-2-fx.jpg")
        # 图像融合
        imageA = cv2.imread(file1_path)
        imageB = cv2.imread(file2_path)
        imageA = imutils.resize(imageA, width=1000)
        imageB = imutils.resize(imageB, width=1000)
        stitcher = Stitcher()
        (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
        result = remove_back(result)

        new_path = os.path.join(save_path, file_name + ".jpg")
        cv2.imwrite(new_path, result)


def img_fuse2():
    """
    图像融合方法2，解决图像融合后左右的颜色存在差异的问题，但是花费的时间较长
    :return:
    """
    time_start = time.time()
    MIN = 10
    img1 = cv2.imread(r'D:\dataset\Bubble\20180718\1059-1-fx.jpg')  # query
    img2 = cv2.imread(r'D:\dataset\Bubble\20180718\1059-2-fx.jpg')  # train
    # 选择特征类型
    # surf = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
    surf=cv2.xfeatures2d.SIFT_create()  # 可以改为SIFT
    kp1, descrip1 = surf.detectAndCompute(img1, None)
    kp2, descrip2 = surf.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    match = flann.knnMatch(descrip1, descrip2, k=2)

    good = []
    for i, (m, n) in enumerate(match):
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) > MIN:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
        warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
        direct = warpImg.copy()
        direct[0:img1.shape[0], 0:img1.shape[1]] = img1
        simple = time.time()
        rows, cols = img1.shape[:2]
        for col in range(0, cols):
            if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                    res[row, col] = warpImg[row, col]
                elif not warpImg[row, col].any():
                    res[row, col] = img1[row, col]
                else:
                    srcImgLen = float(abs(col - left))
                    testImgLen = float(abs(col - right))
                    alpha = srcImgLen / (srcImgLen + testImgLen)
                    res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

        warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
        final = time.time()
        img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
        # plt.imshow(img3, ), plt.show()
        img4 = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
        # plt.imshow(img4, ), plt.show()
        print("simple stich cost %f" % (simple - time_start))
        print("\ntotal cost %f" % (final - time_start))
        cv2.imwrite(r"D:\Experiments\res\fuse\simplepanorma.png", direct)
        cv2.imwrite(r"D:\Experiments\res\fuse\bestpanorma.png", warpImg)
    else:
        print("not enough matches!")
    print(f'单次融合时间为：{time.time() - time_start:.3f}s')
