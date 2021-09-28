import os
from pathlib import Path

import cv2

from interface.detect_hxq import detect_hxq
from interface.fuse import get_unique_name, img_fuse
from my_test.correct import img_correct
from utils.image_process import clahe, filter_choose
from utils.util import *


def batch_process(path, save_path):
    """
    批量处理
    :param path:
    :return:
    """
    i = 0
    # 该文件夹下所有的文件（包括文件夹）
    FileList = os.listdir(path)
    # 遍历所有文件
    for file in FileList:
        # 原来的文件路径
        oldDirPath = os.path.join(path, file)
        # 如果是文件夹则递归调用
        if os.path.isdir(oldDirPath):
            fold_create(save_path, file)
            batch_process(oldDirPath, os.path.join(save_path, file))
        else:
            # 文件信息
            fileName = os.path.splitext(file)[0]       # 文件名
            fileType = os.path.splitext(file)[1]       # 文件扩展名
            # 如果当前的不是文件夹，则把path送入
            """ 图像融合
            img_fuse(FileList, path, save_path)
            break
            """
            """ 图像矫正
            res = img_correct(oldDirPath)
            cv2.imwrite(os.path.join(save_path, fileName + fileType), res)
            """


def fold_create(save_path, fold_name):
    """
    建立同结构的存储文件夹
    :param save_path:
    :param fold_name:
    :return:
    """
    fold_path = os.path.join(save_path, fold_name)
    if not Path(fold_path).exists():
        os.mkdir(fold_path)
        print("新建文件夹：" + fold_path)


template = cv2.imread(r"../data/hxq.png", 0)
path = r"D:\dataset\Bubble"
save_path = r"D:\Experiments\res\fuse_line"
batch_process(path, save_path)