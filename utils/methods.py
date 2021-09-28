import cv2 as cv
import numpy as np

from utils.util import print_img_size


def template_match(templates, target_img, threshold):
    """
    使用opencv自带api进行检测
    :param threshold: 阈值
    :param templates: 模板图
    :param target_img: 目标检测
    :return: 检测结果图
    """
    target_img_gray = to_binary(target_img, 80)
    # target_img_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    for template in templates:
        w, h = template.shape[::-1]
        res = cv.matchTemplate(target_img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            # 对应参数image, start_point, end_point, color, thickness
            # pt是左上点，(pt[0] + w, pt[1] + h)是右下点
            cv.rectangle(target_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    return target_img


def left_cut(template, target_img, threshold):
    """
    将完整的图像中的左边部分裁剪出来
    :param template: 模板
    :param target_img: 原始图像
    :param threshold: 阈值
    :return: 裁剪后的图像
    """
    c, w, h = target_img.shape[::-1]
    print_img_size("原图", w, h)
    t_w, t_h = template.shape[::-1]
    img_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)  # 拿到匹配位置的左上点
    try:
        left = []
        for pt in zip(*loc[::-1]):
            left.append(pt[0])  # 添加起始坐标点
        left.sort()
        right = left[int(len(left) / 2)]
        img_cut = target_img[0:h, 0:right]  # 截止坐标取匹配得到的中间值
        print_img_size("left图片", right, h)
        return img_cut
    except:
        print("未匹配到中间的白条")
        return


def left_top_cut(template, target_img, threshold):
    """
    将完整的图像中的左边四个小基板裁剪出来
    :param template: 模板
    :param target_img: 原始图像
    :param threshold: 阈值
    :return: 裁剪后的图像
    """
    try:
        c, w, h = target_img.shape[::-1]
        t_w, t_h = template.shape[::-1]
        img_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        top = []
        for pt in zip(*loc[::-1]):
            top.append(pt[1])
        top.sort()
        bottom = top[0] + int(t_h / 4)  # 一定是取最上边的那个点
        img_cut = target_img[0:bottom, :]
        print_img_size("left_top图片", w, bottom)
        return img_cut
    except:
        print("当前未匹配到环形器，无法完成切割！")
        return


def to_binary(img, thresh):
    """
    转换为二值图像
    :param img:
    :param thresh:
    :return:
    """
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # template_grey = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_binary = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)[1]
    return img_binary