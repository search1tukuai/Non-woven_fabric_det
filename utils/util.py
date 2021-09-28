# -*- coding = utf-8 -*-
# @Time : 2021/7/1 11:07
# @Author : 天枢
# @File : util.py
# @Software : PyCharm
import os

import cv2
import numpy as np
import math

from matplotlib import pyplot as plt

from utils.image_process import image_compare


def get_max_theta(image, line_len, draw_line=False, remove_horizontal=True):
    """
    求直线的最大偏离角度
    :param remove_horizontal: 是否移除横线
    :param draw_line: 是否画出检测到的直线
    :param image: 输入图像
    :param line_len: 检测直线的最大距离
    :return:
    """
    h, w = image.shape[:2]
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # canny边缘检测
    # ret, edges = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow("edges", edges)
    # 霍夫直线检测
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_len)
    if lines is None:
        return None, 0, 0, 0, 0
    # 求直线的起点与终点
    lines_xy = []
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho  # 代表x = r * cos（theta）
        y0 = b * rho  # 代表y = r * sin（theta）
        # 计算直角坐标(注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长)
        l = 800
        x1 = int(x0 + l * (-b))  # 计算直线起点横坐标
        y1 = int(y0 + l * a)  # 计算起始起点纵坐标
        x2 = int(x0 - l * (-b))  # 计算直线终点横坐标
        y2 = int(y0 - l * a)  # 计算直线终点纵坐标
        lines_xy.append([x1, y1, x2, y2])
        # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # 去除横线或垂直线
    lines_xy = remove_lines(lines_xy, remove_horizontal)
    # 画直线
    if draw_line:
        for i in range(len(lines_xy)):
            cv2.line(image, (lines_xy[i][0], lines_xy[i][1]), (lines_xy[i][2], lines_xy[i][3]), (0, 0, 255), 1)
    # 求垂直线的最大偏离
    max_theta = caculate_difference3(lines_xy)
    if max_theta is None:
        return None, 0, 0, 0, 0
    print("偏差值：" + str(max_theta))
    angle = math.atan(max_theta)
    print("弧度偏差为：" + str(angle) + "°")
    # angle = angle * (180 / np.pi)
    # angle = (angle - 90) / (w / h)
    center = (w // 2, h // 2)
    return angle, max_theta, center, w, h


# 旋转图像
def img_rotated(image, angle, max_theta, center, w, h, orientation=True):
    if orientation:
        # 以竖线为判断依据
        print("以竖线为判断依据进行旋转")
        if angle < 0:
            if max_theta > -50:
                angle *= -2
            if -50 >= max_theta > -60:
                angle *= -1
        else:
            if max_theta < 20:
                angle *= -1
            if 20 <= max_theta < 29:
                angle *= -1
            if 29 <= max_theta:
                angle *= -0.2
    else:
        # 以横线为判断依据
        print("以横线为判断依据进行旋转")
        if angle < 0:
            angle *= -0.0001
        else:
            angle += 0.5
    print("处理后的angle:" + str(angle))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # (旋转中心,旋转角度,图像缩放因子)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)  # 仿射变换(输入图像,变换矩阵,输出图像大小)
    return rotated


# 移除直线
def remove_lines(lines_xy, remove_horizontal=True):
    # 默认去除横线
    unqualified = []
    cnt = 0
    if remove_horizontal:
        # 去除横线
        for i in range(len(lines_xy)):
            if abs(lines_xy[i][2] - lines_xy[i][0]) > 100 or lines_xy[i][2] - lines_xy[i][0] == 0:
                unqualified.append(i)  # 记录横线坐标
        for j in unqualified:
            del lines_xy[j - cnt]
            cnt += 1
    else:
        # 去除竖线
        print("去除竖线")
        for i in range(len(lines_xy)):
            if abs(lines_xy[i][3] - lines_xy[i][1]) > 100 or lines_xy[i][2] - lines_xy[i][0] == 0:
                unqualified.append(i)  # 记录竖线坐标
        for j in unqualified:
            del lines_xy[j - cnt]
            cnt += 1
    return lines_xy


# 计算偏离方法3-中位数
def caculate_difference3(lines_xy):
    length = len(lines_xy)
    mid = int(length/2)
    data = []
    for i in range(len(lines_xy)):
        x1, y1, x2, y2 = lines_xy[i]
        theta = (y2 - y1) / (x2 - x1)
        data.append(theta)
    data.sort()
    if len(lines_xy) == 0:
        return None
    else:
        return data[mid]


# 计算偏离方法1
def caculate_difference1(lines_xy):
    max_theta = 0
    min_theta = 0
    count = 0
    for i in range(len(lines_xy)):
        x1, y1, x2, y2 = lines_xy[i]
        theta = (y2 - y1) / (x2 - x1)
        if theta > 0:
            count += 1
        else:
            count -= 1
        max_theta = max(max_theta, theta)
        print("max_theta:" + str(max_theta))
        min_theta = min(min_theta, theta)
        print("min_theta:" + str(min_theta))
        print("*" * 50)
    if count > 0:
        return max_theta
    else:
        return min_theta


# 计算偏离方法2
def caculate_difference2(lines_xy):
    posi_count = 0
    posi_sum = 0
    nega_count = 0
    nega_sum = 0
    for i in range(len(lines_xy)):
        x1, y1, x2, y2 = lines_xy[i]
        theta = (y2 - y1) / (x2 - x1)
        if theta > 0:
            posi_sum += theta
            posi_count += 1
        else:
            nega_sum += theta
            nega_count += 1
    # 计算偏离角度(可优化)
    if posi_sum / posi_count > nega_sum / nega_count:
        res = posi_sum / posi_count
    else:
        res = nega_sum / nega_count
    return res


def creat_in_out(img, template, name, store_path="", FLAG=False):
    """
    对环形器内部的不同区域分为内部和外部区域
    :param img: 原始灰度图
    :param FLAG: 是否存储
    :param template: 模板
    :param store_path: 指定存储路径
    :param name: 图片的名字
    :return: 返回内部与外部图片
    """
    # img = cv2.imread(img_path, 0)
    img2 = img.copy()
    img = img2.copy()
    w, h = template.shape[::-1]
    # 模板匹配,选择合适的模板
    # cv2.imshow("a", template)
    # cv2.waitKey(0)
    if img.shape[0] < template.shape[0] or img.shape[1] < template.shape[1]:
        return None
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    if res is None:
        return None
    # 寻找最值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 使用不同的比较方法，对结果的解释不同
    if 'cv2.TM_CCOEFF' in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 里面的部分
    inner = img[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]]
    # 外面的部分
    outer = cv2.rectangle(img2, top_left, bottom_right, 0, -1)
    # 是否存储
    if FLAG:
        inner_file = os.path.join(store_path, name + "_inner.png")
        cv2.imwrite(inner_file, inner)
        outer_file = os.path.join(store_path, name + "_outer.png")
        cv2.imwrite(outer_file, outer)
    # 返回结果图像
    # img_and = image_join(1, [inner, outer])
    # cv2.imshow("and", inner)
    # cv2.waitKey(0)
    images = {
        "inner": inner,
        "outer": outer
    }
    return images


def resize_img(img, location):
    """
    图像裁剪
    :param img: 原图
    :param location: 截取的边界点
    :return: 裁剪后的图像
    """
    x1 = location[0][0]
    y1 = location[0][1]
    x2 = location[1][0]
    y2 = location[1][1]
    img_res = img[y1: y2, x1: x2]
    return img_res


def get_files(file_path):
    """
    根据文件夹路径获取文件夹下的内容
    :param file_path: 文件夹路径
    :return: 文件路径
    """
    files_path = []
    files_name = []
    for root, dirs, files in os.walk(file_path):
        files_name = files
        for file in files:
            files_path.append(root + "/" + file)
    return files_name, files_path


def get_dirs(dir_path):
    """
    获取当前路径下的目录
    :param dir_path:
    :return:
    """
    dirs_path = []
    for root, dirs, files in os.walk(dir_path):
        for dir in dirs:
            dirs_path.append(dir)
    return dirs_path


def get_cur_files(file_path):
    """
    获取文件夹下的文件
    :param file_path:
    :return:
    """
    files_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            files_list.append(file)
    return files_list


def print_img_size(name, width, high):
    print(name + "的大小为：" + str(width) + "x" + str(high))


def save_img(folder, filename, img):
    save_path = "data/output/"
    cv2.imwrite(save_path + "/" + folder + filename, img)


def template_xjb_add():
    """
    添加小基板模板
    :return: 模板数组
    """
    imgs = []
    path = "data/template/xiaojiban"
    files = os.listdir(path)
    for file in files:
        # 判断是否是文件夹，不是文件夹才打开
        if not os.path.isdir(file):
            img = cv2.imread(path + "/" + file, 0)
            imgs.append(img)
    return imgs


def template_add():
    """
    读取模板数据
    :return: 返回模板
    """
    template_centre_path = "./data/template/centre/centre1.jpg"
    template_centre = cv2.imread(template_centre_path, 0)  # 中间板
    template_huanxingqi_path = "./data/template/huanxingqi/h1.jpg"
    template_huanxingqi = cv2.imread(template_huanxingqi_path, 0)  # 环形器
    templates = {'centre': template_centre,
                 'hxq': template_huanxingqi,
                 'xjb': template_xjb_add()}
    return templates


def target_add():
    """
    读取要检测的文件夹
    :return: 返回数组
    """
    target = []
    path = "data/target/entire"
    files = os.listdir(path)
    for file in files:
        # 判断是否是文件夹，不是文件夹才打开
        if not os.path.isdir(file):
            file_info = {'name': file,
                         'img': cv2.imread(path + "/" + file)}
            target.append(file_info)
    return target


def hist_show(img, start=0, end=256):
    """
    直方图展示
    :param img:
    :param start:
    :param end:
    :return:
    """
    ravel = img.ravel()
    # ravel = sorted(ravel)
    # mid_value = ravel[int(len(ravel) / 2)]
    # 绘制直方图
    plt.hist(ravel, 256, [start, end])
    plt.show()


def hist_line_show(img, start=0, end=256):
    """
    直方图拟形
    :param img:
    :param start:
    :param end:
    :return:
    """
    b, g, r = cv2.split(img)
    hists, bins = np.histogram(b.flatten(), 256, [start, end])
    # num_peak = signal.find_peaks(hists, distance=50)    # 获取波峰
    # sorted(num_peak[0])
    # for ii in range(len(num_peak[0])):
    #     plt.plot(num_peak[0][ii], hists[num_peak[0][ii]], '*', markersize=10)
    plt.plot(hists, color='r')
    plt.show()


def caculate_area(contours):
    """
    计算轮廓中的面积
    :param contours: 轮廓
    :return: 轮廓面积
    """
    area = 0
    for i in contours:
        area += cv2.contourArea(i)
    return area


def caculate_contrast(img):
    """
    计算图片的对比度
    :param img: 输入图片
    :return: 结果
    """
    ravel = img.ravel()
    rate = round(float(len(ravel[ravel < 40]) / len(ravel)), 2) * 100
    return rate


def caculate_rate(a, b, n):
    """
    钎透率计算
    :param a:分子
    :param b:分母
    :param n: 保留几位小数
    :return: 计算结果
    """
    if b == 0:
        res = 0
    else:
        res = round((float(a / b)) * 100, n)
    return res


def str_joint(arrs, type):
    """
    金丝检测中的焊点和芯片信息拼接
    :param arrs:
    :param type:
    :return:
    """
    if len(arrs) == 0:
        return ""
    res = type + ":" + "\r\n"
    for arr in arrs:
        res += str(arr)
        res += "\r\n"
    return res
