import cv2
import numpy as np
from utils.util import *


def correct(ori_img):
    # 资源导入
    # img_path = "./data/47.jpg"
    # ori_img = cv2.imread(img_path)

    # 图像处理
    draw_line = False
    remove_horizontal = True
    angle, center, w, h = get_max_theta(ori_img, 150, draw_line, remove_horizontal)  # 获取偏离角度
    res_img = img_rotated(ori_img, angle, center, w, h, remove_horizontal)
    # 显示对比图
    compare = np.hstack([ori_img, res_img])
    cv2.imshow("compare", compare)
    cv2.waitKey(0)
    return res_img
