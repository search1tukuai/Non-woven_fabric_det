import os

import cv2
from interface.detect_qp import run_seg
from utils.util import resize_img


def image_seg(info, save_dir):
    img_ori = cv2.imread(info['path'])
    xjbs_location = info['location']['xjb']
    hxqs_location = info['location']['hxq']
    djbs_location = info['location']['djb']
    # 图片裁剪
    i = 1
    for location in xjbs_location:
        name = "xjb_" + str(i) + ".jpg"
        res = resize_img(img_ori, location)
        cv2.imwrite(os.path.join(save_dir, name), res)
        i += 1

    i = 1
    for location in hxqs_location:
        name = "hxq_" + str(i) + ".jpg"
        res = resize_img(img_ori, location)
        cv2.imwrite(os.path.join(save_dir, name), res)
        i += 1

    i = 1
    for location in djbs_location:
        name = "djb_" + str(i) + ".jpg"
        res = resize_img(img_ori, location)
        cv2.imwrite(os.path.join(save_dir, name), res)
        i += 1
