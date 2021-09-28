import cv2

from utils.image_process import clahe, threshold_track, filter_choose
from utils.util import *


def detect_hxq(image, file):
    """
    计算环形器的钎透率
    :param image: 图片
    :param file: 文件名字
    :return: 钎透率
    """
    # img1 = cv2.imread(image_path, 0)
    img1 = cv2.cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(r"../data/hxq.png", 0)
    w, h, c = image.shape
    area = w * h
    # 文件信息
    file_name = os.path.splitext(file)[0]  # 文件名
    file_type = os.path.splitext(file)[1]  # 文件扩展名
    # 将环形器分成内部与外部
    img_clahe = clahe(image)  # 自适应均衡化
    gray = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2GRAY)
    images = creat_in_out(gray, template, file)     # 返回单通道
    if images is None:
        return 0

    if caculate_contrast(img1) < 30:
        is_sub = False
    else:
        is_sub = True
    # 里面和外面
    img_inner = images["inner"]
    img_outer = images["outer"]
    # 不同区域选择不同的方法
    inner_area, inner_res1, inner_res2 = flow_path(img_inner, "inner", is_sub)
    outer_area, outer_res1, outer_res2 = flow_path(img_outer, "outer", is_sub)
    # 计算钎透率
    rate = round(float((inner_area + outer_area) / area) * 100, 3)
    compare = image_compare(1, [[image, image, image],
                                [img_inner, inner_res1, inner_res2],
                                [img_outer, outer_res1, outer_res2]])
    # print(str(rate))
    # print(str(caculate_contrast(img1)))
    # cv2.imshow("compare", compare)
    # cv2.waitKey(0)
    # cv2.imwrite(os.path.join(r"D:\compare", str(rate) + "_" + file), compare)
    # return rate
    return inner_area + outer_area


def flow_path(image, type, is_sub):
    """
    算法处理流程
    :param is_sub: 是否除白边
    :param type: 求哪一块的面积
    :param image: 输入图片
    :return: 求出来的真实面积
    """
    blur = filter_choose(image, 1)  # 滤波
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
    if type == "inner":
        threshold, threshold_sub = get_inner_threshold(ret)
        real_area, res1, res2 = get_real_area(blur, threshold, threshold_sub, is_sub)
    else:
        threshold, threshold_sub = get_outer_threshold(ret)
        real_area, res1, res2 = get_real_area(blur, threshold, threshold_sub, is_sub)
    return real_area, res1, res2


def get_inner_threshold(th):
    """
    获取合适的阈值
    :param th: 大津法计算出来的阈值
    :return: 自行处理的阈值
    """
    threshold = th + 70
    threshold_sub = th + 100
    return threshold, threshold_sub


def get_outer_threshold(th):
    """
    获取合适的阈值
    :param th: 大津法计算出来的阈值
    :return: 自行处理的阈值
    """
    threshold = th + 125
    threshold_sub = th + 140
    return threshold, threshold_sub


def get_real_area(gray, threshold1, threshold2, type=True):
    """
    消除阈值分割之后的白边
    :param type: 是否采用相减
    :param gray: 灰度图
    :param threshold1: 阈值1
    :param threshold2: 阈值2
    :return: 真实的面积
    """
    if type:
        ret1, res1 = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY)
        ret2, res2 = cv2.threshold(gray, threshold2, 255, cv2.THRESH_BINARY)

        res11, contours1, heridency1 = cv2.findContours(res1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area1 = caculate_area(contours1)
        res22, contours2, heridency2 = cv2.findContours(res2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area2 = caculate_area(contours2)

        real_area = area1 - area2  # 有些图像由于截出来的不对，使用阈值分割会出现白边，消除白边
        return real_area, res1, res2
    else:
        ret1, res1 = cv2.threshold(gray, threshold1, 255, cv2.THRESH_BINARY)
        res11, contours1, heridency1 = cv2.findContours(res1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        real_area = caculate_area(contours1)
        return real_area, res1, res1


# img_path = r"D:\Download\test_data\hxq\hxq_25.png"
# img = cv2.imread(img_path)
# detect_hxq(img, "1")
