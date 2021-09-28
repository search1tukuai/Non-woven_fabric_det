import cv2


def detect_gjq(img):
    """
    求共晶区的气泡面积
    :param img:
    :return:
    """
    # 求面积
    height, width, chanel = img.shape

    # 动态求阈值，提高图像对比度
    # img = img[2:height - 1, 2:width - 1]
    img = img[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)]
    contrast = cv2.convertScaleAbs(img, alpha=2, beta=1)

    # 把输入图像灰度化
    gray = cv2.cvtColor(contrast, cv2.COLOR_RGB2GRAY)

    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ravel = contrast.ravel()
    ravel = sorted(ravel)
    mid_value = ravel[int(len(ravel) / 2)]
    if mid_value < 100:
        mid_value += 5
    else:
        mid_value += 10

    ret, threshold = cv2.threshold(gray, mid_value, 255, cv2.THRESH_BINARY)

    # 提取轮廓
    res, contours, heridency = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算轮廓面积
    gjq_area = 0
    for i in contours:
        gjq_area += cv2.contourArea(i)
    # print("钎透率为：" + str(round(float(gjq_area / area) * 100, 3)) + "%")
    return gjq_area
