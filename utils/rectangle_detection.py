import cv2
import numpy as np


def lines(img):
    # 直线检测
    img2 = cv2.Canny(img, 20, 250)  # 边缘检测
    line = 4
    minLineLength = 50
    maxLineGap = 150
    # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
    lines = cv2.HoughLinesP(img2, 1, np.pi / 180, 120, lines=line, minLineLength=minLineLength, maxLineGap=maxLineGap)
    lines1 = lines[:, 0, :]  # 降维处理
    # line 函数勾画直线
    # (x1,y1),(x2,y2)坐标位置
    # (0,255,0)设置BGR通道颜色
    # 2 是设置颜色粗浅度
    for x1, y1, x2, y2 in lines1:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img


def contour(img):
    # 检测轮廓
    # ret, thresh = cv2.threshold(cv2.cvtColor(img, 127, 255, cv2.THRESH_BINARY))
    image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv2.minAreaRect(c)  # 生成最小外接矩形
        box_ = cv2.boxPoints(rect)
        h = abs(box_[3, 1] - box_[1, 1])
        w = abs(box_[3, 0] - box_[1, 0])
        print("宽，高", w, h)
        # 只保留需要的轮廓
        if h > 3000 or w > 2200:
            continue
        if h < 2500 or w < 1500:
            continue
        box = cv2.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        angle = rect[2]  # 获取矩形相对于水平面的角度
        if angle > 0:
            if abs(angle) > 45:
                angle = 90 - abs(angle)
        else:
            if abs(angle) > 45:
                angle = (90 - abs(angle))
        # 绘制矩形
        # cv2.drawContours(img, [box], 0, (255, 0, 255), 3)
    print("轮廓数量", len(contours))
    return img, box, angle


def rotate(image, angle):
    # 旋转图片
    (h, w) = image.shape[:2]  # 获得图片高，宽
    center = (w // 2, h // 2)  # 获得图片中心点
    img_ratete = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, img_ratete, (w, h))
    return rotated


img_path = "../data/output/temp/out.jpg"
img = cv2.imread(img_path)
mediu = cv2.medianBlur(img, 19)  # 中值滤波,过滤除最外层框线以外的线条
img_lines = lines(mediu)  # 直线检测，补充矩形框线
img_contours, box, angle = contour(img_lines)  # 轮廓检测，获取最外层矩形框的偏转角度
print("角度", angle, "坐标", box)
img_rotate = rotate(img_lines, angle)  # 旋转图像至水平方向
img_contours, box, _ = contour(img_rotate)
cv2.imshow("res", img_contours)
cv2.waitKey(0)