import cv2

from utils.image_process import image_compare


def detect_djb(img):
    img_copy = img.copy()
    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # track(img)
    ret, thres = cv2.threshold(img, 55, 255, cv2.THRESH_BINARY)  # 无关区域
    # 求总体面积
    ret_entire, thres_entire = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    thres_entire = thres_entire[thres_entire == 255]
    area_entire = len(thres_entire)
    # 边缘检测
    edges_all = cv2.Canny(img, 50, 80, apertureSize=3)
    edges_irre = cv2.Canny(thres, 50, 150, apertureSize=3)
    # 轮廓检测
    res_all, contours_all, heridency_all = cv2.findContours(edges_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_all, -1, (0, 0, 255), 1)
    res_irre, contours_irre, heridency_irre = cv2.findContours(edges_irre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_copy, contours_irre, -1, (0, 0, 255), 1)
    # 求面积
    area_all = 0
    for i in contours_all:
        if cv2.contourArea(i) > 0:
            area_all += cv2.contourArea(i)
    area_irre = 0

    for i in contours_irre:
        if cv2.contourArea(i) > 0:
            area_irre += cv2.contourArea(i)
    area_bubble = area_all - area_irre

    rate = str(round(float(area_bubble / area_entire), 2) * 100) + "%"
    # 结果展示
    print("=" * 75)
    print("整体区域的面积" + str(area_all))
    print("无关区域的面积" + str(area_irre))
    print("气泡面积为：" + str(area_bubble))
    print("大基板面积为：" + str(area_entire))
    print("钎透率为：" + rate)
    info = {
        "area_all": area_all,
        "area_irre": area_irre,
        "area_bubble": area_bubble,
        "area_entire": area_entire,
        "rate": rate
    }

    # compare = image_compare(0.8, [img_copy, img])
    # cv2.imshow("compare", compare)
    # cv2.waitKey(0)
    return img_copy, img, info