import argparse
import time
from pathlib import Path

import cv2
import torch

from algorithm.region.utils.cut import get_file_name
from algorithm.region.utils.datasets import LoadImages
from algorithm.region.utils.general import check_img_size, non_max_suppression, apply_classifier, increment_path, \
    scale_coords, xyxy2xywh, save_one_box
from algorithm.region.utils.plots import plot_one_box, colors
from algorithm.region.utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.util import str_joint


def detect_js(opt, model, img_path, img_save_path):
    """

    :param opt: 参数
    :param model: 模型
    :param img_path: 图片路径
    :param img_save_path: 图片存储路径
    :return: 处理后的图片，有缺陷（True），排灯位置（从左至右）
    """
    global im0
    defect = False #缺陷标志
     #排灯位置
    l1, l2, l3 = 0, 0, 0
    flag1, flag2, flag3 = True, True, True
    t_start = time.time()

    # opt = load_parameter(img_path)
    source = img_path
    # weights = opt.weights
    view_img, save_txt, imgsz = opt.view_img, opt.save_txt, opt.img_size
    save_img = True
    save_dir = img_save_path        # 目录
    # 运行设备
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'
    # 加载模型
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)

        # 结果处理
        for i, det in enumerate(pred):  # 检测每一张图片
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # 测试集文件地址
            save_path = save_dir + "/" + p.name  # 文件保存地址
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                file_name = get_file_name(path)     # [文件名+后缀]
                for *xyxy, conf, cls in reversed(det):  # xyxy是坐标点，(x1, y1),(x2, y2)代表左上和右下两个点
                    # location = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]     # 左上和右下坐标点
                    if flag1:
                        if int(xyxy[0]) <= int(im0.shape[1] / 3):
                            # print(int(xyxy[1]), int(xyxy[3]), im0.shape)
                            l1 = 1
                            flag1 = False
                    if flag2:
                        if (int(im0.shape[1] / 3) <= int(xyxy[2]) or int(xyxy[0]) <= int(2 * im0.shape[1] / 3)) or (
                                int(xyxy[0]) < int(im0.shape[1] / 3) and int(xyxy[2]) > int(2 * im0.shape[1] / 3)):
                            # print(int(xyxy[0]), int(xyxy[2]), im0.shape)
                            l2 = 1
                            flag2 = False
                    if flag3:
                        if int(xyxy[2]) >= int(2 * im0.shape[1] / 3):
                            # print(int(xyxy[0]), int(xyxy[2]), im0.shape)
                            l3 = 1
                            flag3 = False
                    if save_img or opt.save_crop or view_img:  # 加框
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')  # 标签 置信度
                        if names[c] == "burrs" or "swrinkles" or "hole" or "shadow" or "fold" or "reticulated" or "spot" or "oily":
                            defect = True
                        else:
                            defect = False
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)  # 画框和文字
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 是否展示结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
    # dict_info = {
    #     "Ball": str_joint(ball, "ball"),
    #     "Chip": str_joint(chip, "chip")
    # }
        print(f'目标检测花费时间：{time.time() - t_start:.3f}s')
    #dict_info = str_joint(ball, "ball") + str_joint(chip, "chip")
    light_id = [l1, l2, l3]
    dict_info = [defect, light_id]

    return im0, dict_info