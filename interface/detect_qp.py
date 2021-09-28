import argparse
import time
from pathlib import Path

import cv2
import torch

from algorithm.region.models.experimental import attempt_load
from algorithm.region.utils.cut import get_file_name
from algorithm.region.utils.datasets import LoadImages
from algorithm.region.utils.general import check_img_size, non_max_suppression, apply_classifier, increment_path, \
    scale_coords, xyxy2xywh, save_one_box
from algorithm.region.utils.plots import plot_one_box, colors
from algorithm.region.utils.torch_utils import select_device, load_classifier, time_synchronized
from interface.detect_gjq import detect_gjq
from interface.detect_hxq import detect_hxq
from utils.util import resize_img


def run_seg(opt, model, img_path, img_save_path, img_save_name):
    """
    图像分割运行主函数
    :return:
    """
    global xjb_area, hxq_area, djb_area, file_name, vid_writer, vid_path
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
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # 检测每一张图片
            p,im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # 测试集文件地址
            # save_path = save_dir + "/" + p.name  # 文件保存地址
            save_path = save_dir + "/" + img_save_name  # 文件保存地址
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                file_name = get_file_name(path)     # [文件名+后缀]
                xjbs = []       # 小基板
                hxqs = []       # 环形器
                djbs = []       # 大基板
                gjqs = []       # 共晶去
                xjb_area = 0    # 小基板面积
                hxq_area = 0    # 环形器面积
                djb_area = 0    # 大基板面积
                gjq_area = 0    # 共晶区面积
                for *xyxy, conf, cls in reversed(det):  # xyxy是坐标点，(x1, y1),(x2, y2)代表左上和右下两个点
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))     # 左上和右下坐标点
                    if save_img or opt.save_crop or view_img:  # 加框
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')  # 标签 置信度
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)  # 画框和文字
                        """
                        在这个位置开始写气泡检测
                        """
                        # 整理检测出来的区域
                        if c == 0:
                            xjbs.append([c1, c2])
                            xjb_area += (c2[0] - c1[0]) * (c2[1] - c1[1])
                        elif c == 1:
                            hxqs.append([c1, c2])
                            hxq_area += (c2[0] - c1[0]) * (c2[1] - c1[1])
                        elif c == 2:
                            djbs.append([c1, c2])
                            djb_area += (c2[0] - c1[0]) * (c2[1] - c1[1])
                        elif c == 3:
                            gjqs.append([c1, c2])
                            gjq_area += (c2[0] - c1[0]) * (c2[1] - c1[1])
                        """
                        结束
                        """
                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            print("\n")
            print("检测到小基板：" + str(len(xjbs)) + "个")
            print("检测到环形器：" + str(len(hxqs)) + "个")
            print("检测到大基板：" + str(len(djbs)) + "个")
            print("检测到共晶区：" + str(len(gjqs)) + "个")
            print(f'检测花费时间：{t2 - t1:.3f}s')

            # 是否展示结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            # 是否保存结果图
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    dict_res = {
        'name': file_name,
        'path': img_path,
        'location': {
            'xjb': xjbs,
            'hxq': hxqs,
            'djb': djbs,
            'gjq': gjqs
        },
        'area': {
            'xjb': xjb_area,
            'hxq': hxq_area,
            'djb': djb_area,
            'gjq': gjq_area
        }
    }
    print(f'区域检测花费时间：{time.time() - t_start:.3f}s')
    return dict_res


def bubble_detect(img_path, info):
    img = cv2.imread(img_path)
    xjbs_location = info['location']['xjb']
    hxqs_location = info['location']['hxq']
    djbs_location = info['location']['djb']
    gjqs_location = info['location']['gjq']
    hxqs_bubble_area = 0    # 环形器的气泡面积
    gjqs_bubble_area = 0    # 共晶区的气泡面积
    xjbs_bubble_area = 0    # 小基板的气泡面积
    djbs_bubble_area = 0    # 大基板的气泡面积

    # 求共晶区的气泡面积
    for location in gjqs_location:
        img_target = resize_img(img, location)
        res = detect_gjq(img_target)
        gjqs_bubble_area += res

    # 求环形器的气泡面积
    for location in hxqs_location:
        img_target = resize_img(img, location)
        res = detect_hxq(img_target, info["name"])
        hxqs_bubble_area += res

    dict_res = {
        'gjq': gjqs_bubble_area,
        'hxq': hxqs_bubble_area,
        'djb': djbs_bubble_area,
        'xjb': xjbs_bubble_area,
    }
    return dict_res