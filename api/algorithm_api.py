# -*- coding = utf-8 -*-
# @Time : 2020/11/1 20:24
# @Author : 李傲杰
# @File : algorithmAPI.py
# @Software : PyCharm
import os
import re
#
# import airport as airport
from flask import Flask, request, jsonify

from interface.detect_js import detect_js
from interface.detect_qp import *
from interface.get_opt_seg import *
from utils.util import caculate_rate

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

opt_fabric = get_fabric_opt()
device = select_device(opt_fabric.device)
print("加载无纺布缺陷检测模型")
js_model = attempt_load("../algorithm/region/model_data/best112531.pt", map_location=device)
print("无纺布缺陷检测模型加载完成")

# 无纺布缺陷检测检测接口
@app.route("/jinsi", methods=["POST"])
def jinsi():
    data = request.get_json()
    print(data)
    file_name = data.get("filenames")
    ori_file_path = data.get("filepath") + "/"
    save_path = data.get("savepath") + "/"
    # 1、数据拼接
    file_path = os.path.join(ori_file_path + file_name)
    save_name = str(round(time.time() * 1000)) + "_" + file_name
    # 2、图像识别
    time_start = time.time()
    img_res, info_res = detect_js(opt_fabric, js_model, file_path, save_path)
    # os.remove(ori_file_path + file_name)
    # 3、图像保存
    cv2.imwrite(os.path.join(save_path, save_name), img_res)
    # 4、返回json
    # info = {
    #     "resname": save_name,
    #     "info": info_res
    # }
    info = {
        "resname": save_name,
    }
    print(info)
    print(f'检测计算花费时间：{time.time() - time_start:.3f}s')
    return jsonify(fileInfo=info)


app.run(host="127.0.0.1", port="5000")
