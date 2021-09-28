# -*- coding = utf-8 -*-

import flask
from controller.controller import *
import os,sys
sys.path.append(r"D:\PycharmProjects\detection")

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


# 获取算法url
@app.route("/get_url", methods=["GET"])
def get_all_url():
    data = get_url()
    return data


app.run(host="0.0.0.0", port="5001")
