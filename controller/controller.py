from utils.mysqlUtils import MysqlConn
from flask import *


# 获取url
def get_url():
    conn = MysqlConn()
    sql = "select * from algorithm"
    info = conn.retrieve(sql)
    if info:
        return jsonify(info)
    else:
        return jsonify(status="false", msg="查询失败!")