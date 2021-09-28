# -*- coding = utf-8 -*-
# @Time : 2020/11/27 11:54
# @Author : 天枢
# @File : mysqlUtils.py
# @Software : PyCharm

import pymysql


class MysqlConn():
    """
    数据库连接的公共类，提供连接数据库，查询，删除语句等操作
    """
    def __init__(self, dbName=None):
        self.currentConn = None
        self.host = "localhost"
        self.user = "root"
        self.password = "123456"
        self.dbName = "38"
        self.charset = "utf8"
        self.resultList = []

    def open(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.dbName,
            charset=self.charset,
        )
        self.currentConn = conn  # 数据库连接完成
        self.cursor = self.currentConn.cursor(cursor=pymysql.cursors.DictCursor)  # 游标，用来执行数据库

    # (多条)删除,修改sql
    def cuds(self, sql: str, args, closeConn=True) -> bool:
        self.open()
        try:
            self.cursor.executemany(sql, args)
            self.currentConn.commit()
            return True
        except:
            self.currentConn.rollback()
            return False
        finally:
            self.close()

    # 查询
    def retrieve(self, sql: str, closeConn=True) -> str:
        self.open()
        with self.cursor as my_cursor:
            try:
                my_cursor.execute(sql)  # 执行sql语句
                self.result = my_cursor.fetchall()  # 获取数据
                return self.result
            except:
                self.currentConn.rollback()
                return ""
            finally:
                self.close()

    # (单条)新增,删除,修改
    def cud(self, sql: str, closeConn=True) -> bool:
        self.open()
        with self.cursor as my_cursor:
            try:
                my_cursor.execute(sql)
                self.result = my_cursor.fetchall()
                self.currentConn.commit()
                return True
            except:
                self.currentConn.rollback()
                return False
            finally:
                self.close()

    # 关闭连接
    def close(self):
        if self.cursor:
            self.cursor.close()
        self.currentConn.close()
