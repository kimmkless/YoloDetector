import sys
import pymysql

from PyQt5.QtCore import pyqtSignal
from PyQt5 import QtWidgets, uic

class LoginWindow(QtWidgets.QWidget):

    login_success = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/LoginGUI.ui", self)
        self.initUI()
        self.connectDatabase()

    def initUI(self):
        self.loginButton.clicked.connect(self.login)
        self.registerButton.clicked.connect(self.register)
        self.statusLabel.setStyleSheet("color: white")

    def connectDatabase(self):
        self.conn = pymysql.connect(
            host="rm-bp15k902w8a33bs6x9o.mysql.rds.aliyuncs.com",
            user="root",
            password="123456Lyx",  # 修改为你的MySQL密码
            database="user",
            charset="utf8mb4"
        )
        self.cursor = self.conn.cursor()

    def login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text().strip()
        if not username or not password:
            self.statusLabel.setText("请输入用户名和密码")
            return

        sql = "SELECT id, username FROM users WHERE username=%s AND password=%s"
        self.cursor.execute(sql, (username, password))
        result = self.cursor.fetchone()
        if result:
            user_id, username = result
            user_info = {
                'user_id': user_id,
                'username': username
            }
            self.statusLabel.setText("登录成功！")
            self.login_success.emit(user_info)
            self.close()
        else:
            self.statusLabel.setText("用户名或密码错误")

    def register(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text().strip()
        if not username or not password:
            self.statusLabel.setText("用户名和密码不能为空")
            return

        try:
            sql = "INSERT INTO users (username, password) VALUES (%s, %s)"
            self.cursor.execute(sql, (username, password))
            self.conn.commit()
            self.statusLabel.setText("注册成功，请登录")
        except pymysql.err.IntegrityError:
            self.statusLabel.setText("用户名已存在")
