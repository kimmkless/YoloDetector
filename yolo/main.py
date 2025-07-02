import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from TrainMainWindow import YoloTrainerApp
from MainLogic import LogicMixin
from Validator import ValidatorApp
from DatasetAnnotator import DatasetAnnotator
from Login import LoginWindow

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/MainGUI.ui", self)
        db_config = {
            "host": "localhost",
            "user": "root",
            "password": "your_password",
            "database": "user_system",
            "charset": "utf8mb4"
        }

        self.trainButton.clicked.connect(self.open_train)
        self.detectButton.clicked.connect(self.open_detect)
        self.labelButton.clicked.connect(self.open_label)
        self.validButton.clicked.connect(self.open_valid)
        self.loginButton.clicked.connect(self.open_login)

        self.loginLabel.hide()  # 初始隐藏

        self.train_window = None
        self.detect_window = None
        self.label_window = None
        self.valid_window = None
        self.file_window = None
        self.login_window = None
        self.userId = None

    def open_login(self):
        if self.login_window is None:
            self.login_window = LoginWindow()
            self.login_window.login_success.connect(self.on_login_success)
        self.login_window.show()

    def open_train(self):
        if self.train_window is None:
            self.train_window = YoloTrainerApp(self.userId)
        self.train_window.show()

    def open_detect(self):
        if self.detect_window is None:
            self.detect_window = LogicMixin(self.userId)
        self.detect_window.show()

    def open_label(self):
        if self.label_window is None:
            self.label_window = DatasetAnnotator()
        self.label_window.show()

    def open_valid(self):
        if self.valid_window is None:
            self.valid_window = ValidatorApp()
        self.valid_window.show()
    def on_login_success(self, user_info):
        self.loginButton.setEnabled(False)
        self.loginButton.hide()

        self.userId = user_info["user_id"]
        self.loginLabel.setText(f"欢迎，{user_info['username']}（ID: {user_info['user_id']}）")
        self.loginLabel.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainController()
    window.show()
    sys.exit(app.exec_())
