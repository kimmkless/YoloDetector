import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow

# 假设你的训练模块和检测模块是两个QWidget类
from TrainMainWindow import YoloTrainerApp
from MainLogic import LogicMixin

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/MainGUI.ui", self)

        self.trainButton.clicked.connect(self.open_train)
        self.detectButton.clicked.connect(self.open_detect)

        self.train_window = None
        self.detect_window = None

    def open_train(self):
        if self.train_window is None:
            self.train_window = YoloTrainerApp()
        self.train_window.show()

    def open_detect(self):
        if self.detect_window is None:
            self.detect_window = LogicMixin()
        self.detect_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainController()
    window.show()
    sys.exit(app.exec_())
