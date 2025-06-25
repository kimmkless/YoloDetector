import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from GUI import Ui_MainWindow  # 确保 gui.py 文件与 main.py 在同一目录

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # 初始化 UI 界面

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
