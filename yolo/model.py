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

# 加载预训练模型
#model = YOLO('yolov8n.pt')
#model("liuwei.gif",show=True,save=True)
# 训练配置
#results = model.train(
#    data='dataset.yaml',  # 数据集配置文件
#    epochs=500,           # 训练轮次
#    batch=32,             # 每轮批量
#    imgsz=640,            # 图片尺寸
#    lr0=0.01,             # 学习率
#    device='0'            # 使用GPU
#)
