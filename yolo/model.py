from ultralytics import YOLO
import sys,os,PyQt5
from PyQt5.QtWidgets import QMainWindow, QApplication
from GUI import Ui_MainWindow  # 导入生成的界面类
dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  # 自动创建所有界面元素

        # 访问UI中的控件
        self.ui.detectBtn.clicked.connect(self.start_detection)

    def start_detection(self):
        print("检测按钮被点击")


if __name__ == "__main__":
    app = QApplication(sys.argv)
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