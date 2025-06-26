import os
import cv2
import traceback
import PyQt5
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO


dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir


class LogicMixin:
    def __init__(self):
        uic.loadUi("./Assets/UI/GUI.ui", self)
        self.model = None
        self.file_path = None
        self.filePath = None
        self.is_detecting = False
        self.input_type = None

        # 信号绑定：新的命名规则
        self.loadModelBtn_5.clicked.connect(self.load_model)
        self.detectBtn_5.clicked.connect(self.run_detection)

        self.videoButton_5.clicked.connect(lambda: self.select_file("视频"))
        self.pictureButton_5.clicked.connect(lambda: self.select_file("图片"))
        self.cameraButton_5.clicked.connect(lambda: self.select_file("摄像头"))

        self.confSlider_5.setRange(1, 100)
        self.confSlider_5.setValue(50)
        self.confSpin_5.setValue(0.5)

        self.confSlider_5.valueChanged.connect(
            lambda val: self.confSpin_5.setValue(val / 100.0)
        )
        self.confSpin_5.valueChanged.connect(
            lambda val: self.confSlider_5.setValue(int(val * 100))
        )

    def load_model(self):
        try:
            model_name = self.modelCombo_5.currentText() + ".pt"
            self.model = YOLO(model_name)
            self.statusbar.showMessage(f"模型加载成功: {model_name}")
            self.detectBtn_5.setEnabled(True)
        except Exception as e:
            self.statusbar.showMessage(f"错误: {str(e)}")
            self.model = None
            self.detectBtn_5.setEnabled(False)

    def select_file(self, input_type):
        try:
            self.input_type = input_type
            options = QFileDialog.Options()

            if input_type == "图片":
                file, _ = QFileDialog.getOpenFileName(None, "选择图片", "",
                                                      "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)",
                                                      options=options)
                if file and os.path.exists(file):
                    frame = cv2.imread(file)
                    if frame is not None:
                        self.file_path = file
                        self.filePath = None
                        self.display_image(frame)
                        self.statusbar.showMessage(f"已加载图片: {os.path.basename(file)}")
                        self.detectBtn_5.setEnabled(True)
                    else:
                        QMessageBox.critical(None, "错误", "无法读取图片文件！")

            elif input_type == "视频":
                file, _ = QFileDialog.getOpenFileName(None, "选择视频", "",
                                                      "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
                                                      options=options)
                if file:
                    self.filePath = file
                    self.file_path = None
                    self.statusbar.showMessage(f"已加载视频: {os.path.basename(file)}")
                    self.detectBtn_5.setEnabled(True)
                    cap = cv2.VideoCapture(file)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        self.display_image(frame)
                    else:
                        QMessageBox.critical(None, "错误", "无法读取视频第一帧！")

            elif input_type == "摄像头":
                self.filePath = None
                self.file_path = None
                self.statusbar.showMessage("准备使用摄像头")
                self.detectBtn_5.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(None, "文件选择失败", str(e))
            print(traceback.format_exc())

    def display_image(self, cv_img):
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.videoLabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.statusbar.showMessage(f"显示错误: {str(e)}")

    def run_detection(self):
        try:
            if self.model is None:
                QMessageBox.warning(None, "模型未加载", "请先加载模型！")
                return

            conf_threshold = self.confSpin_5.value()
            input_type = self.input_type

            if input_type == "摄像头":
                self.statusbar.showMessage("打开摄像头中...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    QMessageBox.critical(None, "错误", "无法打开摄像头！")
                    return

                self.statusbar.showMessage("摄像头检测进行中... 按 Ctrl+C 或关闭窗口以终止")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = self.model(frame, conf=conf_threshold)[0]
                    result_frame = results.plot()
                    self.display_image(result_frame)
                    QApplication.processEvents()
                cap.release()
                self.statusbar.showMessage("摄像头检测结束")

            elif input_type == "图片":
                if not self.file_path:
                    QMessageBox.warning(None, "未选择文件", "请选择图片文件！")
                    return
                results = self.model(self.file_path, conf=conf_threshold)[0]
                result_img = results.plot()
                self.display_image(result_img)
                self.statusbar.showMessage("图片检测完成")

            elif input_type == "视频":
                if not self.filePath:
                    QMessageBox.warning(None, "未选择视频", "请选择视频文件！")
                    return
                cap = cv2.VideoCapture(self.filePath)
                if not cap.isOpened():
                    QMessageBox.critical(None, "错误", "无法打开视频文件！")
                    return

                self.statusbar.showMessage("开始处理视频...")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = self.model(frame, conf=conf_threshold)[0]
                    result_frame = results.plot()
                    self.display_image(result_frame)
                    QApplication.processEvents()
                cap.release()
                self.statusbar.showMessage("视频检测完成")

        except Exception as e:
            QMessageBox.critical(None, "检测失败", str(e))
