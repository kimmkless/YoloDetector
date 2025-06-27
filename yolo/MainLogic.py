import os
import cv2
import PyQt5
from RunDetector import DetectionWorker
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDockWidget
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
from ultralytics import YOLO

dirname = os.path.dirname(PyQt5.__file__)
qt_dir = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_dir

class LogicMixin(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./Assets/UI/DetectorGUI.ui", self)

        self.last_frame = None
        self.model = None
        self.file_path = None
        self.filePath = None
        self.input_type = None
        self.worker = None
        self.is_paused = False
        self.detection_started = False
        self.slider_is_dragging = False

        self.videoProgressSlider.sliderPressed.connect(self.on_slider_pressed)
        self.videoProgressSlider.sliderReleased.connect(self.slider_released)

        self.loadModelBtn_5.clicked.connect(self.load_model)
        self.detectBtn_5.clicked.connect(self.run_detection)
        self.stopBtn.clicked.connect(self.toggle_pause_resume)
        self.forwardBtn.clicked.connect(self.forward_video)
        self.backwardBtn.clicked.connect(self.backward_video)
        self.videoProgressSlider.sliderReleased.connect(self.slider_released)

        self.detectBtn_5.setEnabled(False)
        self.stopBtn.setEnabled(True)  # 启动时就启用

        self.videoButton_5.clicked.connect(lambda: self.select_file("视频"))
        self.pictureButton_5.clicked.connect(lambda: self.select_file("图片"))
        self.cameraButton_5.clicked.connect(lambda: self.select_file("摄像头"))

        self.confSlider_5.setRange(1, 100)
        self.confSlider_5.setValue(50)
        self.confSpin_5.setValue(0.5)
        self.confSlider_5.valueChanged.connect(lambda val: self.confSpin_5.setValue(val / 100.0))
        self.confSpin_5.valueChanged.connect(lambda val: self.confSlider_5.setValue(int(val * 100)))

        self.loUSlider_5.setRange(1, 100)
        self.loUSlider_5.setValue(50)
        self.loUSpinBox_5.setValue(0.5)
        self.loUSlider_5.valueChanged.connect(lambda val: self.loUSpinBox_5.setValue(val / 100.0))
        self.loUSpinBox_5.valueChanged.connect(lambda val: self.loUSlider_5.setValue(int(val * 100)))

        self.delaySlider_5.setRange(0, 100)
        self.delaySlider_5.setValue(10)
        self.delaySpinBox_5.setValue(0.1)
        self.delaySlider_5.valueChanged.connect(lambda val: self.delaySpinBox_5.setValue(val / 100.0))
        self.delaySpinBox_5.valueChanged.connect(lambda val: self.delaySlider_5.setValue(int(val * 100)))

        # --- 找到 UI 中原有控件 ---
        self.settingWidget = self.findChild(QtWidgets.QWidget, "settingWidget")
        self.tabWidget = self.findChild(QtWidgets.QTabWidget, "tabWidget")

        # --- 设置为 Dock ---
        self.settingDock = QtWidgets.QDockWidget("设置面板", self)
        self.settingDock.setWidget(self.settingWidget)
        self.settingDock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.settingDock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)

        self.tabDock = QtWidgets.QDockWidget("信息面板", self)
        self.tabDock.setWidget(self.tabWidget)
        self.tabDock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.tabDock.setFeatures(QtWidgets.QDockWidget.AllDockWidgetFeatures)

        # --- 添加 Dock ---
        self.addDockWidget(Qt.LeftDockWidgetArea, self.settingDock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.tabDock)

        self.settingWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tabWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # 可选：设置初始大小比例
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        new_tab_width = int(self.width() * 0.7)
        new_tab_height = int(self.height() * 1)
        new_setting_width = int(self.width() * 0.3)
        new_setting_height = int(self.height() * 1)
        self.resizeDocks([self.settingDock], [new_setting_width], Qt.Horizontal)
        self.resizeDocks([self.settingDock], [new_setting_height], Qt.Vertical)
        self.resizeDocks([self.tabDock], [new_tab_width], Qt.Horizontal)
        self.resizeDocks([self.tabDock], [new_tab_height], Qt.Vertical)

        # 添加显示/隐藏 dock 的动作
        self.viewMenu.settingDockAction = self.settingDock.toggleViewAction()
        self.viewMenu.tabDockAction = self.tabDock.toggleViewAction()

        self.viewMenu.settingDockAction.setText("· 设置面板")
        self.viewMenu.tabDockAction.setText("· 信息面板")

        self.viewMenu.settingDockAction.triggered.connect(self.toggle_setting_dock)
        self.viewMenu.tabDockAction.triggered.connect(self.toggle_tab_dock)

        self.videoProgressSlider.hide()
        self.stopBtn.hide()
        self.forwardBtn.hide()
        self.backwardBtn.hide()

    def toggle_setting_dock(self):
        if self.settingDock.isVisible():
            self.settingDock.hide()
            self.toggleSettingAction.setText("  设置面板")
        else:
            self.settingDock.show()
            self.toggleSettingAction.setText("· 设置面板")

    def toggle_tab_dock(self):
        if self.tabDock.isVisible():
            self.tabDock.hide()
            self.toggleTabAction.setText("  信息面板")
        else:
            self.tabDock.show()
            self.toggleTabAction.setText("· 信息面板")

    def load_model(self):
        try:
            model_name = "Assets/Model/" + self.modelCombo_5.currentText() + ".pt"
            self.model = YOLO(model_name)
            self.statusbar.showMessage(f"模型加载成功: {model_name}")
        except Exception as e:
            self.statusbar.showMessage(f"错误: {str(e)}")
            self.model = None

    def select_file(self, input_type):
        try:
            self.input_type = input_type
            options = QFileDialog.Options()
            self.stop_current_worker_if_running()
            if input_type == "图片":
                file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "*.jpg *.jpeg *.png *.bmp")
                if file and os.path.exists(file):
                    frame = cv2.imread(file)
                    if frame is not None:
                        self.file_path = file
                        self.filePath = None
                        self.detectBtn_5.setEnabled(True)
                        self.statusbar.showMessage(f"已加载图片: {os.path.basename(file)}")
                        self.display_image(frame)
                        self.videoProgressSlider.hide()
                        self.stopBtn.hide()
                        self.forwardBtn.hide()
                        self.backwardBtn.hide()
            elif input_type == "视频":
                file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "*.mp4 *.avi *.mov *.mkv")
                if file:
                    self.filePath = file
                    self.file_path = None
                    self.detectBtn_5.setEnabled(True)
                    cap = cv2.VideoCapture(file)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        self.display_image(frame)
                        self.statusbar.showMessage(f"已加载视频: {os.path.basename(file)}")
                        self.videoProgressSlider.show()
                        self.stopBtn.show()
                        self.forwardBtn.show()
                        self.backwardBtn.show()
            elif input_type == "摄像头":
                self.filePath = None
                self.file_path = None
                self.detectBtn_5.setEnabled(True)
                self.statusbar.showMessage("准备使用摄像头")
                self.videoProgressSlider.hide()
                self.stopBtn.hide()
                self.forwardBtn.hide()
                self.backwardBtn.hide()
        except Exception as e:
            QMessageBox.critical(self, "文件选择失败", str(e))

    def display_image(self, cv_img):
        try:
            self.last_frame = cv_img.copy()
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.videoLabel.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.statusbar.showMessage(f"显示错误: {str(e)}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_tab_width = int(self.width() * 0.7)
        new_tab_height = int(self.height() * 1)
        new_setting_width = int(self.width() * 0.3)
        new_setting_height = int(self.height() * 1)
        self.resizeDocks([self.settingDock], [new_setting_width], Qt.Horizontal)
        self.resizeDocks([self.settingDock], [new_setting_height], Qt.Vertical)
        self.resizeDocks([self.tabDock], [new_tab_width], Qt.Horizontal)
        self.resizeDocks([self.tabDock], [new_tab_height], Qt.Vertical)
        if self.last_frame is not None:
            self.display_image(self.last_frame)

    def run_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型！")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "提示", "检测已在运行中")
            return

        path = self.file_path if self.input_type == "图片" else self.filePath

        def get_current_params():
            return self.confSpin_5.value(), self.loUSpinBox_5.value(), self.delaySpinBox_5.value()

        self.worker = DetectionWorker(self.model, get_current_params, self.input_type, path)
        self.worker.frame_processed.connect(self.display_image)
        self.worker.result_updated.connect(lambda text: self.resultDisplay.setText(text))
        self.worker.fps_updated.connect(lambda fps: self.FPS.setText(f"FPS: {fps:.2f}"))
        self.worker.progress_updated.connect(self.update_progress_slider)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.start()
        self.detectBtn_5.setEnabled(False)
        self.statusbar.showMessage("检测中...")
        self.is_paused = False
        self.detection_started = True

    def toggle_pause_resume(self):
        if not self.worker or not self.detection_started:
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            self.worker.pause()
            self.stopBtn.setIcon(QIcon("./Assets/Picture/resume.png"))
            self.statusbar.showMessage("检测已暂停")
            self.worker.single_step_mode = True
        else:
            self.worker.resume()
            self.stopBtn.setIcon(QIcon("./Assets/Picture/stop.png"))
            self.statusbar.showMessage("检测继续")
            self.worker.single_step_mode = False

    def on_worker_finished(self):
        self.detectBtn_5.setEnabled(True)
        self.stopBtn.setIcon(QIcon("./Assets/Picture/stop.png"))
        self.statusbar.showMessage("检测结束")
        self.worker = None
        self.is_paused = False
        self.detection_started = False

    def update_progress_slider(self, current, total):
        self.videoProgressSlider.setMaximum(total)
        if not self.slider_is_dragging:
            self.videoProgressSlider.setValue(current)

    def forward_video(self):
        if self.worker:
            self.worker.target_frame_index = self.worker.current_frame_index + 10
            if self.is_paused:
                self.worker.resume()

    def backward_video(self):
        if self.worker:
            self.worker.target_frame_index = max(0, self.worker.current_frame_index - 10)
            if self.is_paused:
                self.worker.resume()

    def on_slider_pressed(self):
        self.slider_is_dragging = True

    def slider_released(self):
        self.slider_is_dragging = False
        if self.worker:
            self.worker.target_frame_index = self.videoProgressSlider.value()
            if self.is_paused:
                self.worker.resume()
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        event.accept()
    def stop_current_worker_if_running(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.is_paused = False
            self.detection_started = False
            self.statusbar.showMessage("已终止当前检测任务")