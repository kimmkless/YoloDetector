import os
import subprocess
import threading
import shutil
import matplotlib.pyplot as plt

from DataEnhanced import EnhancementThread
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from PlotCanvas import PlotCanvas
from ReadData import get_latest_results_csv, get_latest_model_dir, read_loss_from_results_csv, \
    read_metrics_from_results_csv
from ConvertDataset import convert_voc_dataset, convert_coco_dataset
from Login import LoginWindow
from CloudFile import FileManagerApp

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

LOG_FILE = "train_output.log"
SAVE_MODEL_DIR = "saved_models"


class YoloTrainerApp(QtWidgets.QMainWindow):
    def __init__(self, user_id=None):
        super().__init__()
        uic.loadUi("Assets/UI/TrainGUI.ui", self)

        self.userId = user_id
        self.file_window = None
        self.login_window = None

        self.central_placeholder = QtWidgets.QWidget()
        self.setCentralWidget(self.central_placeholder)
        # 创建三个 dock 窗口
        self.create_dock_windows()
        self.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks | QtWidgets.QMainWindow.AllowNestedDocks | QtWidgets.QMainWindow.AllowTabbedDocks)

        # 连接信号槽
        self.btnSelectDataset.clicked.connect(self.select_dataset_folder)
        self.btnUploadDataset.clicked.connect(self.upload_dataset)
        self.btnStartTraining.clicked.connect(self.start_data_enhancement)
        self.plotSelect.currentIndexChanged.connect(self.update_selected_plot)
        self.btnStopTraining.clicked.connect(self.stop_training)
        self.smallFontAction.triggered.connect(lambda: self.set_global_font_size(10))
        self.mediumFontAction.triggered.connect(lambda: self.set_global_font_size(12))
        self.largeFontAction.triggered.connect(lambda: self.set_global_font_size(16))
        self.customFontAction.triggered.connect(self.set_custom_font_size)
        self.fileAction.triggered.connect(self.toggle_file)
        self.dataAction.triggered.connect(self.toggle_data_dock)
        self.configAction.triggered.connect(self.toggle_config_dock)
        self.monitorAction.triggered.connect(self.toggle_monitor_dock)
        self.dock_trainMonitor.visibilityChanged.connect(
            lambda visible: self.monitorAction.setChecked(visible))
        self.dock_trainConfig.visibilityChanged.connect(
            lambda visible: self.configAction.setChecked(visible))
        self.dock_dataManager.visibilityChanged.connect(
            lambda visible: self.dataAction.setChecked(visible))

        self.btnStopTraining.setEnabled(False)
        self.btnSaveModel.clicked.connect(self.save_model_manually)
        self.btnSaveModel.setEnabled(False)

        self.dataset_path = ""

        # 初始化绘图区域
        self.plotWidget.setLayout(QtWidgets.QVBoxLayout())
        self.plot_canvas = PlotCanvas(self.plotWidget)
        self.plotWidget.layout().addWidget(self.plot_canvas)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_from_csv)

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.read_log_file)

        self.index = 0
        self.last_log_position = 0

        self.enhancement_thread = None
        self.yolo_process = None
        self.current_results_csv = None
    def toggle_file(self):
        if self.userId:
            if self.file_window is None:
                self.file_window = FileManagerApp(self.userId)
                self.file_window.setWindowTitle("用户文件管理")
            self.file_window.show()
        else:
            QMessageBox.warning(self, "提示", "请先登录！")
            if self.login_window is None:
                self.login_window = LoginWindow()
                self.login_window.login_success.connect(self.on_login_success)
            self.login_window.show()

    def on_login_success(self, user_info):
        QMessageBox.warning(self, "提示", "登录成功！")
        self.login_window.deleteLater()
        self.userId = user_info["user_id"]
    def toggle_data_dock(self):
        if self.dock_dataManager.isVisible():
            self.dock_dataManager.hide()
        else:
            self.dock_dataManager.show()

    def toggle_monitor_dock(self):
        if self.dock_trainMonitor.isVisible():
            self.dock_trainMonitor.hide()
        else:
            self.dock_trainMonitor.show()

    def toggle_config_dock(self):
        if self.dock_trainConfig.isVisible():
            self.dock_trainConfig.hide()
        else:
            if not self.dock_trainConfig.isFloating():
                self.splitDockWidget(self.dock_dataManager, self.dock_trainConfig, Qt.Vertical)
            self.dock_trainConfig.show()

    def applyFontToChildren(self, widget, font):
        widget.setFont(font)
        for child in widget.children():
            if hasattr(child, 'setFont') and isinstance(child, QtWidgets.QWidget):
                self.applyFontToChildren(child, font)

    def set_global_font_size(self, size):
        """设置全局字体大小"""
        font = QFont("Arial", size)
        self.applyFontToChildren(self, font)
        self.statusbar.showMessage(f"字体大小已设置为 {size}pt")

    def set_custom_font_size(self):
        """弹出对话框让用户自定义字体大小"""
        size, ok = QtWidgets.QInputDialog.getInt(self, "自定义字体大小", "请输入字体大小（pt）", min=6, max=40)
        if ok:
            self.set_global_font_size(size)

    def create_dock_windows(self):

        self.setDockOptions(
            QtWidgets.QMainWindow.AnimatedDocks |
            QtWidgets.QMainWindow.AllowNestedDocks |
            QtWidgets.QMainWindow.AllowTabbedDocks
        )

        # === Dock 1: trainMonitor（左半边） ===
        self.dock_trainMonitor = QDockWidget("训练监控", self)
        self.dock_trainMonitor.setWidget(self.trainMonitor)
        self.dock_trainMonitor.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_trainMonitor.setFeatures(QDockWidget.AllDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_trainMonitor)

        # === Dock 2: dataManager（右上） ===
        self.dock_dataManager = QDockWidget("数据集管理", self)
        self.dock_dataManager.setWidget(self.dataManager)
        self.dock_dataManager.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_dataManager.setFeatures(QDockWidget.AllDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_dataManager)

        # === Dock 3: trainConfig（右下） ===
        self.dock_trainConfig = QDockWidget("训练配置", self)
        self.dock_trainConfig.setWidget(self.trainConfig)
        self.dock_trainConfig.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.dock_trainConfig.setFeatures(QDockWidget.AllDockWidgetFeatures)

        self.splitDockWidget(self.dock_trainMonitor, self.dock_trainConfig, Qt.Horizontal)
        self.splitDockWidget(self.dock_trainMonitor, self.dock_dataManager, Qt.Horizontal)
        self.splitDockWidget(self.dock_dataManager, self.dock_trainConfig, Qt.Vertical)

        # === 设置初始比例大小 ===
        new_monitor_width = int(self.width() * 0.5)
        new_monitor_height = int(self.height() * 1)
        new_data_width = int(self.width() * 0.5)
        new_data_height = int(self.height() * 0.5)
        new_config_width = int(self.width() * 0.5)
        new_config_height = int(self.height() * 0.5)
        self.resizeDocks([self.dock_trainMonitor], [new_monitor_width], Qt.Horizontal)
        self.resizeDocks([self.dock_trainMonitor], [new_monitor_height], Qt.Vertical)
        self.resizeDocks([self.dock_dataManager], [new_data_width], Qt.Horizontal)
        self.resizeDocks([self.dock_dataManager], [new_data_height], Qt.Vertical)
        self.resizeDocks([self.dock_trainConfig], [new_config_width], Qt.Horizontal)
        self.resizeDocks([self.dock_trainConfig], [new_config_height], Qt.Vertical)


    def closeEvent(self, event):
        if self.yolo_process and self.yolo_process.poll() is None:
            self.yolo_process.terminate()
            self.yolo_process.wait()
        event.accept()

    def log_text(self, text):
        QTimer.singleShot(0, lambda: self.textEditLog.append(text))

    def set_progress(self, value, status=""):
        def update():
            self.progressBar.setValue(value)
            self.labelProgress.setText(status)
        QTimer.singleShot(0, update)

    def update_selected_plot(self):
        self.index = self.plotSelect.currentIndex()
        csv_path = self.current_results_csv or get_latest_results_csv()
        if not csv_path or not os.path.exists(csv_path):
            self.log_text("无法找到 results.csv")
            return

        loss_list, map50_list, map5095_list, p_list, r_list = read_metrics_from_results_csv(csv_path,extended=True)

        if self.index == 0:
            if loss_list:
                self.update_loss_plot(loss_list)
                self.log_text("显示训练 Loss 曲线")
            else:
                self.log_text("无有效 Loss 数据")
        elif self.index == 1:
            if map50_list:
                self.update_map_plot(map50_list, map5095_list)
                self.log_text("显示 mAP 曲线")
            else:
                self.log_text("无有效 mAP 数据")
        elif self.index == 2:
            if p_list:
                self.plot_canvas.update_custom_curve(p_list, title="Precision 曲线", label="Precision", color='green')
                self.log_text("显示 Precision 曲线")
            else:
                self.log_text("无有效 Precision 数据")
        elif self.index == 3:
            if r_list:
                self.plot_canvas.update_custom_curve(r_list, title="Recall 曲线", label="Recall", color='orange')
                self.log_text("显示 Recall 曲线")
            else:
                self.log_text("无有效 Recall 数据")

    def update_map_plot(self, map50_list, map5095_list):
        QTimer.singleShot(0, lambda: self.plot_canvas.update_map_curve(map50_list, map5095_list))

    def update_loss_plot(self, loss_list):
        QTimer.singleShot(0, lambda: self.plot_canvas.update_loss_curve(loss_list))

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if folder:
            self.dataset_path = folder
            self.lineDatasetPath.setText(folder)
            self.preview_images(folder)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        new_monitor_width = int(self.width() * 0.5)
        new_monitor_height = int(self.height() * 1)
        new_data_width = int(self.width() * 0.5)
        new_data_height = int(self.height() * 0.5)
        new_config_width = int(self.width() * 0.5)
        new_config_height = int(self.height() * 0.5)
        self.resizeDocks([self.dock_trainMonitor], [new_monitor_width], Qt.Horizontal)
        self.resizeDocks([self.dock_trainMonitor], [new_monitor_height], Qt.Vertical)
        self.resizeDocks([self.dock_dataManager], [new_data_width], Qt.Horizontal)
        self.resizeDocks([self.dock_dataManager], [new_data_height], Qt.Vertical)
        self.resizeDocks([self.dock_trainConfig], [new_config_width], Qt.Horizontal)
        self.resizeDocks([self.dock_trainConfig], [new_config_height], Qt.Vertical)

    def preview_images(self, folder):
        image_labels = [self.labelImage1, self.labelImage2, self.labelImage3, self.labelImage4]
        img_folder = os.path.join(folder, "train", "images")
        image_files = sorted([
            f for f in os.listdir(img_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])[:4]

        for i in range(4):
            label = image_labels[i]
            if i < len(image_files):
                path = os.path.join(img_folder, image_files[i])
                pixmap = QPixmap(path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(pixmap)
            else:
                label.clear()
                label.setText("图片预览")
                label.setAlignment(Qt.AlignCenter)

    def upload_dataset(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "错误", "请先选择数据集目录")
            return
        QMessageBox.information(self, "提示", f"成功上传数据集：{self.dataset_path}")

    def start_training(self):

        model_type = self.comboModelType.currentText()
        model_name = model_type.split("(")[-1].replace(")", "").strip()

        self.epochs = self.spinEpochs.value()
        self.batch_size = self.spinBatchSize.value()
        self.lr = self.spinLearningRate.value()

        self.log_text("训练配置如下：")
        self.log_text(f"模型: {model_name}")
        self.log_text(f"轮次: {self.epochs}，批大小: {self.batch_size}，学习率: {self.lr}")
        self.log_text(f"数据目录: {self.dataset_path}")

        self.log_text("正在开始训练...\n")
        self.set_progress(0, "正在初始化训练任务...")

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        self.last_log_position = 0
        self.log_timer.start(1000)
        self.progress_timer.start(1000)
        thread = threading.Thread(target=self.run_yolo_subprocess, args=(model_name,))
        thread.start()

    def stop_training(self):
        if self.yolo_process and self.yolo_process.poll() is None:
            self.yolo_process.terminate()
            self.yolo_process.wait()
            self.log_text("用户中止了训练进程。")
            self.set_progress(0, "训练已中止")
            self.delayed_reset_ui()

    def run_yolo_subprocess(self, model_name):
        data_yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        if not os.path.exists(data_yaml_path):
            self.log_text("未找到 data.yaml，请确保数据目录正确")
            self.set_progress(0, "训练失败")
            return

        cmd = [
            "yolo",
            "task=detect",
            "mode=train",
            f"model={model_name}",
            f"data={data_yaml_path}",
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"lr0={self.lr}",
            "project=runs/train",
            "name=exp"
        ]

        with open(LOG_FILE, 'w', encoding='utf-8') as logfile:
            try:
                self.yolo_process = subprocess.Popen(
                    cmd,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8',
                    errors='replace'
                )
                self.yolo_process.wait()
                if self.yolo_process.returncode == 0:
                    self.set_progress(100, "训练完成")
                    self.progress_timer.stop()
                    self.current_results_csv = get_latest_results_csv()
                    self.handle_training_completion()
                else:
                    self.set_progress(0, "训练中断")
            except Exception as e:
                self.log_text(f"训练过程出错：{e}")
                self.set_progress(0, "训练失败")
            finally:
                self.delayed_reset_ui()
                self.log_timer.stop()
                self.progress_timer.stop()

    def on_enhancement_done(self):
        self.enhancement_thread.wait()
        self.log_text("数据增强完成，开始训练")
        self.set_progress(0, "开始训练")
        self.start_training()

    def start_data_enhancement(self):
        self.btnStartTraining.setEnabled(False)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(True)

        if not self.dataset_path:
            QMessageBox.warning(self, "错误", "请先选择数据集目录！")
            self.btnStartTraining.setEnabled(True)
            if self.btnStopTraining:
                self.btnStopTraining.setEnabled(False)
            return
        self.enhancement_thread = EnhancementThread(self.dataset_path, self.get_enhance_flags())
        self.enhancement_thread.progress_signal.connect(self.set_progress)
        self.enhancement_thread.log_signal.connect(self.log_text)
        self.enhancement_thread.done_signal.connect(self.on_enhancement_done)
        self.set_progress(0, "开始数据增强")
        self.enhancement_thread.start()

    def get_enhance_flags(self):
        return {
            'flip': self.checkFlip.isChecked(),
            'blur': self.checkBlur.isChecked(),
            'noise': self.checkNoise.isChecked(),
            'rotate': self.checkRotation.isChecked()
        }

    def read_log_file(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.last_log_position)
                new_lines = f.read()
                if new_lines:
                    self.log_text(new_lines.strip())
                self.last_log_position = f.tell()

    def update_progress_from_csv(self):
        csv_path = self.current_results_csv or get_latest_results_csv()
        if not csv_path or not os.path.exists(csv_path):
            return

        index = self.plotSelect.currentIndex()

        if index == 0:  # Loss
            loss_list = read_loss_from_results_csv(csv_path)
            if loss_list:
                self.update_loss_plot(loss_list)
                current_epoch = len(loss_list)
                percent = int((current_epoch / self.epochs) * 100)
                self.progressBar.setValue(min(percent, 100))
                self.labelProgress.setText(f"训练进度：{current_epoch}/{self.epochs}")

        elif index in [1, 2, 3]:  # mAP, precision, recall
            loss_list, map50_list, map5095_list, p_list, r_list = read_metrics_from_results_csv(csv_path, extended=True)
            if index == 1 and map50_list:
                self.update_map_plot(map50_list, map5095_list)
            elif index == 2 and p_list:
                self.plot_canvas.update_custom_curve(p_list, title="Precision 曲线", label="Precision", color='green')
            elif index == 3 and r_list:
                self.plot_canvas.update_custom_curve(r_list, title="Recall 曲线", label="Recall", color='orange')

    # --- 用户终止训练后延迟初始化 ---
    def delayed_reset_ui(self, delay_sec=2):
        QTimer.singleShot(delay_sec * 1000, self.reset)

    def reset(self):
        self.set_progress(0, "就绪")
        self.btnStartTraining.setEnabled(True)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(False)

    # --- 训练结束后自动提示保存模型 ---
    def handle_training_completion(self):
        self.set_progress(100, "训练完成")
        self.btnStartTraining.setEnabled(True)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(False)

        model_dir = get_latest_model_dir()
        if model_dir:
            self.model_to_save = model_dir  # 保存路径待后续按钮点击处理
            if self.btnSaveModel:
                self.btnSaveModel.setEnabled(True)
                self.log_text("训练已完成，可点击“保存模型”按钮保存训练结果。")

    def save_model_manually(self):
        if not hasattr(self, 'model_to_save') or not self.model_to_save:
            QMessageBox.warning(self, "错误", "当前没有可保存的模型目录")
            return

        save_path = QFileDialog.getExistingDirectory(self, "选择保存模型的目录", SAVE_MODEL_DIR)
        if save_path:
            try:
                shutil.copytree(self.model_to_save, os.path.join(save_path, os.path.basename(self.model_to_save)),
                                dirs_exist_ok=True)
                QMessageBox.information(self, "保存成功", f"训练结果已保存到：{save_path}")
                self.btnSaveModel.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存失败：{e}")

    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if not folder:
            return

        self.dataset_path = folder
        format_type = self.comboBoxDatasetFormat.currentText()  # UI 中新增一个下拉框 comboBoxDatasetFormat

        if format_type == "YOLO":
            self.lineDatasetPath.setText(folder)
            self.preview_images(folder)

        elif format_type == "VOC":
            try:
                yolo_dir, class_names = convert_voc_dataset(folder)
                self.dataset_path = yolo_dir
                self.lineDatasetPath.setText(yolo_dir)
                self.preview_images(yolo_dir)
                self.log_text(f"VOC 数据已成功转换为 YOLO 格式，类别：{class_names}")
            except Exception as e:
                QMessageBox.critical(self, "转换失败", str(e))

        elif format_type == "COCO":
            try:
                yolo_dir, class_names = convert_coco_dataset(folder)
                self.dataset_path = yolo_dir
                self.lineDatasetPath.setText(yolo_dir)
                self.preview_images(yolo_dir)
                self.log_text(f"COCO 数据已成功转换为 YOLO 格式，类别：{class_names}")
            except Exception as e:
                QMessageBox.critical(self, "转换失败", str(e))
