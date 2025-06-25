import sys
import os
import time
import subprocess
import threading
import matplotlib.pyplot as plt
from ReadLoss import read_loss_from_results_csv
from ReadMetrics import read_metrics_from_results_csv
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer,Qt
from threading import Event
from PlotCanvas import PlotCanvas
from glob import glob

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class YoloTrainerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Train.ui", self)

        self.btnSelectDataset.clicked.connect(self.select_dataset_folder)
        self.btnUploadDataset.clicked.connect(self.upload_dataset)
        self.btnStartTraining.clicked.connect(self.start_training)

        self.dataset_path = ""

        self.plot_canvas = PlotCanvas(self.plotWidget)
        self.verticalLayout_8.addWidget(self.plot_canvas)
        self.stop_event = Event()

    def closeEvent(self, event):
        self.stop_event.set()
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            self.training_thread.join(timeout=2)
        if hasattr(self, 'progress_updating_thread') and self.progress_updating_thread.is_alive():
            self.progress_updating_thread.join(timeout=2)
        event.accept()

    # ---------- 主线程安全更新方法 ----------
    def log_text(self, text):
        QTimer.singleShot(0, lambda: self.textEditLog.append(text))

    def set_progress(self, value, status=""):
        def update():
            self.progressBar.setValue(value)
            self.labelProgress.setText(status)
        QTimer.singleShot(0, update)

    #----------添加UI安全调用封装----------
    def update_map_plot(self, map50_list, map5095_list):
        QTimer.singleShot(0, lambda: self.plot_canvas.update_map_curve(map50_list, map5095_list))

    def update_loss_plot(self, loss_list):
        QTimer.singleShot(0, lambda: self.plot_canvas.update_loss_curve(loss_list))

    def get_latest_results_csv(self,base_dir='runs/train'):
        exp_dirs = glob(os.path.join(base_dir, 'exp*'))
        latest_csv = None
        latest_time = 0

        for exp_path in exp_dirs:
            csv_path = os.path.join(exp_path, 'results.csv')
            if os.path.exists(csv_path):
                mtime = os.path.getmtime(csv_path)
                if mtime > latest_time:
                    latest_time = mtime
                    latest_csv = csv_path

        return latest_csv  # 若没有找到则为 None
    # ---------- UI 控件操作 ----------
    def select_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if folder:
            self.dataset_path = folder
            self.lineDatasetPath.setText(folder)
            self.preview_images(folder)

    def preview_images(self, folder):
        # 训练目标预览控件
        image_labels = [self.labelImage1, self.labelImage2, self.labelImage3, self.labelImage4]
        img_folder = folder + "/train/images"

        # 只读取图片文件
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

    # ---------- 训练启动 ----------

    def start_training(self):
        self.btnStartTraining.setEnabled(False)
        if not self.dataset_path:
            QMessageBox.warning(self, "错误", "请先选择数据集目录！")
            self.btnStartTraining.setEnabled(True)
            return

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
        self.progressBar.setValue(0)

        thread = threading.Thread(target=self.run_yolo_subprocess, args=(model_name,))
        thread.start()

        progress_update_thread = threading.Thread(target=self.update_progress_from_csv)
        progress_update_thread.start()


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

        try:
            self.yolo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            for line in self.yolo_process.stdout:
                self.log_text(line.strip())

            self.yolo_process.wait()
            self.set_progress(100, "训练完成")
            self.update_final_results()
            self.btnStartTraining.setEnabled(True)

        except Exception as e:
            self.log_text(f"训练过程出错：{e}")
            self.set_progress(0, "训练失败")

    def update_progress_from_csv(self):
        while not self.stop_event.is_set():
            csv_path = self.get_latest_results_csv()
            if os.path.exists(csv_path):
                loss_list = read_loss_from_results_csv(csv_path)
                if loss_list:
                    self.update_loss_plot(loss_list)
                    current_epoch = len(loss_list)
                    percent = int((current_epoch / self.epochs) * 100)
                    self.progressBar.setValue(min(percent, 100))
                    self.labelProgress.setText(f"训练进度：{current_epoch}/{self.epochs}")
            time.sleep(1)

    def update_final_results(self):
        self.plot_metrics_from_csv()

    def plot_metrics_from_csv(self):
        csv_path = "runs/train/exp/results.csv"
        if not os.path.exists(csv_path):
            self.log_text("无法找到 results.csv 文件")
            return

        loss_list, map50_list, map5095_list = read_metrics_from_results_csv(csv_path)

        if not map50_list:
            self.log_text("CSV 文件中无有效 mAP 数据")
            return

        self.update_map_plot(map50_list, map5095_list)
        self.log_text("已绘制 mAP 曲线")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = YoloTrainerApp()
    window.show()
    sys.exit(app.exec_())
