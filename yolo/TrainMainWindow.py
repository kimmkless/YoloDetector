import sys
import cv2
import numpy as np
import random
import os
import time
import subprocess
import threading
import yaml
import matplotlib.pyplot as plt
from ReadLoss import read_loss_from_results_csv
from ReadMetrics import read_metrics_from_results_csv
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt
from PlotCanvas import PlotCanvas

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_TIME = time.time()
LOG_FILE = "train_output.log"
SAVE_MODEL_DIR = "saved_models"

class YoloTrainerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/Train.ui", self)

        self.btnSelectDataset.clicked.connect(self.select_dataset_folder)
        self.btnUploadDataset.clicked.connect(self.upload_dataset)
        self.btnStartTraining.clicked.connect(self.start_training)
        self.plotSelect.currentIndexChanged.connect(self.update_selected_plot)
        self.btnStopTraining.clicked.connect(self.stop_training)
        self.btnStopTraining.setEnabled(False)
        self.btnSaveModel.clicked.connect(self.save_model_manually)
        self.btnSaveModel.setEnabled(False)

        self.dataset_path = ""

        self.plotWidget.setLayout(QtWidgets.QVBoxLayout())
        self.plotWidget.layout().addWidget(self.plotSelect)

        self.plot_canvas = PlotCanvas(self.plotWidget)
        self.plot_canvas.setParent(None)
        self.plotWidget.layout().addWidget(self.plot_canvas)

        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_from_csv)

        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.read_log_file)
        self.last_log_position = 0

        self.yolo_process = None
        self.current_results_csv = None

        self.augment_flags = {
            'flip': self.checkFlip.isChecked(),
            'blur': self.checkBlur.isChecked(),
            'noise': self.checkNoise.isChecked(),
            'rotate': self.checkRotation.isChecked()
        }
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
        index = self.plotSelect.currentIndex()
        csv_path = self.current_results_csv or self.get_latest_results_csv()
        if not csv_path or not os.path.exists(csv_path):
            self.log_text("无法找到 results.csv")
            return

        loss_list, map50_list, map5095_list, p_list, r_list = read_metrics_from_results_csv(csv_path,extended=True)

        if index == 0:
            if loss_list:
                self.update_loss_plot(loss_list)
                self.log_text("显示训练 Loss 曲线")
            else:
                self.log_text("无有效 Loss 数据")
        elif index == 1:
            if map50_list:
                self.update_map_plot(map50_list, map5095_list)
                self.log_text("显示 mAP 曲线")
            else:
                self.log_text("无有效 mAP 数据")
        elif index == 2:
            if p_list:
                self.plot_canvas.update_custom_curve(p_list, title="Precision 曲线", label="Precision", color='green')
                self.log_text("显示 Precision 曲线")
            else:
                self.log_text("无有效 Precision 数据")
        elif index == 3:
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

    def preview_images(self, folder):
        image_labels = [self.labelImage1, self.labelImage2, self.labelImage3, self.labelImage4]
        img_folder = folder + "/train/images"

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
        self.btnStartTraining.setEnabled(False)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(True)

        if not self.dataset_path:
            QMessageBox.warning(self, "错误", "请先选择数据集目录！")
            self.btnStartTraining.setEnabled(True)
            if self.btnStopTraining:
                self.btnStopTraining.setEnabled(False)
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

        self.log_text("开始进行数据增强...")
        self.apply_data_augmentation()

        # 若增强图像已保存，尝试修改 data.yaml 的 train 路径
        self.update_yaml_for_augmented_data(self.dataset_path)

        self.set_progress(0, "准备开始训练...")

        self.log_text("正在开始训练...\n")

        self.set_progress(0, "正在初始化训练任务...")

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        self.last_log_position = 0
        self.log_timer.start(1000)

        thread = threading.Thread(target=self.run_yolo_subprocess, args=(model_name,))
        thread.start()
        self.progress_timer.start(1000)

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
                    self.current_results_csv = self.get_latest_results_csv()
                    self.handle_training_completion()
                    self.update_final_results()
                else:
                    self.set_progress(0, "训练中断")
            except Exception as e:
                self.log_text(f"训练过程出错：{e}")
                self.set_progress(0, "训练失败")
            finally:
                self.delayed_reset_ui()
                self.log_timer.stop()

    def read_log_file(self):
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(self.last_log_position)
                new_lines = f.read()
                if new_lines:
                    self.log_text(new_lines.strip())
                self.last_log_position = f.tell()

    def update_progress_from_csv(self):
        csv_path = self.current_results_csv or self.get_latest_results_csv()
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

    def update_final_results(self):
        self.plot_metrics_from_csv()
        self.plotSelect.setCurrentIndex(1)

    def plot_metrics_from_csv(self):
        self.update_selected_plot()

    # --- 用户终止训练后延迟初始化 ---
    def delayed_reset_ui(self, delay_sec=2):
        QTimer.singleShot(delay_sec * 1000, self.reset)

    def reset(self):
        self.set_progress(0, "就绪")
        self.btnStartTraining.setEnabled(True)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(False)

    def get_latest_model_dir(self, base_dir='runs/train'):
        latest_dir = None
        latest_time = 0
        for root, dirs, files in os.walk(base_dir):
            for d in dirs:
                full_path = os.path.join(root, d)
                if os.path.isdir(full_path):
                    mtime = os.path.getmtime(full_path)
                    if mtime > latest_time:
                        latest_time = mtime
                        latest_dir = full_path
        return latest_dir

    # --- 训练结束后自动提示保存模型 ---
    def handle_training_completion(self):
        self.set_progress(100, "训练完成")
        self.btnStartTraining.setEnabled(True)
        if self.btnStopTraining:
            self.btnStopTraining.setEnabled(False)

        model_dir = self.get_latest_model_dir()
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
                import shutil
                shutil.copytree(self.model_to_save, os.path.join(save_path, os.path.basename(self.model_to_save)),
                                dirs_exist_ok=True)
                QMessageBox.information(self, "保存成功", f"训练结果已保存到：{save_path}")
                self.btnSaveModel.setEnabled(False)
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存失败：{e}")


    def get_latest_results_csv(self, base_dir='runs/train'):
        latest_csv = None
        latest_time = 0
        for root, dirs, files in os.walk(base_dir):
            if 'results.csv' in files:
                path = os.path.join(root, 'results.csv')
                mtime = os.path.getmtime(path)
                if mtime > latest_time and mtime > CURRENT_TIME:
                    latest_time = mtime
                    latest_csv = path
        return latest_csv

    def apply_data_augmentation(self):

        image_dir = os.path.join(self.dataset_path, "train/images")
        output_dir = os.path.join(self.dataset_path, "train/images_augmented")

        os.makedirs(output_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        total = len(image_files)
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_dir, img_file)
            image = self.safe_imread(img_path)
            if image is None:
                print(f"读取失败：{img_path}")
                continue

            # 应用增强
            if self.augment_flags.get("flip") and random.random() > 0.5:
                image = cv2.flip(image, 1)

            if self.augment_flags.get("blur") and random.random() > 0.5:
                k = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (k, k), 0)

            if self.augment_flags.get("noise") and random.random() > 0.5:
                noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)

            if self.augment_flags.get("rotate") and random.random() > 0.5:
                angle = random.choice([-15, -10, 10, 15])
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # 保存副本到新路径，保留原文件名
            save_name = os.path.splitext(img_file)[0] + "_aug.jpg"
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, image)

            # 更新进度条
            percent = int((idx + 1) / total * 100)
            self.set_progress(percent, f"正在增强数据集 ({idx + 1}/{total})")

        self.log_text(f"增强完成，增强图片已保存至 {output_dir}")

    def update_yaml_for_augmented_data(self, dataset_path):
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            self.log_text("未找到 data.yaml，跳过增强路径更新")
            return

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            updated = False
            # 更新 train 路径
            train_dir = os.path.join(dataset_path, "train", "images_augmented")
            if os.path.exists(train_dir) and os.listdir(train_dir):
                data["train"] = os.path.relpath(train_dir, start=dataset_path)
                updated = True

            # 如果 test/val 有增强你可以仿照添加
            # val_dir = ...
            # if os.path.exists(val_dir): ...

            if updated:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f, allow_unicode=True)
                self.log_text("已更新 data.yaml 中的 train 路径为增强图像目录")
            else:
                self.log_text("未生成增强图像目录，保持 data.yaml 不变")

        except Exception as e:
            self.log_text(f"修改 data.yaml 时出错：{e}")

    def safe_imread(self, path):
        try:
            # 支持中文路径读取图像
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ERROR] Failed to load image {path}: {e}")
            return None

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = YoloTrainerApp()
    window.show()
    sys.exit(app.exec_())
