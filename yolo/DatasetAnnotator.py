import yaml
import cv2
import os

from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QMessageBox,
                             QListWidgetItem, QLabel, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5 import uic
from ImageLabel import ImageLabel

class DatasetAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/LabelGUI.ui", self)
        self.image_dir = ""
        self.classes = []
        self.current_image_index = 0
        self.image_files = []
        self.imageLabel_original = self.findChild(QLabel, "imageLabel")
        self.imageLabel = ImageLabel(self)
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setMinimumSize(800, 600)
        layout = self.imageLabel_original.parent().layout()
        layout.replaceWidget(self.imageLabel_original, self.imageLabel)
        self.imageLabel_original.deleteLater()
        self.setStyleSheet("background-color: rgb(120, 120, 120);")
        self.connect_signals()
        self.load_default_config()

    def connect_signals(self):
        self.btnOpenDataset.clicked.connect(self.open_dataset)
        self.btnAddClass.clicked.connect(self.add_class)
        self.btnPrevImage.clicked.connect(self.prev_image)
        self.btnNextImage.clicked.connect(self.next_image)
        self.btnSaveAnnotations.clicked.connect(self.save_annotations)
        self.btnGenerateYAML.clicked.connect(self.generate_yaml_config)
        self.classListWidget.itemClicked.connect(self.class_selected)
        self.imageListWidget.itemClicked.connect(self.image_selected)
        self.actionOpenDataset.triggered.connect(self.open_dataset)
        self.actionExit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.show_about)

    def load_default_config(self):
        self.classes = ["cat", "dog", "person"]
        self.imageLabel.set_class_list(self.classes)
        self.update_class_list()

    def open_dataset(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if dir_path:
            self.image_dir = dir_path
            self.load_images()
            self.statusbar.showMessage(f"已加载数据集: {dir_path}")

    def load_images(self):
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_files.append(os.path.join(root, file))
        self.imageListWidget.clear()
        for img_path in self.image_files:
            item = QListWidgetItem(os.path.basename(img_path))
            item.setData(Qt.UserRole, img_path)
            self.imageListWidget.addItem(item)
        if self.image_files:
            self.display_image(0)

    def add_class(self):
        class_name, ok = QInputDialog.getText(self, "添加类别", "输入新类别名称:")
        if ok and class_name:
            if class_name not in self.classes:
                self.classes.append(class_name)
                self.imageLabel.set_class_list(self.classes)
                self.update_class_list()

    def update_class_list(self):
        self.classListWidget.clear()
        for class_name in self.classes:
            self.classListWidget.addItem(class_name)

    def class_selected(self, item):
        pass

    def image_selected(self, item):
        img_path = item.data(Qt.UserRole)
        if img_path in self.image_files:
            self.current_image_index = self.image_files.index(img_path)
            self.display_image(self.current_image_index)

    def display_image(self, index):
        if 0 <= index < len(self.image_files):
            img_path = self.image_files[index]
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = image.shape
                bytes_per_line = ch * w
                q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.imageLabel.setPixmap(scaled_pixmap)
                self.imageLabel.clear_rectangles()
                label_path = os.path.join(self.image_dir, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                self.imageLabel.load_from_file(label_path, w, h)
                self.imageListWidget.setCurrentRow(index)

    def prev_image(self):
        if self.image_files:
            self.current_image_index = max(0, self.current_image_index - 1)
            self.display_image(self.current_image_index)

    def next_image(self):
        if self.image_files:
            self.current_image_index = min(len(self.image_files) - 1, self.current_image_index + 1)
            self.display_image(self.current_image_index)

    def save_annotations(self):
        if not self.image_dir:
            QMessageBox.warning(self, "警告", "请先打开数据集目录!")
            return
        labels_dir = os.path.join(self.image_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        img_path = self.image_files[self.current_image_index]
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        annotations = self.imageLabel.get_normalized_annotations(w, h)
        txt_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
        with open(txt_path, 'w') as f:
            for cls_id, x, y, width, height in annotations:
                f.write(f"{cls_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n")
        classes_path = os.path.join(self.image_dir, "classes.txt")
        with open(classes_path, 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")
        QMessageBox.information(self, "保存成功", f"标注文件已保存: {txt_path}")
        self.statusbar.showMessage(f"标注文件已保存: {txt_path}")

    def generate_yaml_config(self):
        if not self.image_dir:
            QMessageBox.warning(self, "警告", "请先打开数据集目录!")
            return
        config = {
            'path': self.image_dir,
            'train': None,
            'val': None,
            'test': None,
            'names': self.classes
        }
        yaml_path = os.path.join(self.image_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=True, allow_unicode=True)
        QMessageBox.information(self, "成功", f"已生成YAML配置文件: {yaml_path}")
        self.statusbar.showMessage(f"YAML配置文件已生成: {yaml_path}")

    def show_about(self):
        QMessageBox.about(self, "关于", "数据集标注工具\n\n版本: 1.0\n功能: 支持图像分类标注，生成YOLO格式数据集")
