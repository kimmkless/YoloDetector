import os

from PyQt5.QtWidgets import (QMessageBox, QLabel, QMenu)
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRect, QPoint

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.normalized_rects = []  # [(norm_x, norm_y, norm_w, norm_h, class_id)]
        self.class_list = []
        self.pixmap_original = None
        self.scaled_pixmap = None
        self.scale_offset = QPoint(0, 0)
        self.scale_ratio = 1.0

    def set_class_list(self, class_list):
        self.class_list = class_list

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if not self.pixmap_original:
            return
        label_size = self.size()
        pixmap_size = self.pixmap_original.size()
        scaled_pixmap = self.pixmap_original.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled_pixmap
        self.scale_ratio = scaled_pixmap.width() / pixmap_size.width()
        x_offset = (label_size.width() - scaled_pixmap.width()) // 2
        y_offset = (label_size.height() - scaled_pixmap.height()) // 2
        self.scale_offset = QPoint(x_offset, y_offset)
        super().setPixmap(scaled_pixmap)
        self.update()

    def _to_image_coords(self, widget_point):
        x = (widget_point.x() - self.scale_offset.x()) / self.scale_ratio
        y = (widget_point.y() - self.scale_offset.y()) / self.scale_ratio
        return QPoint(int(x), int(y))

    def _to_widget_coords(self, img_x, img_y):
        x = int(img_x * self.scale_ratio) + self.scale_offset.x()
        y = int(img_y * self.scale_ratio) + self.scale_offset.y()
        return QPoint(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = self._to_image_coords(event.pos())
            self.end_point = self.start_point
            self.update()
        elif event.button() == Qt.RightButton:
            for idx, (x, y, w, h, class_id) in enumerate(self.normalized_rects):
                rect = self._denorm_rect(x, y, w, h)
                if rect.contains(event.pos()):
                    self.show_edit_menu(event.globalPos(), idx)
                    break

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = self._to_image_coords(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.show_class_menu(event.globalPos(), rect)
            self.start_point = None
            self.end_point = None
            self.update()

    def show_class_menu(self, global_pos, rect):
        menu = QMenu(self)
        for idx, cls in enumerate(self.class_list):
            action = menu.addAction(cls)
            action.triggered.connect(lambda _, i=idx: self.add_rect_with_class(rect, i))
        menu.exec_(global_pos)

    def show_edit_menu(self, global_pos, index):
        menu = QMenu(self)
        delete_action = menu.addAction("删除标注")
        delete_action.triggered.connect(lambda: self.delete_rect(index))
        submenu = menu.addMenu("更改类别")
        for idx, cls in enumerate(self.class_list):
            action = submenu.addAction(cls)
            action.triggered.connect(lambda _, i=idx: self.change_class(index, i))
        menu.exec_(global_pos)

    def add_rect_with_class(self, rect, class_id):
        iw = self.pixmap_original.width()
        ih = self.pixmap_original.height()
        x = (rect.left() + rect.width() / 2) / iw
        y = (rect.top() + rect.height() / 2) / ih
        w = rect.width() / iw
        h = rect.height() / ih
        self.normalized_rects.append((x, y, w, h, class_id))
        self.update()

    def delete_rect(self, index):
        if 0 <= index < len(self.normalized_rects):
            self.normalized_rects.pop(index)
            self.update()

    def change_class(self, index, new_class_id):
        if 0 <= index < len(self.normalized_rects):
            x, y, w, h, _ = self.normalized_rects[index]
            self.normalized_rects[index] = (x, y, w, h, new_class_id)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.scaled_pixmap:
            return
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        for x, y, w, h, class_id in self.normalized_rects:
            rect = self._denorm_rect(x, y, w, h)
            painter.drawRect(rect)
            if 0 <= class_id < len(self.class_list):
                painter.drawText(rect.topLeft() + QPoint(2, 15), self.class_list[class_id])
        if self.drawing and self.start_point and self.end_point:
            temp_rect = QRect(self._to_widget_coords(self.start_point.x(), self.start_point.y()),
                              self._to_widget_coords(self.end_point.x(), self.end_point.y())).normalized()
            painter.drawRect(temp_rect)

    def _denorm_rect(self, x, y, w, h):
        iw = self.pixmap_original.width()
        ih = self.pixmap_original.height()
        x1 = (x - w / 2) * iw
        y1 = (y - h / 2) * ih
        x2 = (x + w / 2) * iw
        y2 = (y + h / 2) * ih
        top_left = self._to_widget_coords(x1, y1)
        bottom_right = self._to_widget_coords(x2, y2)
        return QRect(top_left, bottom_right)

    def get_normalized_annotations(self, *_):
        return [[cls, x, y, w, h] for x, y, w, h, cls in self.normalized_rects]

    def clear_rectangles(self):
        self.normalized_rects.clear()
        self.update()

    def load_from_file(self, label_path, *_):
        self.normalized_rects.clear()
        if not os.path.exists(label_path):
            return
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x, y, w, h = map(float, parts)
                self.normalized_rects.append((x, y, w, h, int(class_id)))
        self.update()

# Other classes unchanged, only YAML config saving below changed
    def generate_yaml_config(self):
        if not self.image_dir:
            QMessageBox.warning(self, "警告", "请先打开数据集目录!")
            return
        config = {
            'path': self.image_dir,
            'train': None,
            'val': None,
            'test': None,
            'names': self.class_list
        }
        yaml_path = os.path.join(self.image_dir, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {config['path']}\n")
            f.write("train: null\n")
            f.write("val: null\n")
            f.write("test: null\n")
            f.write("names: [" + ", ".join(config['names']) + "]\n")
        QMessageBox.information(self, "成功", f"已生成YAML配置文件: {yaml_path}")
        self.statusbar.showMessage(f"YAML配置文件已生成: {yaml_path}")
