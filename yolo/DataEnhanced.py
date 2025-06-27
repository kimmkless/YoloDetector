import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class EnhancementThread(QThread):
    progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    def __init__(self, dataset_path, flags):
        super().__init__()
        self.dataset_path = os.path.abspath(dataset_path)
        self.flags = flags

    def run(self):
        input_dir = os.path.join(self.dataset_path, "train", "images")
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        total = len(image_files)
        count = 0

        if not any(self.flags.values()):
            self.log_signal.emit("未勾选任何数据增强选项，跳过增强阶段。")
            self.done_signal.emit()
            return

        for i, filename in enumerate(image_files):
            image_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)

            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.log_signal.emit(f"读取失败: {image_path}")
                continue

            augmented = image.copy()
            if self.flags.get("blur"):
                augmented = cv2.GaussianBlur(augmented, (5, 5), 0)
            if self.flags.get("flip"):
                augmented = cv2.flip(augmented, 1)
            if self.flags.get("noise"):
                noise = np.random.normal(0, 25, augmented.shape).astype(np.uint8)
                augmented = cv2.add(augmented, noise)
            if self.flags.get("rotate"):
                (h, w) = augmented.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
                augmented = cv2.warpAffine(augmented, M, (w, h))

            cv2.imencode(ext, augmented)[1].tofile(image_path)
            count += 1

            percent = int(((i + 1) / total) * 100)
            self.progress_signal.emit(percent, f"数据增强中 {percent}%")

        if count > 0:
            self.log_signal.emit(f"数据增强完成，共增强 {count} 张图像。")
        else:
            self.log_signal.emit("未进行任何增强操作。")

        self.done_signal.emit()
