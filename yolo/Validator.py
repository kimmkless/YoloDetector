import os
import sys
import subprocess
import re
import matplotlib
import yaml

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtCore import QThread, pyqtSignal, Qt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class ValidatorWorker(QThread):
    log_signal = pyqtSignal(str)
    result_dir_signal = pyqtSignal(str)
    metrics_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    log_lines_signal = pyqtSignal(str)

    def __init__(self, model_path, data_yaml_path):
        super().__init__()
        self.log_lines = []
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path

    def run(self):
        try:
            # 写入 validate_runner.py 脚本
            runner_code = '''
import sys
import matplotlib
from ultralytics import YOLO

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def main():
    if len(sys.argv) < 3:
        print("[Error] 参数不足：需要模型路径和 data.yaml 路径")
        return

    model_path = sys.argv[1]
    data_yaml_path = sys.argv[2]

    try:
        model = YOLO(model_path)
        results = model.val(data=data_yaml_path)

        print(f"[RESULT_SAVE_DIR] {results.save_dir}")

        box = results.box
        print(f"[METRIC] map={float(box.map):.3f}, map50={float(box.map50):.3f}, map75={float(box.map75):.3f}")

        for i, ap in enumerate(box.maps):
            print(f"[CLASS_AP] {i} {float(ap):.3f}")

    except Exception as e:
        print(f"[Error] {e}")

if __name__ == "__main__":
    main()
'''
            runner_path = os.path.join(os.getcwd(), "validate_runner.py")
            with open(runner_path, "w", encoding="utf-8") as f:
                f.write(runner_code)

            process = subprocess.Popen(
                [sys.executable, runner_path, self.model_path, self.data_yaml_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="ignore",
                bufsize=1
            )

            # 初始化指标字典
            metrics = {
                'mAP50': None, 'mAP75': None, 'mAP95': None,
                'class_names': [], 'instances': [], 'precision': [], 'recall': [], 'ap': [], 'log_lines': []
            }

            for line in process.stdout:
                self.log_signal.emit(line)
                self.log_lines.append(line)

                if "[Error]" in line:
                    self.error_signal.emit(line.strip())
                    break

                if "[METRIC]" in line:
                    match = re.search(r"map=([\d.]+), map50=([\d.]+), map75=([\d.]+)", line)
                    if match:
                        metrics['mAP95'] = float(match.group(1))
                        metrics['mAP50'] = float(match.group(2))
                        metrics['mAP75'] = float(match.group(3))

                if "[CLASS_AP]" in line:
                    match = re.search(r"\[CLASS_AP\] (\d+) ([\d.]+)", line)
                    if match:
                        metrics['ap'].append(float(match.group(2)))

                if "[RESULT_SAVE_DIR]" in line:
                    save_dir = line.strip().split("]", 1)[1].strip()
                    self.result_dir_signal.emit(save_dir)

            metrics['log_lines'] = self.log_lines
            self.metrics_signal.emit(metrics)

            process.stdout.close()
            process.wait()

            try:
                os.remove(runner_path)
            except:
                pass

        except Exception as e:
            self.error_signal.emit(f"[Error] {e}")

class ValidatorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Assets/UI/ValidGUI.ui", self)

        self.model_path = ""
        self.dataset_path = ""
        self.curve_img_path = ""

        self.results = None

        self.btnSelectModel.clicked.connect(self.select_model)
        self.btnSelectDataset.clicked.connect(self.select_dataset)
        self.btnStartValidate.clicked.connect(self.start_validation)
        self.comboCurve.currentIndexChanged.connect(self.show_curve)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窗口大小变化时重新调整指标图片
        if hasattr(self, 'curve_img_path'):
            self.display_metric_image()

    def select_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型", "", "YOLO 模型 (*.pt)")
        if path:
            self.model_path = path
            self.lineModelPath.setText(path)

    def select_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据集主目录")
        if path:
            self.dataset_path = path
            self.lineDatasetPath.setText(path)

    def start_validation(self):
        if not self.model_path or not self.dataset_path:
            QtWidgets.QMessageBox.warning(self, "错误", "请选择模型和数据集路径。")
            return

        yaml_path = os.path.join(self.dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            QtWidgets.QMessageBox.critical(self, "错误", f"未找到 data.yaml 文件：\n{yaml_path}")
            return

        self.textResult.clear()
        if hasattr(self, 'statusBar'):
            self.statusBar.showMessage("正在验证模型，请稍候...")
        self.textResult.append("开始验证模型...\n")

        self.worker = ValidatorWorker(self.model_path, yaml_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.result_dir_signal.connect(self.load_results_and_update_ui)
        self.worker.metrics_signal.connect(self.update_metrics_ui)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.start()

    def append_log(self, msg):
        self.textResult.moveCursor(QTextCursor.End)
        self.textResult.insertPlainText(msg)
        self.textResult.ensureCursorVisible()

    def handle_error(self, msg):
        if hasattr(self, 'statusBar'):
            self.statusBar.showMessage("模型验证失败")
        self.textResult.append(f"\n验证失败：{msg}\n")

    def load_results_and_update_ui(self, save_dir):
        self.results = type('DummyResult', (), {"save_dir": save_dir})()
        self.show_curve()

    def show_curve(self):
        if not self.results:
            return

        curve_map = {
            "Precision-Recall (PR)": "pr_curve.png",
            "Precision": "P_curve.png",
            "Recall": "R_curve.png",
            "F1 Score": "F1_curve.png"
        }

        curve_name = self.comboCurve.currentText()
        img_path = os.path.join(self.results.save_dir, curve_map.get(curve_name, ""))

        self.curve_img_path = img_path  # 保存原始图片路径
        self.update_curve_image()

    def update_metrics_ui(self, metrics):
        try:
            # 更新整体 mAP 指标
            if metrics['mAP95'] is not None:
                self.mAP95.setText(f"{metrics['mAP95']:.3f}")
            if metrics['mAP50'] is not None:
                self.mAP50.setText(f"{metrics['mAP50']:.3f}")
            if metrics['mAP75'] is not None:
                self.mAP75.setText(f"{metrics['mAP75']:.3f}")
        except Exception as e:
            self.textResult.append(f"[UI更新失败] {e}")

        try:
            # 清空控件
            self.classWidget.clear()
            self.imagesWidget.clear()
            self.instancesWidget.clear()
            self.percisionWidget.clear()
            self.recallWidget.clear()
            self.mAPWidget.clear()

            # 加载类别名称
            yaml_path = os.path.join(self.dataset_path, "data.yaml")
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            class_names = data.get("names", [])

            # 提取每类结果表格行
            log_lines = metrics.get("log_lines", [])
            num_classes = len(metrics.get("ap", []))
            table_lines = []

            for line in log_lines:
                # 匹配每类行：开头是数字，有多个数字字段
                if re.match(r"^\s*\d+\s+\d+\s+\d+", line):
                    table_lines.append(line.strip())
                if len(table_lines) == num_classes:
                    break

            for i, line in enumerate(table_lines):
                parts = line.split()
                if len(parts) < 6:
                    continue

                class_id = i
                class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

                images = parts[0]
                instances = parts[1]
                p = parts[2]
                r = parts[3]
                map50 = parts[4]

                print(class_name)
                print(images)
                print(instances)
                print(map50)
                print(p)
                self.classWidget.addItem(class_name)
                self.imagesWidget.addItem(images)
                self.instancesWidget.addItem(instances)
                self.percisionWidget.addItem(f"{float(p):.3f}")
                self.recallWidget.addItem(f"{float(r):.3f}")
                self.mAPWidget.addItem(f"{float(map50):.3f}")

            if hasattr(self, 'statusBar'):
                self.statusBar.showMessage("验证完成！")
            self.textResult.append("\n验证完成，指标已更新。\n")

        except Exception as e:
            self.textResult.append(f"\n[指标解析错误] {e}\n")

    def update_curve_image(self):
        if not os.path.exists(self.curve_img_path):
            self.labelCurve.setText("未找到图像")
            return

        self.display_metric_image()

    def display_metric_image(self):
        """显示或重新调整当前图片的大小"""
        if not hasattr(self, 'curve_img_path') or not os.path.exists(self.curve_img_path):
            return

        pixmap = QPixmap(self.curve_img_path)
        if pixmap.isNull():
            print("[错误] 加载图像失败：", self.curve_img_path)
            self.labelCurve.setText("图像加载失败")
            return

        # 调整图片大小以适应标签
        scaled = pixmap.scaled(
            self.labelCurve.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.labelCurve.setPixmap(scaled)
        self.labelCurve.setAlignment(Qt.AlignCenter)
