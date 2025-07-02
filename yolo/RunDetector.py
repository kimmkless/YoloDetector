import cv2
import time
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
from collections import deque

class DetectionWorker(QThread):

    frame_processed = pyqtSignal(np.ndarray)
    result_updated = pyqtSignal(str)
    fps_updated = pyqtSignal(float)
    progress_updated = pyqtSignal(int, int)  # current frame, total frames

    def __init__(self, model, get_params_func, input_type, input_path=None, save_dir=None, file_path=None):
        super().__init__()
        self.model = model
        self.get_params = get_params_func
        self.input_type = input_type
        self.input_path = input_path
        self.save_dir = save_dir
        self.file_path = file_path
        self.save_frame_index = 0
        self.saved_frames = []  # 存储每帧路径
        self.original_fps = 30  # 默认帧率，如有视频源可动态获取

        self._running = True
        self._paused = False

        self.current_frame_index = 0
        self.target_frame_index = None
        self.single_step_mode = False  # 单帧播放模式

        # 用于暂停/恢复控制
        self.mutex = QMutex()
        self.pause_cond = QWaitCondition()

        self.frame_timestamps = deque(maxlen=30)  # 记录最近 30 帧时间

    def stop(self):
        self._running = False
        self.resume()  # 如果线程暂停状态，确保能被唤醒退出

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        self.pause_cond.wakeAll()

    def run(self):
        cap = None
        if self.input_type == "摄像头":
            cap = cv2.VideoCapture(0)
        elif self.input_type == "视频":
            cap = cv2.VideoCapture(self.input_path)
            self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        elif self.input_type == "图片":
            frame = cv2.imread(self.input_path)
            self.process_frame(frame)
            return

        if not cap or not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self._running:
            self.mutex.lock()
            if self._paused:
                self.pause_cond.wait(self.mutex)
            self.mutex.unlock()

            if self.target_frame_index is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame_index)
                self.current_frame_index = self.target_frame_index
                self.target_frame_index = None

            ret, frame = cap.read()
            if not ret:
                break

            self.current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            start = time.time()
            self.process_frame(frame)
            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            self.update_fps()
            self.progress_updated.emit(self.current_frame_index, total_frames)

            # 单帧模式：仅在暂停状态下，播放一帧就暂停
            if self.single_step_mode and self._paused is False:
                self.pause()

            _, _, delay = self.get_params()
            time.sleep(delay)

        if self.input_type == "视频" and self.save_dir:
            ext = os.path.splitext(self.file_path)[1].lower()
            output_video_path = os.path.join(self.save_dir, "detection_output"+ext)
            self.save_video_from_frames(output_video_path, self.original_fps)
            self.delete_frames()
        cap.release()

    def process_frame(self, frame):
        conf, iou, _ = self.get_params()
        results = self.model(frame, conf=conf, iou=iou)[0]
        plotted = results.plot()
        self.frame_processed.emit(plotted)

        # 保存检测图像
        if self.save_dir:
            save_path = os.path.join(self.save_dir, f"frame_{self.save_frame_index:04d}.jpg")
            cv2.imwrite(save_path, plotted)
            self.save_frame_index += 1
            self.saved_frames.append(save_path)

        names = self.model.names
        class_counts = {}
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        result_text = "\n".join(f"{cls}: {cnt} 个" for cls, cnt in class_counts.items())
        self.result_updated.emit("检测结果：\n" + result_text)

    def update_fps(self):
        now = time.time()
        self.frame_timestamps.append(now)

        if len(self.frame_timestamps) >= 2:
            duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            frame_count = len(self.frame_timestamps) - 1
            avg_fps = frame_count / duration if duration > 0 else 0.0
            self.fps_updated.emit(avg_fps)

    def delete_frames(self):
        for path in self.saved_frames:
            try:
                os.remove(path)
            except Exception as e:
                print(f"[WARN] 无法删除帧 {path}：{e}")

    def save_video_from_frames(self, output_path, fps):
        if not self.saved_frames:
            print("[WARN] 无帧可用于合成视频")
            return

        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        elif ext == '.mkv':
            fourcc = cv2.VideoWriter_fourcc(*'X264')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 默认 mp4

        first_frame = cv2.imread(self.saved_frames[0])
        height, width, _ = first_frame.shape

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_path in sorted(self.saved_frames):
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"[INFO] 视频保存成功: {output_path}")