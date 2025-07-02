import sys
import pymysql
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog

db_config = {
    "host": "rm-bp15k902w8a33bs6x9o.mysql.rds.aliyuncs.com",
    "user": "root",
    "password": "123456Lyx",
    "database": "user",
    "charset": "utf8mb4"
}
class FileUploader:
    def __init__(self):
        self.conn = pymysql.connect(**db_config)
        self.cursor = self.conn.cursor()

    def upload_file(self, user_id, parent=None):
        file_path, _ = QFileDialog.getOpenFileName(parent, "选择文件")
        if file_path:
            self._save_file_to_db(user_id, file_path)

    def upload_folder(self, user_id, parent=None):
        folder_path = QFileDialog.getExistingDirectory(parent, "选择文件夹")
        if folder_path:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    self._save_file_to_db(user_id, os.path.join(root, file))

    def _save_file_to_db(self, user_id, path):
        filename = os.path.basename(path)
        filetype = filename.split('.')[-1].lower()
        with open(path, 'rb') as f:
            content = f.read()

        sql = """
        INSERT INTO user_files (user_id, filename, filetype, filepath, content)
        VALUES (%s, %s, %s, %s, %s)
        """
        self.cursor.execute(sql, (user_id, filename, filetype, path, content))
        self.conn.commit()

    def list_user_files(self, user_id):
        sql = "SELECT id, filename FROM user_files WHERE user_id = %s"
        self.cursor.execute(sql, (user_id,))
        return self.cursor.fetchall()

    def download_file(self, user_id, parent=None):
        files = self.list_user_files(user_id)
        if not files:
            QMessageBox.information(parent, "提示", "暂无上传文件")
            return

        filenames = [f"{fid}: {name}" for fid, name in files]
        selected, ok = QInputDialog.getItem(parent, "选择文件下载", "文件列表：", filenames, 0, False)
        if ok and selected:
            file_id = int(selected.split(":")[0])
            self.cursor.execute("SELECT filename, content FROM user_files WHERE id = %s", (file_id,))
            row = self.cursor.fetchone()
            if row:
                filename, content = row
                save_path, _ = QFileDialog.getSaveFileName(parent, "保存文件", filename)
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(content)
                    QMessageBox.information(parent, "成功", f"文件已保存到：\n{save_path}")

class FileManagerApp(QtWidgets.QWidget):
    def __init__(self, user_id):
        super().__init__()
        uic.loadUi("Assets/UI/CloudFileGUI.ui", self)
        self.user_id = user_id
        self.uploader = FileUploader()

        self.btnUploadFile.clicked.connect(self.upload_file)
        self.btnUploadFolder.clicked.connect(self.upload_folder)
        self.btnDownloadFile.clicked.connect(self.download_file)
        self.statusLabel.setText("欢迎使用文件管理系统")

    def upload_file(self):
        self.uploader.upload_file(self.user_id, self)
        self.statusLabel.setText("文件上传完成")

    def upload_folder(self):
        self.uploader.upload_folder(self.user_id, self)
        self.statusLabel.setText("文件夹上传完成")

    def download_file(self):
        self.uploader.download_file(self.user_id, self)
        self.statusLabel.setText("文件下载完成")
