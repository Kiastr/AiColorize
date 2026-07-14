import sys
import os
import threading
import requests
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# 模型下载配置
MODEL_CONFIGS = {
    "deoldify": {
        "filename": "deoldify.onnx",
        "urls": [
            "https://huggingface.co/AXERA-TECH/DeOldify/resolve/main/ColorizeArtistic.fp16.onnx",
            "https://mirror.ghproxy.com/https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/ColorizeArtistic_gen.onnx"
        ]
    },
    "ddcolor": {
        "filename": "ddcolor.onnx",
        "urls": [
            "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx",
            "https://mirror.ghproxy.com/https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_tiny.onnx"
        ]
    }
}

class WorkerSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

class DownloadDialog(QDialog):
    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"正在下载模型: {model_name}")
        self.setFixedSize(400, 120)
        layout = QVBoxLayout(self)
        self.label = QLabel(f"正在从镜像源下载 {model_name} 模型...")
        layout.addWidget(self.label)
        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)
        self.is_cancelled = False

    def update_progress(self, val):
        self.pbar.setValue(val)

class ColorizeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AiColorize - AI 图片上色工具")
        self.setMinimumWidth(550)
        self.input_path = ""
        self.model_path = ""
        self.signals = WorkerSignals()
        self.signals.finished.connect(self.on_finished)
        self.signals.error.connect(self.on_error)
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 图片选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("第一步：选择需要上色的黑白图片")
        self.input_label.setStyleSheet("color: gray;")
        input_btn = QPushButton("选择图片")
        input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_label, 1)
        input_layout.addWidget(input_btn)
        layout.addLayout(input_layout)
        
        # 模型选择与下载
        model_layout = QHBoxLayout()
        self.model_label = QLabel("第二步：选择或下载 AI 模型")
        self.model_label.setStyleSheet("color: gray;")
        model_btn = QPushButton("选择本地模型")
        model_btn.clicked.connect(self.select_model)
        download_btn = QPushButton("自动下载模型")
        download_btn.setStyleSheet("background-color: #e1f5fe;")
        download_btn.clicked.connect(self.auto_download_model)
        model_layout.addWidget(self.model_label, 1)
        model_layout.addWidget(model_btn)
        model_layout.addWidget(download_btn)
        layout.addLayout(model_layout)
        
        # 配置区
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("模型类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["deoldify", "ddcolor"])
        config_layout.addWidget(self.type_combo)
        
        config_layout.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        config_layout.addWidget(self.device_combo)
        layout.addLayout(config_layout)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # 开始按钮
        self.start_btn = QPushButton("立即开始上色")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("font-weight: bold; font-size: 14px; background-color: #4caf50; color: white;")
        self.start_btn.clicked.connect(self.start_colorization)
        layout.addWidget(self.start_btn)
        
        layout.addWidget(QLabel("提示：如果上色失败，请尝试切换模型或运行设备。"))

    def select_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.input_path = path
            self.input_label.setText(os.path.basename(path))
            self.input_label.setStyleSheet("color: black;")
            
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "ONNX Models (*.onnx)")
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))
            self.model_label.setStyleSheet("color: black;")

    def auto_download_model(self):
        model_type = self.type_combo.currentText()
        config = MODEL_CONFIGS[model_type]
        save_path = os.path.join(os.getcwd(), config["filename"])
        
        if os.path.exists(save_path):
            reply = QMessageBox.question(self, "提示", f"检测到模型 {config['filename']} 已存在，是否重新下载？", 
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                self.model_path = save_path
                self.model_label.setText(config["filename"])
                return

        dialog = DownloadDialog(model_type, self)
        
        def download_task():
            success = False
            for url in config["urls"]:
                try:
                    response = requests.get(url, stream=True, timeout=10)
                    total = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    with open(save_path, 'wb') as f:
                        for data in response.iter_content(chunk_size=4096):
                            if dialog.is_cancelled: break
                            downloaded += len(data)
                            f.write(data)
                            if total > 0:
                                dialog.update_progress(int(100 * downloaded / total))
                    if not dialog.is_cancelled:
                        success = True
                        break
                except:
                    continue
            
            if success:
                self.model_path = save_path
                self.model_label.setText(config["filename"])
                QMessageBox.information(self, "成功", "模型下载完成！")
                dialog.accept()
            else:
                QMessageBox.warning(self, "失败", "所有镜像源均下载失败，请检查网络。")
                dialog.reject()

        threading.Thread(target=download_task, daemon=True).start()
        dialog.exec_()

    def start_colorization(self):
        if not self.input_path:
            QMessageBox.warning(self, "错误", "请先选择一张图片")
            return
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "错误", "请先选择或下载模型文件")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(self, "保存上色后的图片", "colorized_" + os.path.basename(self.input_path), "Images (*.png *.jpg *.jpeg)")
        if not output_path:
            return
            
        self.start_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        threading.Thread(target=self.run_process, args=(output_path,), daemon=True).start()
        
    def run_process(self, output_path):
        try:
            model_type = self.type_combo.currentText()
            device = self.device_combo.currentText()
            from colorize import colorize_deoldify, colorize_ddcolor_tiny
            if model_type == 'deoldify':
                colorize_deoldify(self.input_path, output_path, self.model_path, device)
            else:
                colorize_ddcolor_tiny(self.input_path, output_path, self.model_path, device)
            self.signals.finished.emit(output_path)
        except Exception as e:
            self.signals.error.emit(str(e))
            
    def on_finished(self, path):
        self.reset_ui()
        QMessageBox.information(self, "成功", f"图片已保存至: {path}")
        
    def on_error(self, message):
        self.reset_ui()
        QMessageBox.critical(self, "错误", f"上色失败: {message}\n\n建议：检查模型类型是否匹配，或尝试使用 CPU 运行。")
            
    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.progress.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorizeApp()
    window.show()
    sys.exit(app.exec_())
