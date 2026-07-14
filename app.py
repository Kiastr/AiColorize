import sys
import os
import threading
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

class WorkerSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

def colorize_deoldify(input_path: str, output_path: str, model_path: str, device: str = 'cpu') -> None:
    original = Image.open(input_path).convert('RGB')
    orig_w, orig_h = original.size
    original_np = np.array(original)
    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    target_l, _, _ = cv2.split(original_bgr)
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    input_image = cv2.resize(gray_rgb, (256, 256))
    input_data = input_image.astype(np.float32)
    input_data = input_data.transpose((2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    colorized = session.run(None, {input_name: input_data})[0][0]
    colorized = colorized.transpose(1, 2, 0)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)
    colorized = cv2.resize(colorized, (orig_w, orig_h))
    colorized = cv2.GaussianBlur(colorized, (13, 13), 0)
    colorized_lab = cv2.cvtColor(colorized, cv2.COLOR_BGR2LAB)
    _, a, b = cv2.split(colorized_lab)
    result_lab = cv2.merge((target_l, a, b))
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    result.save(output_path)

def colorize_ddcolor_tiny(input_path: str, output_path: str, model_path: str, device: str = 'cpu') -> None:
    _img = Image.open(input_path).convert('RGB')
    img = cv2.cvtColor(np.array(_img), cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    img_norm = (img / 255.0).astype(np.float32)
    orig_l = cv2.cvtColor(img_norm, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_resized = cv2.resize(img_norm, (256, 256))
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    tensor_gray_rgb = img_gray_rgb.transpose((2, 0, 1))
    tensor_gray_rgb = np.expand_dims(tensor_gray_rgb, axis=0).astype(np.float32)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_ab = session.run(None, {input_name: tensor_gray_rgb})[0][0]
    output_ab = output_ab.transpose(1, 2, 0)
    output_ab_resize = cv2.resize(output_ab, (width, height))
    output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)).save(output_path)

class ColorizeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AiColorize - AI 图片上色工具")
        self.setMinimumWidth(500)
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
        
        input_layout = QHBoxLayout()
        self.input_label = QLabel("未选择图片")
        input_btn = QPushButton("选择图片")
        input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_btn)
        layout.addLayout(input_layout)
        
        model_layout = QHBoxLayout()
        self.model_label = QLabel("未选择模型 (.onnx)")
        model_btn = QPushButton("选择模型")
        model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(model_btn)
        layout.addLayout(model_layout)
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("模型类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["deoldify", "ddcolor"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.start_btn = QPushButton("开始上色")
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.start_colorization)
        layout.addWidget(self.start_btn)
        
    def select_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.input_path = path
            self.input_label.setText(os.path.basename(path))
            
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "ONNX Models (*.onnx)")
        if path:
            self.model_path = path
            self.model_label.setText(os.path.basename(path))
            
    def start_colorization(self):
        if not self.input_path or not self.model_path:
            QMessageBox.warning(self, "错误", "请先选择图片和模型文件")
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
        QMessageBox.critical(self, "错误", f"上色失败: {message}")
            
    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.progress.setVisible(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ColorizeApp()
    window.show()
    sys.exit(app.exec_())
