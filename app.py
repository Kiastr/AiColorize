import sys
import os
import threading
import requests
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar, QMessageBox, QDialog, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal, QObject

# --- 核心上色逻辑合并 ---

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

# --- GUI 相关 ---

MODEL_CONFIGS = {
    "deoldify": {
        "filename": "deoldify.onnx",
        "sources": [
            {"name": "Hugging Face", "url": "https://huggingface.co/AXERA-TECH/DeOldify/resolve/main/ColorizeArtistic.fp16.onnx"},
            {"name": "ghproxy 镜像", "url": "https://mirror.ghproxy.com/https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/ColorizeArtistic_gen.onnx"}
        ]
    },
    "ddcolor": {
        "filename": "ddcolor.onnx",
        "sources": [
            {"name": "Hugging Face", "url": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx"},
            {"name": "ghproxy 镜像", "url": "https://mirror.ghproxy.com/https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_tiny.onnx"}
        ]
    }
}

class WorkerSignals(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int) # current, total

class DownloadDialog(QDialog):
    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"正在下载模型: {model_name}")
        self.setFixedSize(450, 150)
        layout = QVBoxLayout(self)
        self.label = QLabel("正在初始化下载...")
        layout.addWidget(self.label)
        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)
        self.source_label = QLabel("当前源: 等待中")
        layout.addWidget(self.source_label)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.cancel_download)
        layout.addWidget(self.cancel_btn)
        self.is_cancelled = False

    def update_progress(self, val):
        self.pbar.setValue(val)

    def update_status(self, text, source_name=""):
        self.label.setText(text)
        if source_name:
            self.source_label.setText(f"当前源: {source_name}")

    def cancel_download(self):
        self.is_cancelled = True
        self.reject()

class ColorizeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AiColorize v1.2.0 - 漫画 AI 批量上色工具")
        self.setMinimumWidth(600)
        self.input_path = ""
        self.output_dir = ""
        self.model_path = ""
        self.is_batch = False
        
        self.signals = WorkerSignals()
        self.signals.finished.connect(self.on_finished)
        self.signals.error.connect(self.on_error)
        self.signals.progress.connect(self.update_batch_progress)
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_group = QButtonGroup(self)
        self.radio_single = QRadioButton("单图处理")
        self.radio_batch = QRadioButton("批量处理 (文件夹)")
        self.radio_single.setChecked(True)
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_batch)
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_batch)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        self.radio_single.toggled.connect(self.toggle_mode)

        # 输入选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel("请选择需要上色的图片或文件夹")
        self.input_label.setStyleSheet("color: gray;")
        self.input_btn = QPushButton("选择图片")
        self.input_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_label, 1)
        input_layout.addWidget(self.input_btn)
        layout.addLayout(input_layout)
        
        # 输出目录选择 (仅批量模式显示)
        self.output_widget = QWidget()
        output_layout = QHBoxLayout(self.output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_label = QLabel("请选择输出保存文件夹")
        self.output_label.setStyleSheet("color: gray;")
        output_btn = QPushButton("选择输出目录")
        output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_label, 1)
        output_layout.addWidget(output_btn)
        layout.addWidget(self.output_widget)
        self.output_widget.setVisible(False)

        # 模型选择与下载
        model_layout = QHBoxLayout()
        self.model_label = QLabel("选择或下载 AI 模型 (推荐 DDColor)")
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
        config_layout.addWidget(QLabel("模型引擎:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["deoldify", "ddcolor"])
        self.type_combo.setCurrentText("ddcolor")
        config_layout.addWidget(self.type_combo)
        
        config_layout.addWidget(QLabel("运行设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        config_layout.addWidget(self.device_combo)
        layout.addLayout(config_layout)
        
        # 进度条
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # 开始按钮
        self.start_btn = QPushButton("一键开始上色")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4caf50; color: white;")
        self.start_btn.clicked.connect(self.start_process)
        layout.addWidget(self.start_btn)
        
        layout.addWidget(QLabel("支持格式: png, jpg, jpeg, bmp, webp"))

    def toggle_mode(self):
        self.is_batch = self.radio_batch.isChecked()
        if self.is_batch:
            self.input_btn.setText("选择文件夹")
            self.output_widget.setVisible(True)
        else:
            self.input_btn.setText("选择图片")
            self.output_widget.setVisible(False)
        self.input_path = ""
        self.input_label.setText("请选择需要上色的图片或文件夹")
        self.input_label.setStyleSheet("color: gray;")

    def select_input(self):
        if self.is_batch:
            path = QFileDialog.getExistingDirectory(self, "选择包含黑白漫画的文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择黑白图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        
        if path:
            self.input_path = path
            self.input_label.setText(path)
            self.input_label.setStyleSheet("color: black;")

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir = path
            self.output_label.setText(path)
            self.output_label.setStyleSheet("color: black;")

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
            for source in config["sources"]:
                if dialog.is_cancelled: break
                url = source["url"]
                name = source["name"]
                dialog.update_status(f"正在尝试从 {name} 下载...", name)
                try:
                    response = requests.get(url, stream=True, timeout=15)
                    response.raise_for_status()
                    total = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    with open(save_path, 'wb') as f:
                        for data in response.iter_content(chunk_size=8192):
                            if dialog.is_cancelled: break
                            downloaded += len(data)
                            f.write(data)
                            if total > 0:
                                dialog.update_progress(int(100 * downloaded / total))
                    if not dialog.is_cancelled:
                        success = True
                        break
                except Exception as e:
                    print(f"Source {name} failed: {e}")
                    continue
            
            if success:
                self.model_path = save_path
                self.model_label.setText(config["filename"])
                QMessageBox.information(self, "成功", f"模型 {model_type} 下载完成！")
                dialog.accept()
            elif not dialog.is_cancelled:
                QMessageBox.warning(self, "失败", "所有镜像源均下载失败，请检查网络或手动下载。")
                dialog.reject()

        threading.Thread(target=download_task, daemon=True).start()
        dialog.exec_()

    def start_process(self):
        if not self.input_path:
            QMessageBox.warning(self, "错误", "请先选择输入文件或文件夹")
            return
        if not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, "错误", "请先准备好模型文件")
            return
        if self.is_batch and not self.output_dir:
            QMessageBox.warning(self, "错误", "批量模式下请先选择输出目录")
            return
            
        if not self.is_batch:
            # 单图模式选择保存位置
            ext = os.path.splitext(self.input_path)[1]
            output_path, _ = QFileDialog.getSaveFileName(self, "保存上色图片", "colorized_" + os.path.basename(self.input_path), f"Images (*{ext})")
            if not output_path: return
            self.run_batch_logic([self.input_path], [output_path])
        else:
            # 批量模式逻辑
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
            files = [f for f in os.listdir(self.input_path) if f.lower().endswith(valid_exts)]
            if not files:
                QMessageBox.warning(self, "提示", "文件夹内没有找到支持的图片文件")
                return
            
            input_files = [os.path.join(self.input_path, f) for f in files]
            output_files = [os.path.join(self.output_dir, "colorized_" + f) for f in files]
            self.run_batch_logic(input_files, output_files)

    def run_batch_logic(self, input_list, output_list):
        self.start_btn.setEnabled(False)
        self.progress.setRange(0, len(input_list))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.progress_label.setText(f"准备处理 0 / {len(input_list)}")
        
        threading.Thread(target=self.process_thread, args=(input_list, output_list), daemon=True).start()

    def process_thread(self, input_list, output_list):
        model_type = self.type_combo.currentText()
        device = self.device_combo.currentText()
        total = len(input_list)
        
        try:
            for i, (inp, outp) in enumerate(zip(input_list, output_list)):
                self.signals.progress.emit(i + 1, total)
                if model_type == 'deoldify':
                    colorize_deoldify(inp, outp, self.model_path, device)
                else:
                    colorize_ddcolor_tiny(inp, outp, self.model_path, device)
            
            self.signals.finished.emit(self.output_dir if self.is_batch else os.path.dirname(output_list[0]))
        except Exception as e:
            self.signals.error.emit(str(e))

    def update_batch_progress(self, current, total):
        self.progress.setValue(current)
        self.progress_label.setText(f"正在处理: {current} / {total}")

    def on_finished(self, path):
        self.reset_ui()
        QMessageBox.information(self, "处理完成", f"所有任务已完成！\n保存路径: {path}")
        
    def on_error(self, message):
        self.reset_ui()
        QMessageBox.critical(self, "错误", f"处理中断: {message}")
            
    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.progress_label.setText("")

if __name__ == "__main__":
    # 解决高分屏缩放问题
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = ColorizeApp()
    window.show()
    sys.exit(app.exec_())
