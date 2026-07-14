# AiColorize

AiColorize 是一款基于 AI 技术的桌面图片上色工具，集成了 **DeOldify** 和 **DDColor** 两种先进的上色模型。它可以帮助你轻松地为黑白老照片注入鲜活的色彩。

## 功能特点

- **多模型支持**：支持 DeOldify 和 DDColor (Tiny) 模型，满足不同风格的上色需求。
- **硬件加速**：支持 CPU 和 CUDA (NVIDIA GPU) 推理。
- **简单易用**：直观的图形界面，一键操作。
- **本地运行**：所有处理均在本地完成，保护隐私。

## 快速开始

### 环境准备

本项目推荐使用预打包的 Python 环境：
[下载 Python 环境包](https://github.com/Kiastr/JhentaiSR/releases/download/python-env-v1/python_env_deoldify.zip)

### 打包为 .exe (Windows)

1. 下载上述[环境包](https://github.com/Kiastr/JhentaiSR/releases/download/python-env-v1/python_env_deoldify.zip)并解压。
2. 将本项目中的 `app.py`, `colorize.py`, `build.bat` 放入解压后的文件夹（与 `python` 文件夹同级）。
3. 双击运行 `build.bat`。
4. 等待脚本运行完成，在生成的 `dist` 目录下即可找到 `AiColorize.exe`。

## 模型下载

你需要自行下载 ONNX 格式的模型文件：
- [DeOldify ONNX](https://huggingface.co/AXERA-TECH/DeOldify)
- [DDColor ONNX](https://huggingface.co/facefusion/models-3.0.0/blob/main/ddcolor.onnx)

## 鸣谢

- [DeOldify](https://github.com/jantic/DeOldify)
- [DDColor](https://github.com/piddnad/DDColor)
- [JhentaiSR](https://github.com/Kiastr/JhentaiSR)
