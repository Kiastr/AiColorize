# AiColorize

为漫画设计的 AI 一键上色桌面工具，支持批量处理整本漫画。带GUI界面鼠标点击即可操作，本地处理无需联网

## ✨ 功能亮点

- **批量上色**：一次选择整个文件夹，自动处理所有漫画图片（支持 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp` 格式），并将上色结果保存到指定输出目录。
- **双模型支持**：内置 DeOldify 和 DDColor 两种 AI 上色引擎，满足不同风格的上色需求。
- **内置下载器**：无需手动寻找模型，点击按钮自动下载，支持多镜像源，下载时显示进度和当前源名称。
- **开箱即用**：下载 `AiColorize.exe` 直接运行，无需安装 Python 或任何复杂的环境。
- **GPU 加速**：支持 NVIDIA GPU (CUDA) 加速，大幅提升处理速度。
- **简单易用**：直观的图形用户界面，一键操作，无需命令行知识。

## 🚀 使用方法（以windows为例）

1.  **下载程序**：前往 [Releases 页面](https://github.com/Kiastr/AiColorize/releases) 下载最新的 `AiColorize.exe`。
2.  **运行程序**：双击下载的 `AiColorize.exe` 启动工具。
3.  **下载模型**：点击主界面上的 "自动下载模型" 按钮，程序将自动下载所需的 AI 模型（推荐使用 DDColor）。
4.  **选择模式**：根据需求选择 "单图处理" 或 "批量处理 (文件夹)" 模式。
5.  **选择输入/输出**：
    *   **单图模式**：点击 "选择图片" 按钮选择一张黑白图片。
    *   **批量模式**：点击 "选择文件夹" 按钮选择包含黑白漫画图片的文件夹，并点击 "选择输出目录" 按钮指定上色结果的保存位置。
6.  **开始上色**：点击 "一键开始上色" 按钮，等待处理完成。
7.  **查看结果**：处理完成后，程序会弹出提示，告知您上色结果的保存路径。

## 🎨 模型说明

-   **DeOldify**：适合老照片风格，色彩自然，偏向写实。
-   **DDColor**：适合漫画/动漫风格，色彩鲜艳，效果更符合二次元内容。

## 🌐 模型下载镜像源列表

如果您需要手动下载模型，可以使用以下链接：

-   **DeOldify 模型 (`deoldify.onnx`)**
    *   Hugging Face 官方源：[https://huggingface.co/AXERA-TECH/DeOldify/resolve/main/ColorizeArtistic.fp16.onnx](https://huggingface.co/AXERA-TECH/DeOldify/resolve/main/ColorizeArtistic.fp16.onnx)
    *   ghproxy 镜像源（国内加速）：[https://mirror.ghproxy.com/https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/ColorizeArtistic_gen.onnx](https://mirror.ghproxy.com/https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/ColorizeArtistic_gen.onnx)

-   **DDColor 模型 (`ddcolor.onnx`)**
    *   Hugging Face 官方源：[https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx](https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx)
    *   ghproxy 镜像源（国内加速）：[https://mirror.ghproxy.com/https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_tiny.onnx](https://mirror.ghproxy.com/https://github.com/piddnad/DDColor/releases/download/v1.0/ddcolor_tiny.onnx)

## 🛠️ 构建说明 (面向开发者)

### 通过 GitHub Actions 自动编译

本项目配置了 GitHub Actions，当代码 `push` 到 `master` 分支或创建 `tag` (例如 `v1.2.0`) 时，会自动在 `windows-latest` 环境下编译 `AiColorize.exe` 并作为 `artifact` 上传。如果是 `tag` 触发，还会自动创建 GitHub Release 并附加 `exe` 文件。

工作流文件：`.github/workflows/build.yml`

### 本地手动编译

1.  **环境准备**：
    *   下载 [Python 环境包](https://github.com/Kiastr/JhentaiSR/releases/download/python-env-v1/python_env_deoldify.zip) 并解压。
    *   确保已安装 `git` 并克隆本项目到本地。
2.  **文件放置**：将本项目中的 `app.py` 和 `build.bat` 放入解压后的 Python 环境文件夹（与 `python` 文件夹同级）。
3.  **运行编译脚本**：双击运行 `build.bat`。
4.  **获取可执行文件**：等待脚本运行完成，在生成的 `dist` 目录下即可找到 `AiColorize.exe`。

## 🙏 鸣谢

-   [DeOldify](https://github.com/jantic/DeOldify)
-   [DDColor](https://github.com/piddnad/DDColor)
-   [JhentaiSR](https://github.com/Kiastr/JhentaiSR)
