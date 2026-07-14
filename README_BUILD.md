# AI 图片上色桌面软件打包指南

我已经为你准备好了所有的代码文件，你可以按照以下步骤在你的 **Windows** 电脑上完成打包：

### 1. 准备环境
确保你已经解压了你提供的 `python_env_deoldify.zip`。
在这个目录下打开命令行（PowerShell 或 CMD），安装打包工具：
```bash
python\python.exe -m pip install pyinstaller pyqt5
```

### 2. 放置代码
将我为你生成的 `app.py` 和 `colorize.py` 放在解压后的 `python_env` 根目录下。

### 3. 执行打包命令
在命令行中运行以下命令：
```bash
python\python.exe -m PyInstaller --noconsole --onefile --name "AiColorize" app.py
```
*参数说明：*
- `--noconsole`: 运行时不显示黑色命令行窗口。
- `--onefile`: 将所有内容打包成一个单独的 .exe 文件。
- `--name "AiColorize"`: 指定软件名称。

### 4. 获取软件
打包完成后，你会看到一个 `dist` 文件夹，里面就是生成的 `AiColorize.exe`。

### 注意事项
- **模型文件**：由于模型文件较大，建议你单独下载 `.onnx` 模型。运行软件后，手动选择模型文件即可。
- **运行环境**：生成的 `.exe` 已经包含了必要的 Python 运行时，可以在没有安装 Python 的电脑上运行。
