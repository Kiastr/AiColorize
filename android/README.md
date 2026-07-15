# AiColorize (Android)

漫画 **AI 一键上色** 安卓版，移植自 [Kiastr/AiColorize](https://github.com/Kiastr/AiColorize)（Windows 桌面工具）。

支持 **DeOldify** 与 **DDColor** 两种上色引擎，本地离线推理，无需联网即可处理图片。

## 功能

- 单图 / 批量（文件夹）上色
- 双模型：DeOldify（写实老照片风）、DDColor（二次元漫画风）
- 内置模型下载器（多镜像源 fallback + 进度）
- 硬件加速：NNAPI（GPU/NPU）/ CPU 可选
- 支持 `png / jpg / jpeg / bmp / webp`
- 结果 before / after 对比预览

## 架构

```
Flutter (Dart)              MethodChannel            Android Native (Kotlin)
┌─────────────────┐  "colorize"  ┌──────────────────────────────────────┐
│ UI / 编排        │ ───────────▶ │ ColorizeEngine                        │
│ · 文件选择       │              │  · 预处理  (OpenCV cvtColor/resize)    │
│ · 模型下载(dio)  │ ◀─────────── │  · ONNX 推理 (onnxruntime-android)     │
│ · 批量调度       │   输出路径    │  · 后处理  (OpenCV LAB/blur/merge)     │
│ · 结果预览       │              │ ModelManager (session 缓存+NNAPI/CPU)  │
└─────────────────┘              │ ImageUtils (Bitmap↔Mat↔Tensor)        │
                                  └──────────────────────────────────────┘
```

- 推理：`onnxruntime-android`（官方 Maven，支持 NNAPI）
- 图像：`org.opencv:opencv` AAR（与桌面 `cv2` 数值一致）
- 模型：**零转换** 直接使用原 ONNX 文件

## 构建

### 前置

- Flutter 3.0+ / Dart 2.17+（本仓库按 Flutter 3.0.0 / Dart 2.17 开发验证）
- Android SDK（compileSdk ≥ 31，minSdk 24，含 NDK）
- arm64 安卓设备 / 模拟器

### 步骤

```bash
flutter pub get
# 用 Android Studio 打开 android/ 执行 Gradle Sync（拉取 OpenCV / ORT AAR）
flutter run            # 调试
flutter build apk --release   # 发布
```

### 模型准备（App 内一键完成）

首次使用在主页点「自动下载模型」即可，已内置修正后的可用镜像源：

- **DeOldify**：`deoldify.quant.onnx`（int8 量化版）。原桌面 README 的 DeOldify 链接已失效；量化版体积更小（≈61MB）、移动端更友好，已用 Python 跑通验证。
- **DDColor**：`ddcolor.int8.onnx`（int8 量化，59MB）。原 piddnad github `ddcolor_tiny.onnx` 链接已 404 失效；改用 HuggingFace 镜像的 int8 版，I/O spec 已验证与 pipeline 一致。备用为完整版（934MB）。

## 与桌面版对照

| 项 | 桌面版 | 安卓版 |
|---|---|---|
| UI | PyQt5 | Flutter |
| 推理 EP | CUDA / CPU | NNAPI / CPU |
| 图像 | cv2 + PIL + numpy | OpenCV AAR + Bitmap |
| 打包 | PyInstaller → exe | Gradle → APK |
| 模型 | ONNX | ONNX（不变） |

## 已知限制

- 代码编写与 Dart 静态分析在沙箱完成；**APK 编译（debug + release）已在装有 Android SDK / NDK 的沙箱中验证通过**（见 `VERIFICATION.md` 已验证 #6）。**仅真机端到端运行、数值一致性复测仍待你在本机 / CI 完成。**
- OpenCV 4.x Maven AAR 通过 `OpenCVLoader.initDebug()` 初始化（4.11.0 已确认存在且 `.so` 已打入 APK）；若所用版本移除该 API，请改用 `OpenCVLoader.initLocal()` 或自动加载。
- NNAPI 对部分算子 / 量化模型的支持因设备而异；如推理异常请切换为 CPU。
- bmp 格式解码依赖 `BitmapFactory`，个别机型可能不支持，建议优先 png/jpg/webp。

## 算法移植要点

全部硬约束已由 Python 参考实现实证（`docs/algorithm_verification.md`）：

1. **输入值域相反**：DeOldify 喂 `0–255`、DDColor 喂 `0–1`，分别硬编码分支。
2. **OpenCV float-LAB 真实范围**：L∈[0,100]、a,b∈[−128,127]，必须用 OpenCV 转换，禁手写。
3. **DeOldify 后处理** 刻意把 RGB 当 BGR 转 LAB（对齐原版），保证像素级一致。

详见 `docs/algorithm_verification.md` 与 `docs/reference_impl.py`。
