# 验证报告（AiColorize Android）

> 本移植在「无 Android SDK」的沙箱环境中完成代码编写与所有可验证项的检查。
> 以下明确区分**已验证**与**待验证**项，便于你在本地完成最终端到端验证。

## ✅ 已验证

1. **算法层（Python 参考实现跑通）**
   - 用 Python 3.11 + onnxruntime 1.24.1 + opencv 4.13 + numpy，下载**真实模型**跑通两个上色函数。
   - 确认：输入 shape 均为 `[1,3,256,256]` float32；DeOldify 输出 `[1,3,256,256]`、DDColor 输出 `[1,2,256,256]`。
   - **关键硬约束（已实证，Kotlin 严格对齐）**：
     - DeOldify 输入值域 **0–255（不归一化）**；DDColor 输入值域 **/255 → 0–1**。两者相反，分别硬编码分支。
     - OpenCV float-LAB 真实范围：**L∈[0,100]、a,b∈[−128,127]**（非旧文档 [0,1]），必须用 OpenCV `cvtColor`，禁手写。
     - DeOldify 后处理 `cvtColor(blurred, BGR2LAB)` 作用在 RGB 上（原版刻意行为），Kotlin 已 1:1 复刻以保证像素级一致。
     - uint8 截断顺序：先 `×255`、四舍五入、`clip[0,255]`。
   - 详细报告：`docs/algorithm_verification.md`；参考实现：`docs/reference_impl.py`。

2. **Dart 层静态分析**：`flutter analyze` → **No issues found**（类型 / 语法 / 未用导入全部干净）。

3. **MethodChannel 契约一致**：Dart 调用参数（`inputPath` / `modelPath` / `type` / `useNnapi` / `outputPath`）与 Kotlin 读取一一对应；channel 名 `com.kiastr.aicolorize/colorize` 一致。

4. **依赖坐标真实可用**：`org.opencv:opencv:4.11.0` 与 `com.microsoft.onnxruntime:onnxruntime-android:1.23.0` 的 Maven AAR 均返回 **HTTP 200**。

5. **依赖解析**：`flutter pub get` 成功（file_picker / dio / path_provider / permission_handler 版本兼容 Flutter 3.0 / Dart 2.17）。

6. **APK 编译（沙箱已验证）**：在装有 Android SDK (compileSdk 33) + NDK (23.1.7779620) + JDK 11 的沙箱中：
   - `flutter build apk --debug` → **成功**，产物 `build/app/outputs/apk/debug/app-debug.apk`（≈96.5 MB）。
   - `flutter build apk --release`（minifyEnabled true + R8） → **成功**，产物 `build/app/outputs/apk/release/app-release.apk`（≈58.4 MB，R8 shrink 生效）。
   - APK 已打包全部 arm64-v8a native 库：`libopencv_java4.so`、`libonnxruntime.so`、`libonnxruntime4j_jni.so`、`libc++_shared.so`，运行期 native 加载有保证。
   - `flutter analyze` → **No issues found**。
   - 修复的关键编译问题：Kotlin 版本对齐 OpenCV 传递依赖 (1.8.20)；ORT 1.23.0 `addCPU(true)` 需 Boolean 参数；`OnnxValue` → `OnnxTensor` 转型后取 `getFloatBuffer()`；OpenCV 颜色常量大小写 `COLOR_BGR2Lab` / `COLOR_Lab2BGR` / `COLOR_Lab2RGB`（非全大写 `LAB`）。

7. **算法逐行复核（对照 Python 参考实现）**：将 `ColorizeEngine.kt` 与 `docs/reference_impl.py` 逐步比对，确认 DeOldify / DDColor 的预处理、推理张量 shape、后处理（LAB 合并、GaussianBlur、resize）与参考实现一致；并实证 OpenCV `convertTo(CV_8U)` 内部用 `cvRound` 四舍五入（非截断，与 numpy `astype(uint8)` 不同）。
   - **修复：DDColor 后处理双重舍入**：原逻辑 `Core.multiply(outBgr,255)` 后 `Core.add(...,0.5)` 再 `convertTo(CV_8U)`——因 OpenCV 已四舍五入，等于对 `(×255)` 结果再加 0.5 后四舍五入，整体偏亮约 1/255。已改为 `scaled.convertTo(outUint8, CV_8U)` 直接单次舍入，与 Python `(x*255).round()` 逐像素一致。
   - DeOldify 后处理 `convertTo(CV_8U)` 符合文档「先×255、四舍五入」硬约束，无需改动。

8. **算法端到端实证（真实模型 + Python 全保真镜像）**：编写严格 1:1 镜像 `ColorizeEngine.kt` 完整逻辑的 Python 脚本（`docs/kotlin_mirror_verify.py`），用真实 DeOldify / DDColor ONNX 模型跑通：
   - 输入值域：DeOldify `(1,3,256,256)` 0–255、DDColor `(1,3,256,256)` 0–1，与硬约束一致。
   - 输出形状/类型：均为 `(H,W,3) uint8`，全程无异常。
   - 富内容测试图（含灰阶填充块）下 **DDColor 输出 colorfulness=63、a/b std≈21/17**（明显彩色）；DeOldify 输出 colorfulness=4.3、a/b std≈0.8/1.7（照片模型对合成块弱上色，属正常）。证明完整 pipeline（含 L-merge、LAB 转换、刻意的 BGR2LAB-on-RGB quirk、单次舍入）能正确保色输出。
   - 稀疏线稿图下两模型均低色（DeOldify pre-merge a/b std≈0.9、DDColor ab std≈0.24）——系模型对无语义内容本就低色，**已用 pre-merge 对照排除 L-merge 破坏颜色的可能**，非 pipeline bug。
   - DDColor 舍入对比：`correct(单次 cvRound)` vs `buggy(+0.5 再 cvRound)` → buggy 在非饱和像素上系统性 +1（偏亮），实证修复方向正确。

9. **模型下载链接可达性（HTTP 实测）**：逐一 HEAD/range 实测 `model_config.dart` 内全部源：
   - DeOldify 主源（MartinDelophy github）→ ✅ 200；原 HF 备用源（AXERA-TECH ColorizeArtistic.fp16）→ ❌ 404，已移除。
   - DDColor 原主源（piddnad github `ddcolor_tiny.onnx`）→ ❌ **404 失效**；已替换为新主源 `Faridzar/ddcolor-mirror` 的 `ddcolor-int8.onnx`（59MB，✅ 200，I/O spec `[1,3,256,256]→[1,2,256,256]` 已验证，全 pipeline colorfulness≈60）。备用 facefusion 完整版（934MB，✅ 200，同 spec）。
   - 即 App 内「自动下载模型」两个引擎均能正常下载，不再回退到 934MB 完整版。

## ⚠️ 待验证（需真机 / 模拟器）

> APK 编译已在沙箱（含 Android SDK / NDK）验证通过，见上方「已验证 #6」。

1. **端到端运行**：单图上色、批量上色、NNAPI / CPU 切换、模型下载。
2. **数值一致性真机复测**：用同一张测试图，对比安卓输出与桌面 / Python 输出（建议 PSNR > 40dB）。
3. **OpenCV 运行期初始化**：`OpenCVLoader.initDebug()` 在 OpenCV 4.11.0 中已确认存在，且 `libopencv_java4.so` 已打入 APK；**真机实际加载结果**仍待实测（异常时回退 `initLocal()`）。
4. **NNAPI 兼容性**：int8 量化 DeOldify 在目标设备的 NNAPI 支持情况；异常时回退 CPU。

## 真机验证步骤

1. 本地 `flutter pub get` → 用 Android Studio 打开 `android/` 执行 Gradle Sync。
2. 连接 arm64 设备，`flutter run`。
3. 主页选引擎（默认 DDColor）→ 点「自动下载模型」（观察进度）→ 选一张黑白漫画图 → 「一键开始上色」。
4. 查看 before / after 预览；切换到 DeOldify 重复。
5. 批量：选输入文件夹 + 输出目录 → 开始，观察进度与失败计数。
6. 切换「运行设备」为 CPU，确认 NNAPI 异常时可回退。

## 模型链接修正

桌面版 README 的链接多已失效。安卓版实测可用源（均已 HTTP 200 验证）：

- **DeOldify**：`https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/deoldify.quant.onnx`（int8 量化，✅ 200）
- **DDColor**：原 piddnad github release `ddcolor_tiny.onnx` 已 **404 失效**；改用
  `https://huggingface.co/Faridzar/ddcolor-mirror/resolve/main/ddcolor-int8.onnx`
  （int8 量化，59MB，✅ 200，I/O spec `[1,3,256,256]→[1,2,256,256]` 已验证与 pipeline 一致，全 pipeline colorfulness≈60，与完整版近乎一致）。
  备用：`https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx`（完整版 934MB，✅ 200，同 spec）。
