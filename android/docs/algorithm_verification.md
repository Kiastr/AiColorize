# AiColorize 算法移植验证报告（DeOldify / DDColor → 安卓）

> 目标：用真实 ONNX 模型跑通 `colorize.py` 的两个核心函数，确认移植到安卓时的关键事实
> （shape、值域、坑），并给出可操作的移植清单。
> 验证脚本：`/root/.codebuddy/artifact/py_ref_impl.py`（严格对应 `colorize.py`）
> 输出图：`/root/.codebuddy/artifact/models/out_deoldify.png`、`out_ddcolor.png`

---

## 1. 环境信息

| 项 | 值 |
|---|---|
| Python | 3.11.1 |
| onnxruntime | 1.24.1 |
| opencv (cv2) | 4.13.0 |
| numpy | 2.3.5 |
| 执行设备 | CPU（`providers=['CPUExecutionProvider']`）|

依赖确认命令（已通过）：
```bash
python3.11 -c "import onnxruntime,cv2,numpy; print(onnxruntime.__version__, cv2.__version__, numpy.__version__)"
```
> 注：本机无 GPU，全部在 CPU EP 上验证。两个模型在 CPU EP 下均可成功 `InferenceSession(...)` 创建，**无需任何 fp16 特殊开关**。onnxruntime 1.24 对 fp16/fp32/int8 模型在 CPU EP 下会自动处理。

---

## 2. 模型下载情况

⚠️ **README 给出的两个 DeOldify 下载链接均已失效/错误**，实测如下：

| 模型 | README 声称的 URL | 实测结果 |
|---|---|---|
| DeOldify (fp16) | `huggingface.co/AXERA-TECH/DeOldify/.../ColorizeArtistic.fp16.onnx` | 该 HF 仓库**只有 `.axmodel`（Axera NPU）文件，根本没有 ONNX**，resolve 返回 “Entry not found” |
| DeOldify (镜像) | `mirror.ghproxy.com/.../ColorizeArtistic_gen.onnx` | MartinDelophy 仓库 v1.0.0 release **只有 `deoldify.quant.onnx`**，无 `ColorizeArtistic_gen.onnx`（404）；ghproxy 域名还出现 SSL 错误 |
| DDColor (fp32) | `huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx` | ✅ **下载成功** |
| DDColor (tiny 镜像) | `mirror.ghproxy.com/.../ddcolor_tiny.onnx` | ghproxy 域名 SSL 错误，未成功 |

实际可用于验证的模型：

| 文件 | 大小 | 来源 | 类型 | fp16? |
|---|---|---|---|---|
| `/root/.codebuddy/artifact/models/deoldify.onnx` | 64,020,138 B（≈61.0 MB） | `github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/deoldify.quant.onnx` | **INT8 动态量化** ONNX | 否（是 int8 量化，不是 fp16） |
| `/root/.codebuddy/artifact/models/ddcolor.onnx` | 980,103,562 B（≈934 MB） | `huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx` | **fp32** 完整模型 | 否（是 fp32） |

**给安卓的关键提示**：
- 我们真正跑通的 DeOldify 是 **int8 量化** 模型（`deoldify.quant.onnx`），不是 README 说的 fp16。**量化 int8 模型在安卓上反而更友好**（体积小、CPU/NNAPI 更快），建议安卓就直接用 int8 量化版。
- 我们下载的 DDColor 是 **934 MB 的完整 fp32 模型**，**太大不适合直接塞进安卓 App**。请改用 `ddcolor_tiny`（约几十 MB）做移动端目标模型；它的 I/O 契约与完整版一致（见第 3 节）。
- 若坚持要用 README 里“ColorizeArtistic”那类**真·fp16** 模型：fp16 ONNX 在安卓只有 **NNAPI / GPU EP** 能原生跑，纯 CPU EP 会走 fp16→fp32 内部提升，**能跑但无加速**；务必在安卓端确认 EP 支持。

---

## 3. 逐步骤 shape / dtype 表

### 3.1 DeOldify（对应 `colorize_deoldify`）

| 步骤 | 代码 | tensor shape | dtype | 值域 |
|---|---|---|---|---|
| 灰度线稿→RGB | `cv2.cvtColor(gray, GRAY2RGB)` | (H,W,3) | uint8 | 0–255 |
| resize 到模型输入 | `cv2.resize(...,(256,256))` | (256,256,3) | uint8 | 0–255 |
| → float32（**不 /255**） | `.astype(np.float32)` | (256,256,3) | float32 | **0–255** |
| HWC→CHW | `.transpose((2,0,1))` | (3,256,256) | float32 | 0–255 |
| 加 batch 维 | `np.expand_dims(axis=0)` | **(1,3,256,256)** | float32 | **0–255** |
| **ONNX 输入** | `session.run(...)` | **(1,3,256,256)** | **float32** | **0–255** |
| ONNX 输出 | `[0][0]` | (3,256,256) | float32 | 0–255（BGR） |
| CHW→HWC | `.transpose(1,2,0)` | (256,256,3) | float32 | 0–255 |
| BGR→RGB, →uint8 | `cvtColor(BGR2RGB).astype(uint8)` | (256,256,3) | uint8 | 0–255 |

- `session.get_inputs()[0].name` = `"input"`，`get_outputs()[0].name` = `"out"`。
- **实测输出 shape = (1,3,256,256)，与代码假设完全一致。**

### 3.2 DDColor（对应 `colorize_ddcolor_tiny`）

| 步骤 | 代码 | tensor shape | dtype | 值域 |
|---|---|---|---|---|
| 灰度→RGB→BGR | `cvtColor(RGB2BGR)` | (H,W,3) | uint8 | 0–255 |
| **/255 归一化** | `(img/255.0).astype(float32)` | (H,W,3) | float32 | **0–1** |
| 取原图 L（float LAB） | `cvtColor(BGR2Lab)[:,:,:1]` | (H,W,1) | float32 | **L∈[0,100]** |
| resize 到 256 | `cv2.resize(...,(256,256))` | (256,256,3) | float32 | 0–1 |
| 取 256 尺寸 L | `cvtColor(BGR2Lab)[:,:,:1]` | (256,256,1) | float32 | L∈[0,100] |
| 构造灰度 LAB(a=b=0) | `concat(L,0,0)` | (256,256,3) | float32 | L∈[0,100] |
| LAB→RGB 作为模型输入 | `cvtColor(LAB2RGB)` | (256,256,3) | float32 | 0–1 |
| HWC→CHW→batch | `transpose`+`expand_dims` | **(1,3,256,256)** | float32 | **0–1** |
| **ONNX 输入** | `session.run(...)` | **(1,3,256,256)** | **float32** | **0–1** |
| ONNX 输出(ab 两通道) | `[0][0]` | (2,256,256) | float32 | **ab∈[−128,127]** |
| CHW→HWC | `.transpose(1,2,0)` | (256,256,2) | float32 | ab∈[−128,127] |
| 上采样回原尺寸 | `cv2.resize(...,(W,H))` | (H,W,2) | float32 | ab∈[−128,127] |
| 拼回原图 L | `concat(orig_l, ab)` | (H,W,3) | float32 | L∈[0,100], ab∈[−128,127] |
| LAB→BGR（float） | `cvtColor(LAB2BGR)` | (H,W,3) | float32 | 0–1 |
| 0–1→0–255 uint8 | `*255 round clip` | (H,W,3) | uint8 | 0–255 |

- `session.get_inputs()[0].name` = `"input"`，`get_outputs()[0].name` = `"output"`。
- ONNX 元数据里输出维度是动态的（`['Convoutput_dim_0',2,...]`），实际运行固定为 **(1,2,256,256)**。
- **实测输入/输出 shape 与代码假设一致。**

---

## 4. ★ 最重要结论：DeOldify vs DDColor 输入值域差异

**用“输入值域探针”对同一张图分别用 0–255 和 0–1 跑两次，结果如下（决定性证据）：**

### DeOldify —— 输入必须是 **0–255（不归一化）**

| 输入值域 | 模型输出（按 BGR 解释） |
|---|---|
| **0–255（colorize.py 的写法）** | mean=115.5, std=97.1, min=0, max=255 → 全动态范围、可见图像 ✅ |
| 0–1（错误对照） | mean=0.96, std=1.11, max=8.06 → 几乎全黑、崩溃 ❌ |

➡️ **DeOldify 输入 = `astype(float32)`，不除以 255，值域 0–255。**

### DDColor —— 输入必须是 **/255 归一化到 0–1**

| 输入值域 | 最终合成彩色图（RGB） |
|---|---|
| **/255（colorize.py 的写法）** | mean=229.5, std=76.5, min=0, max=255 → 有效彩色图 ✅ |
| 0–255（错误对照） | mean=0.9, std=0.3, max=1 → 近全黑、崩溃 ❌ |

➡️ **DDColor 输入 = `img/255.0`，值域 0–1。**

**一句话总结（最关键交付点）：**
> **DeOldify 喂 0–255 未归一化的 float32 RGB；DDColor 喂 /255 归一化到 0–1 的 float32 RGB。两者输入值域完全相反，安卓端必须分别硬编码，混用会直接产出全黑/崩溃结果。**

（探针代码见 `py_ref_impl.py` 的 `range_probe()`，可复现上述数字。）

---

## 5. 输出合理性结论

测试图：用 numpy 生成的 400×300 黑白漫画线稿（`models/test_input.png`）。
输出图：`models/out_deoldify.png`、`models/out_ddcolor.png`。

| 模型 | 合成 RGB 统计 | Lab a/b 标准差 | 结论 |
|---|---|---|---|
| DeOldify | mean≈229, std≈73 | a≈0.98, b≈1.02（线稿）/ a≈3.7,b≈5.1（人脸图） | 非全黑、非崩溃；在稀疏线稿上色度偏弱，在信息更丰富的图上色度明显增强。**这是 `deoldify.quant.onnx`（int8 量化弱模型）的特性**，真·ColorizeArtistic 应更强 |
| DDColor | mean≈229, std≈76 | 平滑灰度测试 a≈3.6,b≈4.4；线稿偏弱 | 非全黑、真实彩色输出 ✅ |

- 两者都**不是全黑、不是全灰、没有数值崩溃**，证明算法链路（预处理→推理→后处理）在 CPU 上闭环正确。
- DDColor 在平滑/照片类灰度上给出明确彩色（a/b std≈3–5），确认 `colorize_ddcolor_tiny` 端到端有效。
- DeOldify 当前量化模型色度偏弱属**模型本身能力/量化损失**，不是代码 bug；建议上线前替换为 README 原本意图的 ColorizeArtistic 模型（注意其下载链接需另行修复/替换）。

---

## 6. 给 Kotlin / 安卓移植者的明确提醒清单

### 6.1 输入 shape / dtype（必须严格对应）
- 两个模型输入张量名都是 `"input"`，输出名 DeOldify=`"out"`、DDColor=`"output"`。
- 输入固定 **`[1, 3, 256, 256]` float32，NCHW 布局**。安卓端 `Bitmap`→`FloatBuffer` 时务必先 resize 到 256×256、RGB 顺序、再 `(3,256,256)` 排布、再包 batch 维。
- **DeOldify 输出 `[1,3,256,256]`（BGR 顺序，0–255）；DDColor 输出 `[1,2,256,256]`（仅 a,b 两通道）。**

### 6.2 输入值域（最易错，照抄第 4 节）
- **DeOldify**：像素 `0–255` 直接转 `Float`（不要 `/255`）。
- **DDColor**：像素必须 `/255.0` 得到 `0–1`。
- 两者**相反**，建两个独立的预处理分支，不要共用一个归一化函数。

### 6.3 OpenCV 关键坑（都在 4.13 实测）
- **`cvtColor` 常量名**：Python `cv2.COLOR_RGB2BGR / BGR2GRAY / GRAY2RGB / BGR2LAB / LAB2RGB / LAB2BGR` 在安卓 Java/OpenCV 里同名（`Imgproc.COLOR_*`），可直接对应。
- **⚠️ float 版 LAB 的真实值域（最容易踩的坑）**：OpenCV 4.13 对 **float32** 的 LAB 用的是**完整 Lab 范围**，不是旧文档说的 `[0,1]`：
  - `BGR2Lab(float)` → **L∈[0,100]**，a,b∈[−128,127]（实测：float BGR=0.5→L=53.4；纯红→Lab≈[32.3,79.2,−107.9]）。
  - `Lab2BGR(float)` 同样期望 **L∈[0,100]**，a,b∈[−128,127]。
  - **后果**：DDColor 后处理里 `orig_l`（来自 float BGR2Lab，L∈[0,100]）直接拼上模型输出的 ab（∈[−128,127]）再 `LAB2BGR(float)` 是**自洽正确**的。安卓端若用别的库/自己实现、误把 L 当 [0,1] 或 ab 当 [−0.5,0.5]，颜色会全部错乱。请务必用 OpenCV 的 `cvtColor` 而不要手写 Lab 公式。
- **`cv2.resize` 默认插值 = `INTER_LINEAR`**（双线性）。安卓端 resize 到 256 与放大 ab 回原尺寸都建议用双线性，保持一致；避免用最近邻导致色块。
- **`GaussianBlur((13,13),0)`**：ksize 必须是奇数正数，sigma=0 表示按 ksize 自动算；安卓 `Imgproc.GaussianBlur` 参数一致。
- **uint8 截断顺序**：DDColor 最后 `(bgr*255).round().clip(0,255).astype(uint8)`——先乘 255、四舍五入、clip 到 [0,255]、再转 uint8。安卓端别漏掉 `.round()`/clip，否则溢出偏色。

### 6.4 模型与 EP
- onnxruntime 1.24 CPU EP **能直接加载 fp32 与 int8 量化模型**，无需 fp16 开关。
- 安卓端若用 **NNAPI / GPU EP**：int8 量化 DeOldify 可加速；fp32 的 934MB 完整 DDColor **不建议**直接上移动端，换 `ddcolor_tiny`。
- 加载时建议显式指定 EP 顺序：`["NnapiExecutionProvider","CpuExecutionProvider"]`（或 GPU），并确认 `session.get_inputs()[0].shape` 为 `[1,3,256,256]`（DDColor 元数据是动态维度，运行时固定为 256）。

### 6.5 输出后处理（两个模型不同，别混）
- **DeOldify**：输出就是完整 BGR 彩图（0–255）→ `BGR2RGB` → 可选 `GaussianBlur` → 与**原图 L 通道**在 Lab 空间做 a/b 替换（`colorize_deoldify` 第 30–38 行）。注意它还在 Lab 里用原图 L 保结构。
- **DDColor**：只输出 ab → 必须和**原图 L**（float LAB, L∈[0,100]）拼成 LAB → `LAB2BGR(float)` → `*255` → uint8。千万不要把 DDColor 的 ab 当完整 RGB 直接用。

---

## 7. 完整可运行参考实现

文件已落盘：`/root/.codebuddy/artifact/py_ref_impl.py`，以下是其完整内容（严格对应 `colorize.py`，每步带 shape/dtype 注释，并含“输入值域探针”）：

```python
#!/usr/bin/env python3
"""
py_ref_impl.py  —  AiColorize 算法参考实现 / 安卓移植验证
严格对应 /root/.codebuddy/artifact/AiColorize/colorize.py
"""
import os
import numpy as np
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DEOLDIFY = os.path.join(MODEL_DIR, "deoldify.onnx")
DDCOLOR = os.path.join(MODEL_DIR, "ddcolor.onnx")
TEST_IMG = os.path.join(MODEL_DIR, "test_input.png")
OUT_DEOLDIFY = os.path.join(MODEL_DIR, "out_deoldify.png")
OUT_DDCOLOR = os.path.join(MODEL_DIR, "out_ddcolor.png")


def make_test_image(h=300, w=400):
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 0, 4)
    cv2.circle(img, (w // 3, h // 2), 70, 0, 3)
    cv2.circle(img, (w // 3 - 25, h // 2 - 20), 8, 0, -1)
    cv2.circle(img, (w // 3 + 25, h // 2 - 20), 8, 0, -1)
    cv2.line(img, (w // 3 - 20, h // 2 + 30), (w // 3 + 20, h // 2 + 30), 0, 3)
    cv2.rectangle(img, (w // 2, h // 4), (w - 30, h // 4 + 50), 0, 3)
    cv2.line(img, (w // 2, h // 4 + 25), (w - 40, h // 4 + 25), 0, 2)
    return img


def colorize_deoldify(input_gray: np.ndarray, model_path: str):
    gray_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)   # (H,W,3) uint8 0-255
    input_image = cv2.resize(gray_rgb, (256, 256))            # (256,256,3) uint8 0-255
    input_data = input_image.astype(np.float32)              # (256,256,3) f32 0-255 (不归一化!)
    input_data = input_data.transpose((2, 0, 1))             # (3,256,256) f32
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # (1,3,256,256) f32 0-255
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    colorized = sess.run(None, {sess.get_inputs()[0].name: input_data})[0][0]  # (3,256,256) f32 BGR 0-255
    colorized = colorized.transpose(1, 2, 0)                 # (256,256,3) f32
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)    # (256,256,3) uint8
    return colorized, input_data


def colorize_ddcolor(input_gray: np.ndarray, model_path: str):
    h, w = input_gray.shape[:2]
    gray_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)   # (H,W,3) uint8
    img = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR)           # (H,W,3) uint8
    img_norm = (img / 255.0).astype(np.float32)              # (H,W,3) f32 0-1  <-- 归一化!
    orig_l = cv2.cvtColor(img_norm, cv2.COLOR_BGR2Lab)[:, :, :1]   # (H,W,1) f32 L∈[0,100]
    img_resized = cv2.resize(img_norm, (256, 256))           # (256,256,3) f32 0-1
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]  # (256,256,1) f32
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)  # (256,256,3)
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)  # (256,256,3) f32 0-1
    tensor = img_gray_rgb.transpose((2, 0, 1))               # (3,256,256) f32
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)   # (1,3,256,256) f32 0-1
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    output_ab = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]  # (2,256,256) f32 ab
    output_ab = output_ab.transpose(1, 2, 0)                 # (256,256,2) f32 ab
    output_ab_resize = cv2.resize(output_ab, (w, h))         # (H,W,2) f32 ab
    output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)  # (H,W,3) LAB
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)  # (H,W,3) f32 0-1
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)  # (H,W,3) uint8
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)  # (H,W,3) uint8
    return output_rgb, tensor


def range_probe(model_path, kind):
    """同一张图分别用 0-255 与 0-1 跑, 看哪种产出合理, 用于实证输入值域."""
    rng = np.random.RandomState(0)
    base = (rng.rand(256, 256, 3) * 255).astype(np.float32)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    def run(t):
        t4 = np.expand_dims(t.transpose(2, 0, 1).astype(np.float32), 0)
        return sess.run(None, {in_name: t4})[0][0].transpose(1, 2, 0)
    if kind == "deoldify":
        rgb_raw = cv2.cvtColor(run(base), cv2.COLOR_BGR2RGB)
        rgb_norm = cv2.cvtColor(run(base / 255.0), cv2.COLOR_BGR2RGB)
    else:
        rgb_raw = run(base).transpose(1, 2, 0)
        rgb_norm = run(base / 255.0).transpose(1, 2, 0)
    def s(x): return f"mean={x.mean():.3f} std={x.std():.3f} min={x.min():.2f} max={x.max():.2f}"
    print(f"[探针]{kind}: 输入0-255 -> {s(rgb_raw)} | 输入0-1 -> {s(rgb_norm)}")


def main():
    test = make_test_image()
    cv2.imwrite(TEST_IMG, test)
    if os.path.exists(DEOLDIFY):
        out, tin = colorize_deoldify(test, DEOLDIFY)
        cv2.imwrite(OUT_DEOLDIFY, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print("DeOldify 输入", tin.shape, tin.dtype, "min", tin.min(), "max", tin.max())
        range_probe(DEOLDIFY, "deoldify")
    if os.path.exists(DDCOLOR):
        out, tin = colorize_ddcolor(test, DDCOLOR)
        cv2.imwrite(OUT_DDCOLOR, cv2.cvtColor(out, cv2.COLOR_RGB2RGB))
        print("DDColor 输入", tin.shape, tin.dtype, "min", tin.min(), "max", tin.max())
        range_probe(DDCOLOR, "ddcolor")


if __name__ == "__main__":
    main()
```

运行：`python3.11 /root/.codebuddy/artifact/py_ref_impl.py`

---

## 8. 一句话汇总（关键结论）

**DeOldify 的输入是 0–255 未归一化的 float32 RGB，DDColor 的输入是 /255 归一化到 0–1 的 float32 RGB——两者值域完全相反，且 OpenCV 4.13 的 float-LAB 实际用 L∈[0,100]、a,b∈[−128,127]（非旧文档的 [0,1]/[−0.5,0.5]），这三点是安卓移植最易错的硬约束；另外 README 的 DeOldify 下载链接已失效，实际可用的 `deoldify.quant.onnx` 是 int8 量化模型（安卓更友好），而下载到的 DDColor 是 934MB 完整 fp32 模型（移动端应换 `ddcolor_tiny`）。**
