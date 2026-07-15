#!/usr/bin/env python3
"""
py_ref_impl.py  —  AiColorize 算法参考实现 / 安卓移植验证
==========================================================
严格对应 /root/.codebuddy/artifact/AiColorize/colorize.py 中的
colorize_deoldify 与 colorize_ddcolor_tiny 两个函数。

每一步都标注 tensor 的 shape 与 dtype，并在末尾打印关键事实，
同时做"输入值域探针"以 empirically 确认 DeOldify / DDColor 的输入差异。

依赖: onnxruntime>=1.24, opencv-python(>=4.13), numpy>=2.0, Pillow
运行: python3.11 py_ref_impl.py
"""
import os
import sys
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


# ----------------------------------------------------------------------------
# 0. 生成一张模拟"黑白漫画"测试图 (400x300, uint8 灰度线稿)
# ----------------------------------------------------------------------------
def make_test_image(h=300, w=400):
    img = np.full((h, w), 255, dtype=np.uint8)          # 白底
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 0, 4)  # 外框
    cv2.circle(img, (w // 3, h // 2), 70, 0, 3)            # 脸轮廓
    cv2.circle(img, (w // 3 - 25, h // 2 - 20), 8, 0, -1)  # 左眼
    cv2.circle(img, (w // 3 + 25, h // 2 - 20), 8, 0, -1)  # 右眼
    cv2.line(img, (w // 3 - 20, h // 2 + 30), (w // 3 + 20, h // 2 + 30), 0, 3)  # 嘴
    cv2.rectangle(img, (w // 2, h // 4), (w - 30, h // 4 + 50), 0, 3)            # 对白框
    cv2.line(img, (w // 2, h // 4 + 25), (w - 40, h // 4 + 25), 0, 2)           # 对白线
    return img


# ----------------------------------------------------------------------------
# 1. DeOldify  —  严格对应 colorize_deoldify()
#    关键: 输入为 0-255 未归一化的 RGB (灰度转 RGB), dtype=float32
# ----------------------------------------------------------------------------
def colorize_deoldify(input_gray: np.ndarray, model_path: str):
    """
    input_gray: HxW uint8 灰度图 (漫画线稿)
    返回: (colorized_rgb_uint8, input_tensor_for_debug)
    """
    # 灰度 -> RGB (三通道相同)
    gray_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)   # (H,W,3) uint8, 值 0-255
    # resize 到模型固定输入 256x256
    input_image = cv2.resize(gray_rgb, (256, 256))            # (256,256,3) uint8, 0-255
    # 转 float32 —— 注意: 不做 /255, 直接保持 0-255
    input_data = input_image.astype(np.float32)              # (256,256,3) float32, 值 0-255
    # HWC -> CHW
    input_data = input_data.transpose((2, 0, 1))             # (3,256,256) float32
    # 加 batch 维
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # (1,3,256,256) float32, 值 0-255

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    colorized = sess.run(None, {in_name: input_data})[0][0]  # (3,256,256) float32 (模型输出 BGR 0-255)
    colorized = colorized.transpose(1, 2, 0)                 # (256,256,3) float32
    # 模型输出按 BGR 解释, 转 RGB 后转 uint8
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB).astype(np.uint8)  # (256,256,3) uint8
    return colorized, input_data


# ----------------------------------------------------------------------------
# 2. DDColor (tiny)  —  严格对应 colorize_ddcolor_tiny()
#    关键: 输入为 /255 归一化到 0-1 的 RGB, dtype=float32; 只预测 ab 两通道
# ----------------------------------------------------------------------------
def colorize_ddcolor(input_gray: np.ndarray, model_path: str):
    """
    input_gray: HxW uint8 灰度图
    返回: (colorized_rgb_uint8, input_tensor_for_debug)
    """
    h, w = input_gray.shape[:2]
    # 灰度 -> RGB -> BGR
    gray_rgb = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)   # (H,W,3) uint8
    img = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR)           # (H,W,3) uint8
    # 归一化到 0-1 (DDColor 与 DeOldify 的最大差异点!)
    img_norm = (img / 255.0).astype(np.float32)              # (H,W,3) float32, 值 0-1
    # 原图 L 通道 (用于最后拼回), OpenCV 对 float 输入输出 L 也是 0-1
    orig_l = cv2.cvtColor(img_norm, cv2.COLOR_BGR2Lab)[:, :, :1]   # (H,W,1) float32, 0-1
    # resize 到 256x256 再取 L
    img_resized = cv2.resize(img_norm, (256, 256))           # (256,256,3) float32, 0-1
    img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]  # (256,256,1) float32, 0-1
    # 构造灰度 LAB (a=b=0), 再转回 RGB 作为模型输入
    img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)  # (256,256,3) LAB
    img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)  # (256,256,3) float32, 0-1 (灰度)
    # HWC -> CHW -> batch
    tensor = img_gray_rgb.transpose((2, 0, 1))               # (3,256,256) float32
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)   # (1,3,256,256) float32, 0-1

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    output_ab = sess.run(None, {in_name: tensor})[0][0]      # (2,256,256) float32 (ab 两通道)
    output_ab = output_ab.transpose(1, 2, 0)                 # (256,256,2) float32
    # 上采样 ab 到原图尺寸
    output_ab_resize = cv2.resize(output_ab, (w, h))         # (H,W,2) float32
    # 拼回原图 L 通道
    output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)  # (H,W,3) LAB, L 0-1, ab 模型值
    # LAB -> BGR (float 0-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)  # (H,W,3) float32, 0-1
    # 0-1 -> 0-255 uint8
    output_img = (output_bgr * 255.0).round().clip(0, 255).astype(np.uint8)  # (H,W,3) uint8
    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)  # (H,W,3) uint8
    return output_rgb, tensor


# ----------------------------------------------------------------------------
# 3. "输入值域探针": 用两种归一化方式各跑一次, 看哪种产出合理彩色
#    用以 empirically 证明 DeOldify=0-255, DDColor=0-1
# ----------------------------------------------------------------------------
def range_probe(model_path, kind):
    print(f"\n[探针] {kind}: 比较两种输入值域 (0-255 vs 0-1)")
    # 固定一张确定性测试图
    rng = np.random.RandomState(0)
    base = (rng.rand(256, 256, 3) * 255).astype(np.float32)   # 0-255
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    def run(t):
        t4 = np.expand_dims(t.transpose(2, 0, 1).astype(np.float32), 0)
        out = sess.run(None, {in_name: t4})[0][0]            # (C,256,256)
        out = out.transpose(1, 2, 0)
        return out

    if kind == "deoldify":
        out_raw = run(base)                 # 0-255 输入
        out_norm = run(base / 255.0)        # 0-1 输入
        # 模型输出按 BGR 解释
        rgb_raw = cv2.cvtColor(out_raw, cv2.COLOR_BGR2RGB)
        rgb_norm = cv2.cvtColor(out_norm, cv2.COLOR_BGR2RGB)
    else:  # ddcolor: 输出是 ab 两通道
        out_raw = run(base)                 # 0-255 输入 -> (2,256,256)
        out_norm = run(base / 255.0)        # 0-1 输入
        rgb_raw = out_raw.transpose(1, 2, 0)
        rgb_norm = out_norm.transpose(1, 2, 0)

    def stats(x):
        return f"mean={x.mean():.3f} std={x.std():.3f} min={x.min():.2f} max={x.max():.2f}"

    print(f"   输入 0-255 -> {stats(rgb_raw)}")
    print(f"   输入 0-1   -> {stats(rgb_norm)}")


def main():
    # 生成并保存测试图
    test = make_test_image()
    cv2.imwrite(TEST_IMG, test)
    print(f"[test] 生成测试图 {TEST_IMG}  shape={test.shape} dtype={test.dtype}")

    # --- DeOldify ---
    if os.path.exists(DEOLDIFY):
        print("\n" + "=" * 60)
        print("运行 DeOldify")
        out, tin = colorize_deoldify(test, DEOLDIFY)
        cv2.imwrite(OUT_DEOLDIFY, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"  输入 tensor shape={tin.shape} dtype={tin.dtype} min={tin.min():.1f} max={tin.max():.1f}")
        print(f"  输出 RGB   shape={out.shape} dtype={out.dtype} mean={out.mean():.1f} std={out.std():.1f}")
        print(f"  已保存 -> {OUT_DEOLDIFY}")
        range_probe(DEOLDIFY, "deoldify")
    else:
        print("!! 未找到 DeOldify 模型", DEOLDIFY)

    # --- DDColor ---
    if os.path.exists(DDCOLOR):
        print("\n" + "=" * 60)
        print("运行 DDColor")
        out, tin = colorize_ddcolor(test, DDCOLOR)
        cv2.imwrite(OUT_DDCOLOR, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"  输入 tensor shape={tin.shape} dtype={tin.dtype} min={tin.min():.3f} max={tin.max():.3f}")
        print(f"  输出 RGB   shape={out.shape} dtype={out.dtype} mean={out.mean():.1f} std={out.std():.1f}")
        print(f"  已保存 -> {OUT_DDCOLOR}")
        range_probe(DDCOLOR, "ddcolor")
    else:
        print("!! 未找到 DDColor 模型", DDCOLOR)

    print("\n完成。")


if __name__ == "__main__":
    main()
