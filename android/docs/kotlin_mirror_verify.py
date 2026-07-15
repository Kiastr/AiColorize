#!/usr/bin/env python3
"""
kotlin_mirror_verify.py
严格 1:1 镜像 ColorizeEngine.kt 的完整算法逻辑（非简化版 reference_impl），
用真实 ONNX 模型跑通，验证：
  1) DeOldify 全后处理（GaussianBlur + LAB 合并 + 刻意 BGR2LAB-on-RGB quirk + resize）
  2) DDColor 修正后单次舍入
  3) 输出是合理彩色图（非错色/非纯灰）
  4) 实证 DDColor 旧"双重舍入(+0.5)"相对修正版的亮度偏置
"""
import os
import numpy as np
import cv2
import onnxruntime as ort

MODEL_DIR = "/root/.codebuddy/artifact/models"
DEOLDIFY = os.path.join(MODEL_DIR, "deoldify.onnx")
DDCOLOR = os.path.join(MODEL_DIR, "ddcolor.onnx")
OUT_DIR = "/tmp/kotlin_mirror_out"
os.makedirs(OUT_DIR, exist_ok=True)


def make_test_image(h=300, w=400):
    """模拟黑白漫画线稿（灰度 uint8）"""
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 0, 4)
    cv2.circle(img, (w // 3, h // 2), 70, 0, 3)
    cv2.circle(img, (w // 3 - 25, h // 2 - 20), 8, 0, -1)
    cv2.circle(img, (w // 3 + 25, h // 2 - 20), 8, 0, -1)
    cv2.line(img, (w // 3 - 20, h // 2 + 30), (w // 3 + 20, h // 2 + 30), 0, 3)
    cv2.rectangle(img, (w // 2, h // 4), (w - 30, h // 4 + 50), 0, 3)
    cv2.line(img, (w // 2, h // 4 + 25), (w - 40, h // 4 + 25), 0, 2)
    return img


def cvround_uint8(x):
    """镜像 OpenCV convertTo(CV_8U)：cvRound(四舍五入 half-to-even) + 饱和截断 [0,255]"""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ============================================================
# DeOldify —— 严格镜像 ColorizeEngine.colorizeDeoldify
# ============================================================
def colorize_deoldify_kotlin(input_gray_uint8, model_path):
    # 输入视作 BGR（灰度 B=G=R，与 Kotlin bitmapToBgrMat 等价）
    original_bgr = cv2.cvtColor(input_gray_uint8, cv2.COLOR_GRAY2BGR)
    h, w = original_bgr.shape[:2]

    target_l = original_bgr[:, :, 0]            # extractChannel(originalBgr, 0) = B 通道
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    input256 = cv2.resize(gray_rgb, (256, 256))
    input_f = input256.astype(np.float32)       # 0-255，不 /255
    tensor = input_f.transpose(2, 0, 1)[None]   # (1,3,256,256) NCHW 0-255

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: tensor})[0]  # (1,3,256,256) BGR 0-255
    colorized256 = out[0].transpose(1, 2, 0)    # (256,256,3) BGR

    colorized_rgb = cv2.cvtColor(colorized256, cv2.COLOR_BGR2RGB)
    colorized_uint8 = cvround_uint8(colorized_rgb)           # convertTo(CV_8U)
    colorized_full = cv2.resize(colorized_uint8, (w, h))
    blurred = cv2.GaussianBlur(colorized_full, (13, 13), 0)

    # ★ 刻意 quirk：blurred 实际是 RGB，但用 COLOR_BGR2LAB 处理（把 RGB 当 BGR）
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    merged = np.stack([target_l, a, b], axis=-1)            # uint8 LAB
    result_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)    # uint8 BGR
    return result_bgr, tensor


# ============================================================
# DDColor —— 严格镜像 ColorizeEngine.colorizeDdcolor（修正后单次舍入）
# ============================================================
def colorize_ddcolor_kotlin(input_gray_uint8, model_path, rounding="correct"):
    bgr = cv2.cvtColor(input_gray_uint8, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]

    img_norm = bgr.astype(np.float32) / 255.0               # 0-1
    lab_full = cv2.cvtColor(img_norm, cv2.COLOR_BGR2Lab)    # float LAB, L∈[0,100]
    orig_l = lab_full[:, :, 0:1]                            # (H,W,1)

    img_resized = cv2.resize(img_norm, (256, 256))
    lab_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)
    img_l = lab_resized[:, :, 0:1]                          # (256,256,1)
    zeros = np.zeros_like(img_l)
    gray_lab = np.concatenate([img_l, zeros, zeros], axis=-1)
    gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)    # 0-1
    tensor = gray_rgb.transpose(2, 0, 1)[None]              # (1,3,256,256) 0-1

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: tensor})[0]              # (1,2,256,256) ab
    ab256 = out[0].transpose(1, 2, 0)                       # (256,256,2)
    ab_full = cv2.resize(ab256, (w, h))                     # (H,W,2)

    out_lab = np.concatenate([orig_l, ab_full], axis=-1)    # (H,W,3)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)      # float 0-1
    scaled = out_bgr * 255.0

    if rounding == "correct":
        out_uint8 = cvround_uint8(scaled)                   # 单次 cvRound（修正后）
    else:  # "buggy" 旧双重舍入
        out_uint8 = cvround_uint8(scaled + 0.5)             # +0.5 再 cvRound（旧 bug）
    return out_uint8, tensor


def colorfulness(img_bgr):
    """Hasler-Süsstrunk colorfulness metric，越大越彩色"""
    (R, G, B) = (img_bgr[:, :, 2].astype(float), img_bgr[:, :, 1].astype(float), img_bgr[:, :, 0].astype(float))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt((rg.mean() - yb.mean()) ** 2)
    return std_root + 0.3 * mean_root


def main():
    test = make_test_image()
    cv2.imwrite(os.path.join(OUT_DIR, "input_gray.png"), test)
    print(f"[test] 灰度测试图 shape={test.shape} mean={test.mean():.1f}")

    # ---------- DeOldify ----------
    if os.path.exists(DEOLDIFY):
        res, tin = colorize_deoldify_kotlin(test, DEOLDIFY)
        cv2.imwrite(os.path.join(OUT_DIR, "out_deoldify_kotlin.png"), res)
        lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
        print(f"\n[DeOldify] 输入 tensor shape={tin.shape} dtype={tin.dtype} min={tin.min():.1f} max={tin.max():.1f} (应为 0-255)")
        print(f"[DeOldify] 输出 BGR shape={res.shape} dtype={res.dtype} mean={res.mean():.1f} std={res.std():.1f}")
        print(f"[DeOldify] 输出 LAB a通道 std={lab[:,:,1].std():.1f}  b通道 std={lab[:,:,2].std():.1f} (彩色应有非零 std)")
        print(f"[DeOldify] colorfulness={colorfulness(res):.1f} (灰度图≈0)")
    else:
        print("!! 未找到 DeOldify 模型:", DEOLDIFY)

    # ---------- DDColor ----------
    if os.path.exists(DDCOLOR):
        res_c, tin = colorize_ddcolor_kotlin(test, DDCOLOR, rounding="correct")
        res_b, _ = colorize_ddcolor_kotlin(test, DDCOLOR, rounding="buggy")
        cv2.imwrite(os.path.join(OUT_DIR, "out_ddcolor_kotlin_correct.png"), res_c)
        cv2.imwrite(os.path.join(OUT_DIR, "out_ddcolor_kotlin_buggy.png"), res_b)
        lab = cv2.cvtColor(res_c, cv2.COLOR_BGR2LAB)
        print(f"\n[DDColor] 输入 tensor shape={tin.shape} dtype={tin.dtype} min={tin.min():.3f} max={tin.max():.3f} (应为 0-1)")
        print(f"[DDColor] 输出 BGR shape={res_c.shape} dtype={res_c.dtype} mean={res_c.mean():.1f} std={res_c.std():.1f}")
        print(f"[DDColor] 输出 LAB a通道 std={lab[:,:,1].std():.1f}  b通道 std={lab[:,:,2].std():.1f}")
        print(f"[DDColor] colorfulness={colorfulness(res_c):.1f}")
        diff = res_c.astype(int) - res_b.astype(int)
        print(f"\n[DDColor 舍入对比] correct vs buggy(+0.5): mean(diff)={diff.mean():+.3f}  "
              f"非零像素占比={np.mean(diff!=0)*100:.1f}%  max|diff|={np.abs(diff).max()}")
        print("  预期：buggy 相对 correct 整体偏正（偏亮），非零占比高 → 印证双重舍入偏置")
    else:
        print("!! 未找到 DDColor 模型:", DDCOLOR)

    print(f"\n输出图已保存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
