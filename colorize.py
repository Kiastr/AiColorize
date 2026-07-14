#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

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

def main():
    parser = argparse.ArgumentParser(description='Image colorization using DeOldify or DDColor')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    parser.add_argument('-m', '--model', required=True, help='ONNX model path')
    parser.add_argument('--type', choices=['deoldify', 'ddcolor'], default='deoldify', help='Model type')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    args = parser.parse_args()
    if not os.path.exists(args.input):
        sys.exit(1)
    if not os.path.exists(args.model):
        sys.exit(1)
    try:
        if args.type == 'deoldify':
            colorize_deoldify(args.input, args.output, args.model, args.device)
        elif args.type == 'ddcolor':
            colorize_ddcolor_tiny(args.input, args.output, args.model, args.device)
    except Exception as e:
        sys.exit(1)

if __name__ == '__main__':
    main()
