# core/deblur_nafnet.py —— NAFNet 去模糊（OpenCV官方ONNX版，onnxruntime CPU推理）
# 预处理规则来自 opencv/deblurring_nafnet 的参考实现：RGB / 归一化[0,1] / 原生尺寸
# 本封装额外做：边缘复制式 pad 到 8 的倍数（防动态尺寸踩坑），推理后裁回原尺寸
import numpy as np
import onnxruntime as ort
from PIL import Image

_session = None

def _get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(
            "weights/deblurring_nafnet.onnx", providers=["CPUExecutionProvider"])
    return _session

def _to8(x):
    """向上取整到8的倍数：保证 pad 长度恒为正（向下取整会导致负数pad崩溃）"""
    return ((x + 7) // 8) * 8

def deblur(img: Image.Image) -> Image.Image:
    """输入/输出均为 PIL RGB 图。极小图先放大到安全边长再推理（防下采样到0维崩溃）"""
    arr = np.asarray(img.convert("RGB"))
    orig_h, orig_w = arr.shape[:2]
    MIN_SIDE = 384                    # 实测该ONNX模型最低安全边长：256及以下崩SCA节点(Pad 0维)，384起正常
    if min(orig_h, orig_w) < MIN_SIDE:
        r = MIN_SIDE / min(orig_h, orig_w)
        arr = np.array(Image.fromarray(arr).resize((int(orig_w * r), int(orig_h * r)), Image.LANCZOS))
    h, w = arr.shape[:2]
    h8, w8 = _to8(h), _to8(w)
    if (h8, w8) != (h, w):
        arr = np.pad(arr, ((0, h8 - h), (0, w8 - w), (0, 0)), mode="edge")
    x = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    sess = _get_session()
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    out = np.clip(out[0].transpose(1, 2, 0)[:h, :w] * 255, 0, 255).astype(np.uint8)
    result = Image.fromarray(out)
    if min(orig_h, orig_w) < MIN_SIDE:   # 放大过的还原到原始尺寸
        result = result.resize((orig_w, orig_h), Image.LANCZOS)
    return result
