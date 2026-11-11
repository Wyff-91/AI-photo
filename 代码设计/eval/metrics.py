# eval/metrics.py —— 第2周任务2：评价指标计算
# 作用：PSNR / SSIM / LPIPS，以及"只在掩膜区域内计算"的版本（补全评测的业界惯例）
# 依赖：scikit-image(已装) + lpips(pip install lpips)
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None  # 懒加载：第一次调用才下载/加载 VGG 权重

def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    """全图 PSNR，越高越好"""
    return float(peak_signal_noise_ratio(gt, pred, data_range=255))

def ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """全图 SSIM，越高越好"""
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))

def masked_psnr(gt, pred, mask) -> float:
    """只在掩膜像素上算 PSNR（不稀释）：MSE 仅统计掩膜内像素，跨通道平均"""
    m = mask.astype(bool)
    if m.sum() < 10:
        return float("nan")
    mse = ((gt.astype(np.float64) - pred.astype(np.float64)) ** 2)[m].mean()
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))

def masked_ssim(gt, pred, mask) -> float:
    """掩膜外接矩形框内算 SSIM（业界近似）：框住破损区，看这一块的感知结构差"""
    ys, xs = np.where(mask.astype(bool))
    if len(ys) < 10:
        return float("nan")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return ssim(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])

def lpips_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """感知相似度 LPIPS，越低越好（0=感知上完全一样）"""
    global _lpips_model
    if _lpips_model is None:
        import lpips as lpips_lib
        _lpips_model = lpips_lib.LPIPS(net="alex")   # AlexNet backbone，轻量常用
    def to_t(x):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
        return t
    with torch.no_grad():
        return float(_lpips_model(to_t(gt), to_t(pred)).item())

def evaluate_all(gt, pred, mask=None) -> dict:
    """一键算齐：全图 PSNR/SSIM/LPIPS + （给了掩膜时）区域 PSNR/SSIM"""
    r = {"psnr": psnr(gt, pred), "ssim": ssim(gt, pred), "lpips": lpips_score(gt, pred)}
    if mask is not None:
        r["masked_psnr"] = masked_psnr(gt, pred, mask)
        r["masked_ssim"] = masked_ssim(gt, pred, mask)
    return r

if __name__ == "__main__":
    # 自检：拿一张退化图和它的真值算一遍，验证指标方向正确
    import os
    from PIL import Image
    g = np.array(Image.open("data/eval_pairs/gt/building_01.png"))
    b = np.array(Image.open("data/eval_pairs/blur/building_01.png"))
    m = np.array(Image.open("data/eval_pairs/mask_small/building_01.png"))
    print("自检（模糊图 vs 真值，应显示明显劣化）：")
    for k, v in evaluate_all(g, b, m).items():
        print(f"  {k:12s} = {v:.4f}")
