# eval/degrade.py —— 第2周任务1：合成退化管线
# 作用：把 data/clean 的高清图变成"带标准答案的坏图"——
#   GT(干净512图) + 掩膜(小/中/大) + 低分辨率版 + 模糊版
# 用法：在项目目录运行  python eval/degrade.py
# 输出：data/eval_pairs/ 下的 gt/ mask_small/ mask_medium/ mask_large/ lowres/ blur/
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SEED = 42                      # 固定种子：整条评测链可复现（可追溯原则）
EVAL_SIZE = 512                # 统一到 512：SD 原生分辨率，LaMa/超分同尺寸可公平对比
MIN_SIDE = 2000                # 长边低于此值的图自动跳过（GT 本身质量不足）
SRC, DST = "data/clean", "data/eval_pairs"

random.seed(SEED)
np.random.seed(SEED)

def center_crop_512(im):
    """居中裁正方形再缩到 512×512 —— 全模型统一输入尺寸"""
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((EVAL_SIZE, EVAL_SIZE), Image.LANCZOS)

def draw_strokes(size, n_strokes, max_len, width_range):
    """程序生成'划痕状'掩膜：随机游走笔画（白=要修的区域）"""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    for _ in range(n_strokes):
        x, y = random.uniform(0, size), random.uniform(0, size)
        angle = random.uniform(0, 2 * np.pi)
        for _ in range(int(max_len / 8)):
            step = 8
            nx, ny = x + step * np.cos(angle), y + step * np.sin(angle)
            d.line([(x, y), (nx, ny)], fill=255, width=random.randint(*width_range))
            x, y = nx, ny
            angle += random.uniform(-0.6, 0.6)          # 随机转折，模拟裂纹
            x = min(max(x, 0), size); y = min(max(y, 0), size)
    return mask

def main():
    for sub in ["gt", "mask_small", "mask_medium", "mask_large", "lowres", "blur"]:
        os.makedirs(os.path.join(DST, sub), exist_ok=True)

    files = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    used, skipped = [], []
    for f in files:
        im = Image.open(os.path.join(SRC, f)).convert("RGB")
        if max(im.size) < MIN_SIDE:
            skipped.append(f); continue
        name = f.rsplit(".", 1)[0]
        gt = center_crop_512(im)
        gt.save(f"{DST}/gt/{name}.png")                       # 真值

        # 三档缺损掩膜（按白像素占比约计：小≈2% 中≈10% 大≈25%，已实测校准）
        draw_strokes(EVAL_SIZE, 3, 400, (5, 9)).save(f"{DST}/mask_small/{name}.png")
        draw_strokes(EVAL_SIZE, 6, 500, (10, 18)).save(f"{DST}/mask_medium/{name}.png")
        draw_strokes(EVAL_SIZE, 8, 600, (16, 26)).save(f"{DST}/mask_large/{name}.png")

        # 低分辨率退化：×4 下采样 + 强 JPEG 压缩（模拟微信传输图）
        small = gt.resize((EVAL_SIZE // 4, EVAL_SIZE // 4), Image.BICUBIC)
        small.save(f"{DST}/lowres/{name}.jpg", quality=15)

        # 模糊退化：高斯模糊 + 轻噪点
        blurry = gt.filter(ImageFilter.GaussianBlur(3))
        arr = np.array(blurry).astype(np.int16)
        arr = np.clip(arr + np.random.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(f"{DST}/blur/{name}.png")

        used.append(name)

    print(f"完成：{len(used)} 张进入评测集，{len(skipped)} 张因长边<{MIN_SIDE}跳过：{skipped}")
    print(f"输出目录：{DST}/（gt=真值, mask_*三档掩膜, lowres=低分版, blur=模糊版）")

if __name__ == "__main__":
    main()
