# eval/mural_lineart.py —— 第二步·lineart 真线稿底稿升级验证
# 三臂对照（同损伤同种子）：SD自由 vs SD+Canny底稿(洞内空白=应用现状) vs SD+真线稿底稿(洞内有线=DhMurals edges)
# 假设来源：第一步发现"洞内空白底稿下 ControlNet 反而最差"——本实验验证"洞内有真结构线"是否逆转
# 输出：results/mural_lineart_<名>.png 四联（GT红叠 | SD自由 | SD+Canny | SD+线稿）+ 定量汇总表
import os, sys, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

DATA = "../datasets/DhMurals1714/DhMurals-inpainting-dataset"
CN_CANNY = "weights/controlnet_canny"
CN_LINEART = "weights/controlnet_lineart"
N = 8
PROMPT = ("dunhuang mural painting, ancient chinese fresco, mineral pigment texture, "
          "buddhist art, seamless continuation of the surrounding pattern")
NEG = "text, watermark, photorealistic photo, modern style, frame, border"

def paste_back(gt_img, pred, mask):
    g, p = np.array(gt_img), np.array(pred)
    m = np.array(mask) > 127
    p[~m] = g[~m]
    return Image.fromarray(p)

def to_lineart_draft(edge_img, mask):
    """真线稿底稿：DhMurals 边缘图 → lineart 模型期望的白底黑线；掩膜区保留线（洞内有结构！）"""
    arr = np.array(edge_img.convert("L"))
    if arr.mean() < 127:                      # 黑底白线 → 反转为白底黑线
        arr = 255 - arr
    return Image.merge("RGB", (Image.fromarray(arr),) * 3)

def to_canny_draft(img, mask):
    """应用现状底稿：掩膜区置零后取 canny（洞内空白）"""
    arr = np.array(img).copy()
    arr[np.array(mask) > 127] = 0
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return Image.fromarray(cv2.cvtColor(cv2.Canny(gray, 80, 160), cv2.COLOR_GRAY2RGB))

def main():
    from diffusers import (StableDiffusionInpaintPipeline,
                           StableDiffusionControlNetInpaintPipeline, ControlNetModel)
    from eval.metrics import masked_psnr, lpips_score

    print("加载引擎…")
    sd_free = StableDiffusionInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    sd_free.enable_attention_slicing(); sd_free.safety_checker = None
    cn_line = ControlNetModel.from_pretrained(CN_LINEART, torch_dtype=torch.float16, variant="fp16")
    pipe_cn = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", controlnet=cn_line,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe_cn.enable_attention_slicing(); pipe_cn.safety_checker = None
    cn_canny = ControlNetModel.from_pretrained(CN_CANNY, torch_dtype=torch.float16, variant="fp16").to("cuda")  # 必须.to(cuda)：换入CUDA管线时CPU权重会崩

    def fill(pipe, img, mask, control=None, scale=0.7):
        kw = dict(control_image=control, controlnet_conditioning_scale=scale) if control is not None else {}
        g = torch.Generator("cuda").manual_seed(42)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=mask,
                   height=img.height, width=img.width, num_inference_steps=30,
                   generator=g, **kw).images[0]
        return paste_back(img, out, mask)

    names = sorted(os.listdir(os.path.join(DATA, "train", "images")))[:200:25][:N]
    mask_files = sorted(os.listdir(os.path.join(DATA, "train", "masks")))
    random.seed(42)
    rows = []
    for idx, name in enumerate(names):
        gt = Image.open(os.path.join(DATA, "train", "images", name)).convert("RGB").resize((512, 512), Image.LANCZOS)
        edge = Image.open(os.path.join(DATA, "train", "edges", name)).resize((512, 512), Image.LANCZOS)
        mask = Image.open(os.path.join(DATA, "train", "masks", mask_files[idx * 3])).resize((512, 512), Image.NEAREST)
        mask = mask.point(lambda x: 255 if x > 127 else 0)
        m = np.array(mask) > 127

        free_out = fill(sd_free, gt, mask)
        pipe_cn.controlnet = cn_canny
        canny_out = fill(pipe_cn, gt, mask, control=to_canny_draft(gt, mask))
        pipe_cn.controlnet = cn_line
        line_out = fill(pipe_cn, gt, mask, control=to_lineart_draft(edge, mask))

        gtn = np.array(gt)
        r = dict(name=name,
                 free=(masked_psnr(gtn, np.array(free_out), m), lpips_score(gtn, np.array(free_out))),
                 canny=(masked_psnr(gtn, np.array(canny_out), m), lpips_score(gtn, np.array(canny_out))),
                 line=(masked_psnr(gtn, np.array(line_out), m), lpips_score(gtn, np.array(line_out))))
        rows.append(r)

        dmg = gt.copy(); ImageDraw.Draw(dmg).rectangle([170, 170, 342, 342], outline=(255, 0, 0), width=3)
        row_img = Image.new("RGB", (512*4 + 24, 512), (0, 0, 0))
        for j, p in enumerate([dmg, free_out, canny_out, line_out]):
            row_img.paste(p, (j * 520, 0))
        row_img.save(f"results/mural_lineart_{name.replace('.jpg','')}.png")
        print(f"完成 {name}: LPIPS 自由={r['free'][1]:.3f} canny={r['canny'][1]:.3f} 线稿={r['line'][1]:.3f}")

    import statistics as st
    print("\n===== 三臂汇总（区域PSNR↑ / LPIPS↓）=====")
    for arm in ["free", "canny", "line"]:
        print(f"{arm:6s}: PSNR={st.mean(r[arm][0] for r in rows):6.2f}  LPIPS={st.mean(r[arm][1] for r in rows):.3f}")

if __name__ == "__main__":
    main()
