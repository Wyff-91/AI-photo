# eval/probe_sd_drift.py —— v1 后续归因实验：SD 的失利 = 漂移 + 补全质量，各占多少？
# 背景：v1 表所有指标（含 LPIPS）都在全图计算，SD 的全图漂移污染了每一项。
# 方法：构造"隔离图"把两个分量拆开——
#   漂移隔离图 = GT 的掩膜内 + SD输出的掩膜外 → 与GT比，差异只来自漂移
#   补全隔离图 = GT 的掩膜外 + SD输出的掩膜内 → 与GT比，差异只来自补全（=贴回合成）
# 同时生成 GT/破损/LaMa/SD 四联图供肉眼判断。
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image, ImageDraw
from eval.metrics import lpips_score, masked_psnr

IMAGES = ["building_01.png", "land_04.png", "objection_05.png", "people_02.png", "texture_01.png"]
TIER, SEED = "medium", 42
PROMPT = "natural, seamless continuation of the surrounding scene, high quality photo"
NEG = "text, watermark, logo, blurry, distortion, frame, border"

def isolate(gt, pred, mask, which):
    """which='drift'：掩膜外用pred、掩膜内用GT（差异只反映漂移）
       which='fill'  ：掩膜内用pred、掩膜外用GT（差异只反映补全质量，即'贴回合成'）"""
    out = gt.copy()
    if which == "drift":
        out[mask <= 127] = pred[mask <= 127]
    else:
        out[mask > 127] = pred[mask > 127]
    return out

def collage(name, gt, mask, lama_out, sd_out):
    """GT(带红标) / 破损示意 / LaMa / SD 四联图"""
    panels = []
    g = Image.fromarray(gt).copy()
    d = ImageDraw.Draw(g)
    ys, xs = np.where(mask > 127)
    d.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], outline=(255, 0, 0), width=2)
    panels.append(g)
    dmg = Image.fromarray(gt).copy()
    dd = ImageDraw.Draw(dmg)
    dd.rectangle([xs.min(), ys.min(), xs.max(), ys.max()], fill=(128, 128, 128))
    panels.append(dmg)
    panels.append(Image.fromarray(lama_out))
    panels.append(Image.fromarray(sd_out))
    W = sum(p.width for p in panels) + 8 * 3
    canvas = Image.new("RGB", (W, gt.shape[0]), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width + 8
    canvas.save(f"results/probe_{name.replace('.png','')}.png")

def main():
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    from diffusers import StableDiffusionInpaintPipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.enable_attention_slicing()

    print(f"{'图':16s} {'模型':5s} {'全图LPIPS':>9s} {'漂移分量':>9s} {'补全分量':>9s} {'区域PSNR':>9s}")
    for name in IMAGES:
        gt = np.array(Image.open(f"data/eval_pairs/gt/{name}"))
        mask = np.array(Image.open(f"data/eval_pairs/mask_{TIER}/{name}"))
        img, m_img = Image.fromarray(gt), Image.fromarray(mask)

        lama_out = np.array(lama(img, m_img))
        g = torch.Generator("cuda").manual_seed(SEED)
        sd_out = np.array(pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=m_img,
                               height=img.height, width=img.width,
                               num_inference_steps=30, generator=g).images[0])
        for model, out in [("LaMa", lama_out), ("SD", sd_out)]:
            full = lpips_score(gt, out)
            drift = lpips_score(gt, isolate(gt, out, mask, "drift"))
            fill = lpips_score(gt, isolate(gt, out, mask, "fill"))
            mp = masked_psnr(gt, out, mask)
            print(f"{name:16s} {model:5s} {full:9.3f} {drift:9.3f} {fill:9.3f} {mp:9.2f}")
        collage(name, gt, mask, lama_out, sd_out)

    print("\n解读：LaMa的'漂移分量'应≈全图LPIPS（它掩膜外不变，漂移=0，全图差异即补全差异）；")
    print("      SD若'漂移分量'大而'补全分量'小 → v1的失利主要是漂移（贴回优化可救）；")
    print("      若'补全分量'也大 → SD在该分辨率/提示词下补全质量确实不足。")
    print("四联对比图已存 results/probe_*.png（GT红框/灰洞/LaMa/SD），请肉眼对照数字判断。")

if __name__ == "__main__":
    main()
