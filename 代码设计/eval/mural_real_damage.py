# eval/mural_real_damage.py —— 第一步·真实病害复跑（双赛道）
# 赛道A（定性）：Mural_seg 带病害图 + 真实掩膜 → LaMa vs SD 补全 → 对比图（无GT，目验用）
# 赛道B（定量）：真实病害掩膜移植到干净 Mural512 图上造损 → LaMa / SD自由 / SD+ControlNet
#               → 与干净原图算 区域PSNR/SSIM/LPIPS（真实掩膜几何 + 可量化）
# 输出：results/mural_real_A_<名>.png（三联）、results/mural_real_B_<名>.png（四联）、指标表
import os, sys, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image, ImageDraw

SEG = "../datasets/MuralDH/MuralDH/Mural_seg/test"      # 图像+真实病害掩膜
CLEAN = "../datasets/MuralDH/MuralDH/Mural512"          # 干净512切片（赛道B的GT）
N = 8
PROMPT = ("dunhuang mural painting, ancient chinese fresco, mineral pigment texture, "
          "buddhist art, seamless continuation of the surrounding pattern")
NEG = "text, watermark, photorealistic photo, modern style, frame, border"

def paste_back(gt_img, pred, mask):
    g, p = np.array(gt_img), np.array(pred)
    m = np.array(mask) > 127
    p[~m] = g[~m]
    return Image.fromarray(p)

def pick_samples():
    """按掩膜覆盖率分层抽样，兼顾小/中/大病害"""
    files = sorted(os.listdir(os.path.join(SEG, "images")))
    cands = []
    for f in files:
        lab_p = os.path.join(SEG, "labels", f)
        if not os.path.exists(lab_p):
            continue
        cov = (np.array(Image.open(lab_p)) > 127).mean()
        if 0.005 < cov < 0.5:
            cands.append((cov, f))
    cands.sort()
    random.seed(42)
    stride = max(1, len(cands) // N)
    return [cands[min(i * stride, len(cands) - 1)] for i in range(N)]

def main():
    from simple_lama_inpainting import SimpleLama
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel
    from eval.metrics import masked_psnr, masked_ssim, lpips_score
    import cv2

    print("加载引擎…")
    lama = SimpleLama()
    sd = StableDiffusionInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    sd.enable_attention_slicing(); sd.safety_checker = None
    cn = ControlNetModel.from_pretrained("weights/controlnet_canny", torch_dtype=torch.float16, variant="fp16")
    sd_cn = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", controlnet=cn,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    sd_cn.enable_attention_slicing(); sd_cn.safety_checker = None

    def sd_fill(pipe, img, mask, use_cn=False, scale=0.7):
        control = None
        if use_cn:
            arr = np.array(img)
            arr[np.array(mask) > 127] = 0          # 掩膜区置零，避免病害边缘漏进底稿
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            control = Image.fromarray(cv2.cvtColor(cv2.Canny(gray, 80, 160), cv2.COLOR_GRAY2RGB))
        kw = dict(control_image=control, controlnet_conditioning_scale=scale) if use_cn else {}
        g = torch.Generator("cuda").manual_seed(42)
        out = pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=mask,
                   height=img.height, width=img.width, num_inference_steps=30,
                   generator=g, **kw).images[0]
        return paste_back(img, out, mask)

    samples = pick_samples()
    print(f"抽样 {len(samples)} 张，覆盖率: {[f'{c*100:.1f}%' for c, _ in samples]}")

    # ---------- 赛道A：真实病害图 + 真实掩膜（定性） ----------
    print("== 赛道A：真实病害修复（定性）==")
    for cov, f in samples:
        img = Image.open(os.path.join(SEG, "images", f)).convert("RGB").resize((512, 512), Image.LANCZOS)
        mask = Image.open(os.path.join(SEG, "labels", f)).resize((512, 512), Image.NEAREST)
        lama_out = lama(img, mask)
        sd_out = sd_fill(sd, img, mask)
        ov = np.array(img).copy(); m = np.array(mask) > 127
        ov[m] = (ov[m] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
        row = Image.new("RGB", (512*3 + 16, 512), (0, 0, 0))
        row.paste(Image.fromarray(ov), (0, 0)); row.paste(lama_out, (520, 0)); row.paste(sd_out, (1040, 0))
        row.save(f"results/mural_real_A_{f.replace('.png','')}.png")
        print(f"  A完成 {f}（覆盖{cov*100:.0f}%）")

    # ---------- 赛道B：真实掩膜几何 + 干净GT（定量） ----------
    print("== 赛道B：真实掩膜×干净图（定量）==")
    cleans = sorted(os.listdir(CLEAN))[:200]
    rows = []
    for i, (cov, f) in enumerate(samples):
        gt = Image.open(os.path.join(CLEAN, cleans[i * 7])).convert("RGB").resize((512, 512), Image.LANCZOS)
        mask = Image.open(os.path.join(SEG, "labels", f)).resize((512, 512), Image.NEAREST)
        m = np.array(mask) > 127
        lama_out = lama(gt, mask)
        sd_out = sd_fill(sd, gt, mask)
        cn_out = sd_fill(sd_cn, gt, mask, use_cn=True)
        gtn = np.array(gt)
        r = dict(name=f, cov=f"{cov*100:.1f}%",
                 lama_psnr=masked_psnr(gtn, np.array(lama_out), m),
                 sd_psnr=masked_psnr(gtn, np.array(sd_out), m),
                 cn_psnr=masked_psnr(gtn, np.array(cn_out), m),
                 lama_lpips=lpips_score(gtn, np.array(lama_out)),
                 sd_lpips=lpips_score(gtn, np.array(sd_out)),
                 cn_lpips=lpips_score(gtn, np.array(cn_out)))
        rows.append(r)
        row_img = Image.new("RGB", (512*4 + 24, 512), (0, 0, 0))
        for j, p in enumerate([gt, lama_out, sd_out, cn_out]):
            row_img.paste(p, (j*520, 0))
        row_img.save(f"results/mural_real_B_{f.replace('.png','')}.png")
        print(f"  B完成 {f}: LPIPS LaMa={r['lama_lpips']:.3f} SD={r['sd_lpips']:.3f} SD+CN={r['cn_lpips']:.3f}")

    print("\n===== 赛道B 定量汇总（区域PSNR↑ / LPIPS↓）=====")
    print(f"{'样本':12s} {'覆盖':>6s} {'LaMa':>14s} {'SD自由':>14s} {'SD+CN':>14s}")
    for r in rows:
        print(f"{r['name']:12s} {r['cov']:>6s} "
              f"{r['lama_psnr']:6.2f}/{r['lama_lpips']:.3f} "
              f"{r['sd_psnr']:6.2f}/{r['sd_lpips']:.3f} "
              f"{r['cn_psnr']:6.2f}/{r['cn_lpips']:.3f}")
    import statistics as st
    print(f"{'均值':12s} {'':>6s} "
          f"{st.mean(r['lama_psnr'] for r in rows):6.2f}/{st.mean(r['lama_lpips'] for r in rows):.3f} "
          f"{st.mean(r['sd_psnr'] for r in rows):6.2f}/{st.mean(r['sd_lpips'] for r in rows):.3f} "
          f"{st.mean(r['cn_psnr'] for r in rows):6.2f}/{st.mean(r['cn_lpips'] for r in rows):.3f}")

if __name__ == "__main__":
    main()
