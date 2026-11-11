# eval/mural_forward.py —— 壁画前瞻实验：现有"日常照片管线"直接处理壁画图，看哪些能打、哪些等第二阶段
# 素材：data/demo_images/mural/（敦煌风格图 1024×1024，来自 Jayw1/Dunhuang_img）
# 实验设计：每张图造两种损伤——stroke(划痕/龟裂) + rect(成片剥落)——分别用 LaMa 与 SD(壁画提示词+贴回)补全
# 输出：results/mural_forward_<图>_<损伤类型>.png 四联图（原图红框 / 掩膜 / LaMa / SD）
import os, sys, random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image, ImageDraw

MURAL_DIR = "data/demo_images/mural"
PROMPT = ("dunhuang mural painting, ancient chinese fresco, mineral pigment texture, "
          "buddhist art, seamless continuation of the surrounding pattern")
NEG = "text, watermark, photorealistic photo, modern style, frame, border"

def make_stroke_mask(size=512, seed=42):
    """划痕状掩膜：模拟壁画龟裂/刮痕"""
    random.seed(seed)
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    for _ in range(6):
        x, y = random.uniform(0, size), random.uniform(0, size)
        angle = random.uniform(0, 2 * np.pi)
        for _ in range(60):
            nx, ny = x + 8 * np.cos(angle), y + 8 * np.sin(angle)
            d.line([(x, y), (nx, ny)], fill=255, width=random.randint(10, 18))
            x, y = min(max(nx, 0), size), min(max(ny, 0), size)
            angle += random.uniform(-0.6, 0.6)
    return mask

def make_rect_mask(size=512):
    """矩形大洞：模拟成片剥落（约24%画面）"""
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle([int(size*0.38), int(size*0.35), int(size*0.62), int(size*0.65)], fill=255)
    return mask

def paste_back(gt_img, pred, mask):
    """贴回合成：掩膜外用原图（消 SD 全图漂移）——与 app 同款逻辑"""
    g, p = np.array(gt_img), np.array(pred)
    m = np.array(mask) > 127
    p[~m] = g[~m]
    return Image.fromarray(p)

def collage(path, img, mask, lama_out, sd_out):
    w, h = img.size
    dmg = img.copy()
    ImageDraw.Draw(dmg).rectangle([int(w*0.33), int(h*0.30), int(w*0.67), int(h*0.70)],
                                  outline=(255, 0, 0), width=3)
    mask_rgb = Image.merge("RGB", (mask, mask, mask))
    panels = [dmg, mask_rgb, lama_out, sd_out]
    canvas = Image.new("RGB", (w*4 + 8*3, h), (8, 8, 8))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += w + 8
    canvas.save(path)

def main():
    from simple_lama_inpainting import SimpleLama
    from diffusers import StableDiffusionInpaintPipeline
    print("加载 LaMa 与 SD …")
    lama = SimpleLama()
    sd = StableDiffusionInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    sd.enable_attention_slicing()
    sd.safety_checker = None  # 壁画肤色系颜料被NSFW检查器误判→返回黑图；修复场景输入输出均自控，关闭之

    masks = {"stroke": make_stroke_mask(), "rect": make_rect_mask()}
    for name in sorted(os.listdir(MURAL_DIR)):
        base = name.replace(".png", "")
        work = Image.open(os.path.join(MURAL_DIR, name)).convert("RGB").resize((512, 512), Image.LANCZOS)
        for tag, mask in masks.items():
            lama_out = lama(work, mask)
            g = torch.Generator("cuda").manual_seed(42)
            sd_raw = sd(prompt=PROMPT, negative_prompt=NEG, image=work, mask_image=mask,
                        height=512, width=512, num_inference_steps=30, generator=g).images[0]
            sd_out = paste_back(work, sd_raw, mask)
            collage(f"results/mural_forward_{base}_{tag}.png", work, mask, lama_out, sd_out)
            print(f"完成 {base} · {tag}")
    print("全部完成：results/mural_forward_*.png（四联：原图红框 / 掩膜 / LaMa / SD）")

if __name__ == "__main__":
    main()
