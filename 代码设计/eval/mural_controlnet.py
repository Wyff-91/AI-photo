# eval/mural_controlnet.py —— 壁画补全结构约束验证：SD 自由补 vs SD+ControlNet(边缘底稿)补
# 回应反馈："SD补全不能随便进行，需要符合壁画全图"
# 设计：以【未损伤原图的边缘图】作为"参考底稿"（模拟修复师底稿/克孜尔式同题材参照），
#       ControlNet 强制 SD 沿底稿构图填充——对照 SD 自由补的风格漂移。
# 输出：results/mural_cn_<图>_<损伤>.png 五联（原图红框 / 边缘底稿 / LaMa / SD自由 / SD+约束）
# 注：真实壁画的底稿由第二阶段结构重建网络生成（调研"线描引导渐进修复"），本实验用原图模拟上界。
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from eval.mural_forward import make_rect_mask, make_stroke_mask, paste_back, PROMPT, NEG, MURAL_DIR

CN_PATH = "weights/controlnet_canny"   # 本地：config.json + diffusion_pytorch_model.fp16.safetensors

def to_canny(img: Image.Image) -> Image.Image:
    """参考底稿：全图 Canny 边缘（黑底白线）"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def collage(path, panels, w):
    gap = 8
    canvas = Image.new("RGB", (w * len(panels) + gap * (len(panels) - 1), w), (8, 8, 8))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += w + gap
    canvas.save(path)

def main():
    from simple_lama_inpainting import SimpleLama
    from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
    print("加载 LaMa / SD / ControlNet(canny) …")
    lama = SimpleLama()
    cn = ControlNetModel.from_pretrained(CN_PATH, torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", controlnet=cn,
        torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.enable_attention_slicing()
    pipe.safety_checker = None   # 壁画颜料误判NSFW→黑图，修复场景关闭

    for name in sorted(os.listdir(MURAL_DIR)):
        base = name.replace(".png", "")
        work = Image.open(os.path.join(MURAL_DIR, name)).convert("RGB").resize((512, 512), Image.LANCZOS)
        draft = to_canny(work)                     # ★ 底稿 = 原图边缘（模拟专家参考底稿）
        for tag, mask in {"rect": make_rect_mask(), "stroke": make_stroke_mask()}.items():
            lama_out = lama(work, mask)
            # SD 自由补：同一管线、同种子、约束强度=0（黑底稿）——与约束组唯一变量是 controlnet_conditioning_scale
            g1 = torch.Generator("cuda").manual_seed(42)
            free_img = pipe(prompt=PROMPT, negative_prompt=NEG, image=work, mask_image=mask,
                            control_image=Image.new("RGB", (512, 512), (0, 0, 0)),
                            controlnet_conditioning_scale=0.0,   # 约束强度=0 → 完全自由（同种子公平对照）
                            height=512, width=512, num_inference_steps=30, generator=g1).images[0]
            sd_free = paste_back(work, free_img, mask)
            g2 = torch.Generator("cuda").manual_seed(42)
            cn_img = pipe(prompt=PROMPT, negative_prompt=NEG, image=work, mask_image=mask,
                          control_image=draft, controlnet_conditioning_scale=0.7,  # 全强度1.0会压死纹理(实测)，0.7=锁构图不锁笔触
                          height=512, width=512, num_inference_steps=30, generator=g2).images[0]
            sd_cn = paste_back(work, cn_img, mask)

            dmg = work.copy()
            ImageDraw.Draw(dmg).rectangle([int(512*0.33), int(512*0.30), int(512*0.67), int(512*0.70)],
                                          outline=(255, 0, 0), width=3)
            collage(f"results/mural_cn_{base}_{tag}.png",
                    [dmg, draft, lama_out, sd_free, sd_cn], 512)
            print(f"完成 {base} · {tag}")
    print("全部完成：results/mural_cn_*.png（五联：原图红框/边缘底稿/LaMa/SD自由/SD+约束）")

if __name__ == "__main__":
    main()
