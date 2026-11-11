# test_sd_inpaint.py —— 里程碑③：Stable Diffusion Inpainting 首测
# 与 LaMa 同一个任务（掩膜补全），但换了扩散模型——之后两者要同图同掩膜对比
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 镜像写死在脚本里，不依赖终端环境变量
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

MODEL_ID = "Lykon/dreamshaper-8-inpainting"  # SD1.5底座的补全模型；官方SD2-inpainting已被设为门槛访问(gated)，换此公开替代

# 1. 加载管线（首次运行自动下载权重：fp16 版约 2.5GB，经 hf-mirror 镜像，只需一次）
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # 半精度：显存减半、速度加倍，质量几乎无损
    variant="fp16",
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()  # 用一点速度换更稳的显存占用，8GB 显卡的保险

# 2. 读图 + 掩膜（与 LaMa 测试同款：中央矩形洞，保证可对比）
img = Image.open("data/demo_images/test1.jpg").convert("RGB")
w, h = img.size
MAX_SIDE = 512  # SD2-inpainting 的训练分辨率是 512，喂 512 质量最稳
if max(img.size) > MAX_SIDE:
    ratio = MAX_SIDE / max(img.size)
    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    w, h = img.size
    print(f"已缩至 {w}x{h}")
mask = Image.new("L", img.size, 0)
d = ImageDraw.Draw(mask)
d.rectangle([int(w * 0.4), int(h * 0.35), int(w * 0.6), int(h * 0.65)], fill=255)
mask.save("results/test1_sd_mask.png")

# 3. 提示词：扩散模型是"文+图"条件生成——告诉模型洞里应该画什么
#    ★ 根据你照片中央区域实际是什么内容，把 prompt 改写成对应描述效果最好
prompt = "natural, seamless continuation of the surrounding scene, high quality photo"
negative_prompt = "text, watermark, logo, blurry, distortion, frame, border"

# 4. 生成：固定随机种子 → 结果可复现（方案里"可追溯"原则的落地）
generator = torch.Generator("cuda").manual_seed(42)
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=img,
    mask_image=mask,
    height=img.height,       # ★ 必须显式传：否则管线默认按 512×512 正方形输出，
    width=img.width,         #   把宽高比拉伸变形（宽高需为8的倍数，512×288 符合）
    num_inference_steps=30,  # 去噪步数：多=细节好+更慢，30 是质量/速度平衡点
    generator=generator,
).images[0]

result.save("results/test1_sd_result.png")
print("完成！对比 results/test1_sd_result.png（SD）与 test1_lama_result.png（LaMa）：谁的洞补得更自然？")
