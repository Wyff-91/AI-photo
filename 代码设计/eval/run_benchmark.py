# eval/run_benchmark.py —— 第2周任务3：一键基准测试（LaMa vs SD Inpainting）
# 原则：同图、同掩膜、同指标（可复现：全部固定种子）
# 用法：python eval/run_benchmark.py
# 输出：results/benchmark_v1.csv（逐条记录）
#       results/benchmark_v1_summary.md（模型×缺损档位汇总表，可直接进PPT）
import os, sys, csv, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根加入路径，使 eval.metrics 可导入
import numpy as np
import torch
from PIL import Image
from eval.metrics import evaluate_all

N_IMAGES = int(os.environ.get("N_BENCH", 12))  # 默认12张出v1表；设 N_BENCH=1 可快速冒烟
TIERS = ["small", "medium", "large"]
SEED = 42
PROMPT = "natural, seamless continuation of the surrounding scene, high quality photo"
NEG = "text, watermark, logo, blurry, distortion, frame, border"

def load_images():
    gts = sorted(os.listdir("data/eval_pairs/gt"))
    step = max(1, len(gts) // N_IMAGES)   # 等距抽样：文件按类别前缀排序，
    picked = gts[::step][:N_IMAGES]       # 步长取样保证 building/land/objection/people/texture 各类都有代表
    print(f"评测集共 {len(gts)} 张，本次抽取 {len(picked)} 张：{[p.split('_')[0] for p in picked]}")
    return [(name, np.array(Image.open(f"data/eval_pairs/gt/{name}"))) for name in picked]

def run_lama(data, rows):
    from simple_lama_inpainting import SimpleLama
    print("加载 LaMa ...")
    lama = SimpleLama()
    for name, gt in data:
        img = Image.fromarray(gt)
        for tier in TIERS:
            mask = Image.open(f"data/eval_pairs/mask_{tier}/{name}")
            t0 = time.time()
            pred = np.array(lama(img, mask))
            dt = time.time() - t0
            rows.append(dict(image=name, tier=tier, model="LaMa", sec=round(dt, 2),
                             **{k: round(v, 4) for k, v in evaluate_all(gt, pred, np.array(mask)).items()}))
            print(f"  LaMa {name} {tier} 完成 ({dt:.1f}s)")
    del lama
    torch.cuda.empty_cache()

def run_sd(data, rows):
    from diffusers import StableDiffusionInpaintPipeline
    print("加载 SD ...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.enable_attention_slicing()
    for name, gt in data:
        img = Image.fromarray(gt)
        for tier in TIERS:
            mask = Image.open(f"data/eval_pairs/mask_{tier}/{name}")
            g = torch.Generator("cuda").manual_seed(SEED)
            t0 = time.time()
            pred = np.array(pipe(prompt=PROMPT, negative_prompt=NEG, image=img, mask_image=mask,
                                 height=img.height, width=img.width,
                                 num_inference_steps=30, generator=g).images[0])
            dt = time.time() - t0
            rows.append(dict(image=name, tier=tier, model="SD", sec=round(dt, 2),
                             **{k: round(v, 4) for k, v in evaluate_all(gt, pred, np.array(mask)).items()}))
            print(f"  SD   {name} {tier} 完成 ({dt:.1f}s)")
    del pipe
    torch.cuda.empty_cache()

def summarize(rows):
    keys = ["psnr", "ssim", "lpips", "masked_psnr", "masked_ssim", "sec"]
    lines = ["# 基准测试 v1：LaMa vs SD Inpainting", "",
             f"样本：{N_IMAGES} 张 × 3 档掩膜 × 2 模型（种子 {SEED}，SD 30 步）", "",
             "| 缺损档位 | 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | 区域PSNR↑ | 区域SSIM↑ | 耗时s |",
             "|---|---|---|---|---|---|---|---|"]
    for tier in TIERS:
        for model in ["LaMa", "SD"]:
            sub = [r for r in rows if r["tier"] == tier and r["model"] == model]
            if not sub: continue
            m = {k: np.mean([r[k] for r in sub]) for k in keys}
            lines.append(f"| {tier} | {model} | {m['psnr']:.2f} | {m['ssim']:.3f} | {m['lpips']:.3f} "
                         f"| {m['masked_psnr']:.2f} | {m['masked_ssim']:.3f} | {m['sec']:.1f} |")
    with open("results/benchmark_v1_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

def main():
    data = load_images()
    rows = []
    run_lama(data, rows)   # 先跑完LaMa并释放显存，再加载SD（8GB装不下两个同时在线）
    run_sd(data, rows)
    with open("results/benchmark_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summarize(rows)
    print("\n已保存：results/benchmark_v1.csv 与 benchmark_v1_summary.md")

if __name__ == "__main__":
    main()
