# make_compare.py —— 生成"原图 / LaMa / SD Inpainting"三联对比图（上排全图，下排补全区放大）
# 用法：python make_compare.py（项目目录、venv 激活状态下运行）
from PIL import Image, ImageDraw, ImageFont

# 1. 读三张图
original = Image.open("data/demo_images/test1.jpg").convert("RGB")
lama     = Image.open("results/test1_lama_result.png").convert("RGB")
sd       = Image.open("results/test1_sd_result.png").convert("RGB")

# 2. 掩膜矩形的相对位置（与生成脚本一致：40%-60% 宽，35%-65% 高）
def hole_box(size):
    w, h = size
    return (int(w*0.4), int(h*0.35), int(w*0.6), int(h*0.65))

# 3. 字体：Windows 自带微软雅黑
try:
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
except OSError:
    font = ImageFont.load_default()

def label_panel(img, text):
    """图上方加一条深色标签栏"""
    bar = Image.new("RGB", (img.width, 44), (24, 24, 24))
    d = ImageDraw.Draw(bar)
    d.text((10, 8), text, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (img.width, 44 + img.height))
    out.paste(bar, (0, 0))
    out.paste(img, (0, 44))
    return out

def fit_h(im, h):
    w = int(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)

gap = 6

# 4. 上排：三张全图统一高度后拼接；原图上用红框标出"模拟破损"的位置
ROW_H = 380
orig_fit = fit_h(original, ROW_H)
d = ImageDraw.Draw(orig_fit)
d.rectangle(hole_box(orig_fit.size), outline=(255, 40, 40), width=3)

panels = [
    label_panel(orig_fit,            "原图（红框=模拟破损区）"),
    label_panel(fit_h(lama, ROW_H),  "LaMa 补全（GAN 基线，快）"),
    label_panel(fit_h(sd, ROW_H),    "SD Inpainting（扩散，慢但细）"),
]
W = sum(p.width for p in panels) + gap * (len(panels) + 1)
H = max(p.height for p in panels) + gap * 2
canvas1 = Image.new("RGB", (W, H), (8, 8, 8))
x = gap
for p in panels:
    canvas1.paste(p, (x, gap))
    x += p.width + gap

# 5. 下排：三个"洞区"等比放大对比（评判补全质量主要看这排）
ZOOM_H = 300
def zoom(im):
    crop = im.crop(hole_box(im.size))
    w = int(crop.width * ZOOM_H / crop.height)
    return crop.resize((w, ZOOM_H), Image.LANCZOS)

zpanels = [
    label_panel(zoom(original), "原图 · 破损区"),
    label_panel(zoom(lama),     "LaMa · 补全区"),
    label_panel(zoom(sd),       "SD · 补全区"),
]
W2 = sum(p.width for p in zpanels) + gap * (len(zpanels) + 1)
H2 = max(p.height for p in zpanels) + gap * 2
canvas2 = Image.new("RGB", (W2, H2), (8, 8, 8))
x = gap
for p in zpanels:
    canvas2.paste(p, (x, gap))
    x += p.width + gap

# 6. 两排叠成总图
total = Image.new("RGB", (max(W, W2), H + H2 + gap), (8, 8, 8))
total.paste(canvas1, (0, 0))
total.paste(canvas2, (0, H + gap))
total.save("results/compare_lama_sd.png")
print("已生成 results/compare_lama_sd.png（上排全图对比，下排补全区放大对比）")
