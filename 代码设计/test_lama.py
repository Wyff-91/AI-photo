# test_lama.py —— 里程碑②-A：LaMa 补全首测
# 思路：人为在照片中间"挖一个洞"，让 LaMa 把洞补上——这就是图像补全的本质
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama

# 1. 读入照片
img = Image.open("data/demo_images/test1.jpg").convert("RGB")
w, h = img.size
print(f"原图尺寸: {w}x{h}")

# 1.5 显存保护：大图先缩小再补全
#     LaMa 的 FFC 结构在高分辨率下显存开销巨大，手机原图(12MP+)会让 8GB 显卡 OOM
#     标准做法："缩小修复 → 之后再超分放大"（②-B 的 Real-ESRGAN 负责放大）
MAX_SIDE = 1280
if max(img.size) > MAX_SIDE:
    ratio = MAX_SIDE / max(img.size)
    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    w, h = img.size
    print(f"图片过大，已缩至 {w}x{h} 再补全")

# 2. 生成"破损"掩膜(mask)：白色 = 需要修复的区域，黑色 = 保留原样的区域
#    这就是方案里"画面二：破损标注"的数据基础——之后界面上画笔涂的就是这种图
mask = Image.new("L", img.size, 0)
d = ImageDraw.Draw(mask)
d.rectangle([int(w*0.4), int(h*0.35), int(w*0.6), int(h*0.65)], fill=255)
mask.save("results/test1_mask.png")

# 3. 加载 LaMa（首次运行自动下载权重，约200MB，只需一次）
lama = SimpleLama()

# 4. 补全并保存
result = lama(img, mask)
result.save("results/test1_lama_result.png")
print("完成！去 results 文件夹对比三张图：原图 / 掩膜 / 修复结果")
