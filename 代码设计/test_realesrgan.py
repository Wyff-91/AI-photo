# test_realesrgan.py —— 里程碑②-B：Real-ESRGAN 超分首测
# 输入用 LaMa 的补全结果(1280x720)，放大4倍，同时验证"缩小修复→超分放大"管线衔接
import cv2  # realesrgan 的依赖，读图用 OpenCV（BGR 通道顺序是其惯例）
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 1. 定义 RealESRGAN_x4plus 对应的网络结构（RRDBNet，64特征通道/23个残差块，x4倍率）
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# 2. 加载超分器（首次运行自动下载权重约64MB，走系统代理从GitHub，缓存到torch hub）
upsampler = RealESRGANer(
    scale=4,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    model=model,
    tile=480,      # 分块推理：每次只处理480px的图块，8GB显存的标配做法（不分块跑高分辨率会OOM）
    tile_pad=10,   # 块与块之间重叠10px，避免拼接缝
    pre_pad=0,
    half=False,    # 不用fp16，保持精度（想提速可改True）
)

# 3. 读入 LaMa 的补全结果
img = cv2.imread("results/test1_lama_result.png", cv2.IMREAD_COLOR)
print("输入尺寸:", img.shape[1], "x", img.shape[0])

# 4. 超分 ×4
out, _ = upsampler.enhance(img, outscale=4)
print("输出尺寸:", out.shape[1], "x", out.shape[0])
cv2.imwrite("results/test1_lama_x4.png", out)
print("完成！用看图软件放大对比 test1_lama_result.png 与 test1_lama_x4.png 的云朵边缘细节")
