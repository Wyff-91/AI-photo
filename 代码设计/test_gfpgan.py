# test_gfpgan.py —— 里程碑②-C：GFPGAN 人脸修复首测
# 流程：人脸检测 → 裁出人脸 → GFPGAN修复（眼睛/发丝/皮肤）→ 贴回原图
import cv2
from gfpgan import GFPGANer

# 0. 显存保护：与 LaMa 同款做法，大图先缩小
MAX_SIDE = 1536
img = cv2.imread("data/demo_images/face1.jpg", cv2.IMREAD_COLOR)
h, w = img.shape[:2]
print("原图尺寸:", w, "x", h)
if max(w, h) > MAX_SIDE:
    ratio = MAX_SIDE / max(w, h)
    img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    print("已缩至:", img.shape[1], "x", img.shape[0])

# 1. 加载 GFPGAN v1.4（首次运行自动下载两个权重：主模型333MB + 人脸检测约105MB，走系统代理，只需一次）
restorer = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    upscale=2,               # 修复的同时整图顺带放大2倍
    arch="clean",
    channel_multiplier=2,
)

# 2. 人脸修复
#    weight 参数 = 修复强度：0 = 尽量保留原图，1 = 完全采用生成结果，0.5 是稳妥默认
cropped_faces, restored_faces, restored_img = restorer.enhance(
    img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5
)

print(f"检测到人脸数: {len(restored_faces)}")
cv2.imwrite("results/face1_restored.png", restored_img)

# 每张脸存三个版本：修复前 / 修复后 / 左右拼接对比图
for i, (before, after) in enumerate(zip(cropped_faces, restored_faces)):
    cv2.imwrite(f"results/face{i}_before.png", before)   # 修复前的脸
    cv2.imwrite(f"results/face{i}_after.png", after)     # 修复后的脸
    b = cv2.resize(before, (after.shape[1], after.shape[0]))  # 统一尺寸后拼接
    cv2.imwrite(f"results/face{i}_compare.png", cv2.hconcat([b, after]))  # 左=前 右=后
print("完成！直接看 results/faceN_compare.png（左修复前/右修复后），整图效果看 face1_restored.png")
