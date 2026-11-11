# app/app.py —— 第3周：Gradio 四画面主界面（v1 骨架）
# 四画面：①上传/示例 ②画笔破损标注 ③一键修复 ④前后对比
# 管线固定顺序：补全 → 人脸 → 超分；SD 路径内置"贴回合成"（v2 优化，归因实验论证）
# 启动：在项目目录运行  python app/app.py  → 浏览器打开 http://127.0.0.1:7860
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 所有HF模型均已本地缓存：强制离线，杜绝加载时联网校验的401瞬断
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 同上（transformers侧同款开关）
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")   # 本机免代理：否则gradio自检会走系统代理
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")   # 代理客户端没开时直接崩溃（亲测踩坑）
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False") # 关闭遥测，启动更干净
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 锚定项目根：任意目录启动，相对路径(weights/、CodeFormer/、data/)一律有效

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

MAX_SIDE_INPAINT = 1280   # 补全工作分辨率上限（36MP 原图 OOM 的教训）
MAX_SIDE_SD = 768         # SD系引擎上限：1280下实测 28.9% 掩膜区未被有效重画（diag_B 实验），768 兼顾质量与显存
MAX_SIDE_FACE = 1536      # 人脸修复上限
SD_SEED_DEFAULT = 42

_models = {}  # 惰性加载缓存：首次用到才加载，避免启动时占满显存

def get_lama():
    if "lama" not in _models:
        from simple_lama_inpainting import SimpleLama
        _models["lama"] = SimpleLama()
    return _models["lama"]

def get_sd():
    if "sd" not in _models:
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "Lykon/dreamshaper-8-inpainting", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.enable_attention_slicing()
        _models["sd"] = pipe
    return _models["sd"]

def _cn_modules():
    """两枚 ControlNet 模块（canny/lineart），都驻 GPU；共用同一条 CN 管线按需换装（省显存）"""
    import torch
    from diffusers import ControlNetModel
    if "cn_modules" not in _models:
        _models["cn_modules"] = {
            "canny": ControlNetModel.from_pretrained(
                "weights/controlnet_canny", torch_dtype=torch.float16, variant="fp16").to("cuda"),
            "lineart": ControlNetModel.from_pretrained(
                "weights/controlnet_lineart", torch_dtype=torch.float16, variant="fp16").to("cuda"),
        }
    return _models["cn_modules"]

def get_sd_cn():
    """SD + ControlNet 结构约束补全（canny 与 lineart 共用此管线，运行时换 controlnet 模块）"""
    if "sd_cn" not in _models:
        import torch
        from diffusers import StableDiffusionControlNetInpaintPipeline
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "Lykon/dreamshaper-8-inpainting", controlnet=_cn_modules()["canny"],
            torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.enable_attention_slicing()
        pipe.safety_checker = None   # 壁画肤色系颜料会被误判NSFW→黑图，修复场景关闭
        _models["sd_cn"] = pipe
    return _models["sd_cn"]

def _canny_draft(img: Image.Image) -> Image.Image:
    """自动底稿兜底：当前图的Canny边缘（洞内无线——上传线稿可解，见 _inpaint_once）"""
    import cv2
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def _lineart_from_upload(draft) -> Image.Image:
    """上传底稿归一化：转单通道、黑底白线自动反转为 lineart 模型期望的白底黑线"""
    d = Image.fromarray(draft[..., :3]) if isinstance(draft, np.ndarray) else draft.convert("RGB")
    arr = np.array(d.convert("L"))
    if arr.mean() < 127:
        arr = 255 - arr
    return Image.merge("RGB", (Image.fromarray(arr),) * 3)

def _harmonize_fill(original, generated, mask) -> Image.Image:
    """色彩协调 v2：优先 OpenCV 泊松融合（seamlessClone 梯度域融合，数学上消除色调接缝）；
    异常时退回 Reinhard 统计迁移（对齐掩膜环均值/方差）——治 SD 补全的'浅色框/深色框'"""
    try:
        import cv2
        o = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
        g = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
        m8 = ((np.array(mask) > 127) * 255).astype(np.uint8)
        m8 = cv2.erode(m8, np.ones((5, 5), np.uint8))    # 收缩5px防边缘毛刺
        ys, xs = np.nonzero(m8)
        if len(ys) < 50:
            return generated
        cy, cx = int(ys.mean()), int(xs.mean())
        if not m8[cy, cx]:                                # 中心若落在掩膜外(非凸形状)，取最近点
            i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            cy, cx = int(ys[i]), int(xs[i])
        blend = cv2.seamlessClone(g, o, m8, (cx, cy), cv2.NORMAL_CLONE)
        return Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    except Exception:
        from PIL import ImageFilter
        o = np.array(original).astype(np.float32)
        g = np.array(generated).astype(np.float32)
        m = np.array(mask) > 127
        dil = np.array(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(21))) > 127
        ring = dil & ~m
        if ring.sum() < 100 or m.sum() < 100:
            return Image.fromarray(g.astype(np.uint8))
        for c in range(3):
            src_mu, src_sd = g[m, c].mean(), g[m, c].std() + 1e-5
            tgt_mu, tgt_sd = o[ring, c].mean(), o[ring, c].std() + 1e-5
            g[m, c] = np.clip((g[m, c] - src_mu) * (tgt_sd / src_sd) + tgt_mu, 0, 255)
        return Image.fromarray(g.astype(np.uint8))

def get_gfpgan():
    if "gfpgan" not in _models:
        from gfpgan import GFPGANer
        _models["gfpgan"] = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            upscale=1, arch="clean", channel_multiplier=2)
    return _models["gfpgan"]

def get_realesrgan():
    # realesr-general-x4v3：小而快、带压缩降噪(dni)，专治微信压缩图（x4plus 会放大压缩噪点）
    if "sr" not in _models:
        from realesrgan import RealESRGANer
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        _models["sr"] = RealESRGANer(
            scale=4,
            model_path="weights/realesr-general-x4v3.pth",
            model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                                  upscale=4, act_type="prelu"),
            tile=480, tile_pad=10, pre_pad=0, half=False)
    return _models["sr"]

# ---------- 工具 ----------
def fit_max_side(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    r = max_side / max(img.size)
    return img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)

def editor_to_mask(editor_value):
    """从 ImageEditor 的返回值提取画笔掩膜（画过的地方=白）
    正确姿势：读"图层"的 alpha——图层是透明画布，只有笔画处不透明；
    ★不能读 composite 的 alpha：背景本身不透明（全图alpha=255），会把整张图误当掩膜"""
    if editor_value is None:
        return None
    val = editor_value if isinstance(editor_value, dict) else {"layers": [editor_value]}
    def to_arr(x):
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return np.array(x.convert("RGBA"))
        return np.asarray(x)
    strokes = None
    for layer in (val.get("layers") or []):
        arr = to_arr(layer)
        if arr is None:
            continue
        stroke = arr[:, :, 3] > 10 if (arr.ndim == 3 and arr.shape[2] == 4) else np.ones(arr.shape[:2], bool)
        strokes = stroke if strokes is None else (strokes | stroke)
    if strokes is None or not strokes.any():
        return None
    return Image.fromarray(strokes.astype(np.uint8) * 255)

def make_test_hole(img: Image.Image):
    """一键生成中央测试洞（与评测脚本同款相对位置）"""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([int(w*0.4), int(h*0.35), int(w*0.6), int(h*0.65)], fill=255)
    return mask

def on_check_mask(editor_value):
    """检查标注：把系统实际识别到的掩膜显示出来，用户确认'涂的地方被接受了'"""
    mask = editor_to_mask(editor_value)
    if mask is None:
        return None, "未识别到任何笔画——请用画笔在照片上涂抹，或勾选'中央测试洞'"
    cov = (np.array(mask) > 127).mean() * 100
    return mask, f"已识别标注：白色区域覆盖画面的 {cov:.1f}%，修复时将重画这里"

def paste_back(gt_img: Image.Image, pred: Image.Image, mask: Image.Image) -> Image.Image:
    """贴回合成（v2 优化）：掩膜外用原图，消除 SD 全图漂移"""
    g, p = np.array(gt_img), np.array(pred)
    m = np.array(mask) > 127
    p[~m] = g[~m]
    return Image.fromarray(p)

# ---------- 画面逻辑 ----------
def on_load_image(image):
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image).convert("RGB")
    # 同步进画面二的编辑器；同时重置分步状态（换图后旧步骤作废）
    editor_dict = {"background": np.array(img), "layers": [], "composite": None}
    return editor_dict, f"已载入：{img.width}×{img.height}px（请在画面二画笔标注破损区）", None

# ---------- 自由顺序修复引擎：hist = [(操作名, 图), ...]；hist[0]=("原图",基础图)，当前图=hist[-1][1] ----------
def _base_from(image, editor_value):
    """取基础图：画面一优先，回退画面二编辑器背景"""
    if image is not None:
        return np.array(image[..., :3]) if isinstance(image, np.ndarray) else np.array(image.convert("RGB"))
    if isinstance(editor_value, dict):
        bg = editor_value.get("background")
        if bg is not None:
            return np.array(bg.convert("RGB")) if isinstance(bg, Image.Image) else np.asarray(bg)[..., :3]
    return None

def _ensure_hist(image, editor_value, hist):
    if hist:
        return hist
    base = _base_from(image, editor_value)
    return None if base is None else [("原图", base)]

def _inpaint_once_impl(work, mask, engine, prompt, negative, seed, sd_steps, cn_scale, draft_img):
    """单次补全执行：引擎分发 + 底稿来源（上传线稿 > 自动canny）+ 贴回。供单次与多候选共用"""
    if engine == "LaMa（快）":
        return get_lama()(work, mask)
    import torch
    g = torch.Generator("cuda").manual_seed(int(seed))
    if engine.startswith("SD+ControlNet"):
        pipe = get_sd_cn()
        if draft_img is not None:                 # ★ 上传线稿：洞内有真结构（实验证明优于空白/canny）
            pipe.controlnet = _cn_modules()["lineart"]
            control = _lineart_from_upload(draft_img)
        else:                                     # 自动兜底：canny（洞内空白）
            pipe.controlnet = _cn_modules()["canny"]
            control = _canny_draft(work)
        out = pipe(prompt=prompt, negative_prompt=negative, image=work, mask_image=mask,
                   control_image=control, controlnet_conditioning_scale=float(cn_scale),
                   height=work.height, width=work.width,
                   num_inference_steps=int(sd_steps), generator=g).images[0]
        out = _harmonize_fill(work, out, mask)    # 色彩协调：治洞内外色调跳变（浅色框）
        return paste_back(work, out, mask)
    else:
        # 自由版复用 CN 管线（scale=0 数学等价自由生成）——单管线常驻，根治切换引擎时的显存叠加 MemoryError
        pipe = get_sd_cn()
        pipe.controlnet = _cn_modules()["canny"]
        black = Image.new("RGB", work.size, (0, 0, 0))
        out = pipe(prompt=prompt, negative_prompt=negative, image=work, mask_image=mask,
                   control_image=black, controlnet_conditioning_scale=0.0,
                   height=work.height, width=work.width,
                   num_inference_steps=int(sd_steps), generator=g).images[0]
        out = _harmonize_fill(work, out, mask)
        return paste_back(work, out, mask)

def _inpaint_once(work, mask, engine, prompt, negative, seed, sd_steps, cn_scale, draft_img):
    """包装：每次补全后清显存缓存——防多步会话的显存碎片累积（连续第三次生成 CUDA unknown error 的根治）"""
    out = _inpaint_once_impl(work, mask, engine, prompt, negative, seed, sd_steps, cn_scale, draft_img)
    import torch
    torch.cuda.empty_cache()
    return out

def _op_inpaint(work, editor_value, use_test_hole, engine, prompt, negative, seed, sd_steps, cn_scale, draft_img):
    """补全：量化到8倍数后按掩膜重画（掩膜自动缩放到当前工作图尺寸）"""
    w8 = max((work.width // 8) * 8, 8)
    h8 = max((work.height // 8) * 8, 8)
    if (w8, h8) != work.size:
        work = work.resize((w8, h8), Image.LANCZOS)
    mask = editor_to_mask(editor_value)
    if mask is not None:
        mask = mask.resize(work.size, Image.NEAREST) if mask.size != work.size else mask
        mask = mask.point(lambda x: 255 if x > 127 else 0)   # 强制二值化，防缩放灰度边缘
    elif use_test_hole:
        mask = make_test_hole(work)
    if mask is None:
        return None, "⚠️ 补全需要掩膜：请在画面二涂画笔，或勾选'中央测试洞'"
    out = _inpaint_once(work, mask, engine, prompt, negative, seed, sd_steps, cn_scale, draft_img)
    return out, "补全·" + engine.split("（")[0]

def _op_deblur(work):
    from core.deblur_nafnet import deblur
    return deblur(fit_max_side(work, 1024)), "去模糊·NAFNet"

def _op_face(work, face_engine, face_weight, face_fidelity):
    import cv2
    src = fit_max_side(work, MAX_SIDE_FACE)
    if face_engine == "CodeFormer（保真优先）":
        from core.face_codeformer import restore_face
        out = restore_face(src, fidelity=float(face_fidelity))
    else:
        bgr = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
        _, _, restored = get_gfpgan().enhance(bgr, has_aligned=False, only_center_face=False,
                                              paste_back=True, weight=float(face_weight))
        out = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    return out, "人脸·" + face_engine.split("（")[0]

def _op_sr(work):
    import cv2
    bgr = cv2.cvtColor(np.array(work), cv2.COLOR_RGB2BGR)
    out, _ = get_realesrgan().enhance(bgr, outscale=4)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), "超分×4"

def _chain(hist):
    return " → ".join(h[0] for h in hist)

def apply_op(fn_name, image, editor_value, use_test_hole, hist, engine, prompt, negative,
             seed, sd_steps, cn_scale, draft_img, face_engine, face_weight, face_fidelity, progress=gr.Progress()):
    """执行所选功能：输入=当前结果，输出追加进历史栈"""
    hist = _ensure_hist(image, editor_value, hist)
    if hist is None:
        return None, None, "⚠️ 请先在画面一上传照片"
    try:
        work = Image.fromarray(hist[-1][1])
        progress(0.2, desc=f"执行 {fn_name}…")
        if fn_name == "补全缺损":
            cap = MAX_SIDE_INPAINT if engine == "LaMa（快）" else MAX_SIDE_SD  # SD系引擎用更小上限防大图伪影
            out, msg = _op_inpaint(fit_max_side(work, cap), editor_value, use_test_hole,
                                   engine, prompt, negative, seed, sd_steps, cn_scale, draft_img)
            if out is None:
                return hist, hist[-1][1], msg
        elif fn_name == "去模糊":
            out, msg = _op_deblur(work)
        elif fn_name == "人脸修复":
            out, msg = _op_face(work, face_engine, face_weight, face_fidelity)
        else:
            out, msg = _op_sr(work)
        hist = hist + [(msg if not msg.startswith("⚠️") else fn_name, np.array(out))]
        return hist, hist[-1][1], "✅ " + _chain(hist)
    except Exception as e:
        import traceback
        return hist, hist[-1][1], "❌ 出错：" + traceback.format_exc().strip().splitlines()[-1]

def undo_op(image, editor_value, hist):
    """撤销上一步 = 弹出历史栈末项"""
    hist = _ensure_hist(image, editor_value, hist)
    if hist is None:
        return None, None, "⚠️ 尚无操作"
    if len(hist) <= 1:
        return hist, hist[0][1], "已是最初状态（原图）"
    hist = hist[:-1]
    return hist, hist[-1][1], "↩ 已撤销，当前链：" + _chain(hist)

def reset_op(image, editor_value, hist):
    """重置回原图，清空历史"""
    base = _base_from(image, editor_value)
    if base is None:
        return None, None, "⚠️ 请先上传照片"
    hist = [("原图", base)]
    return hist, base, "🔄 已重置到原图"

def run_recommended(image, editor_value, use_test_hole, hist, engine, prompt, negative,
                    seed, sd_steps, cn_scale, draft_img, face_engine, face_weight, face_fidelity, progress=gr.Progress()):
    """按推荐顺序一键执行（补全→去模糊→人脸→超分）；失败即停并说明原因"""
    hist = _ensure_hist(image, editor_value, hist)
    if hist is None:
        return None, None, "⚠️ 请先在画面一上传照片"
    for i, fn in enumerate(["补全缺损", "去模糊", "人脸修复", "超分×4"]):
        progress(0.1 + 0.22 * i, desc=f"推荐顺序 {i+1}/4：{fn}")
        hist, cur, msg = apply_op(fn, image, editor_value, use_test_hole, hist, engine, prompt,
                                  negative, seed, sd_steps, cn_scale, draft_img,
                                  face_engine, face_weight, face_fidelity)
        if msg.startswith(("⚠️", "❌")):
            return hist, cur, msg + "（后续步骤已停止）"
    return hist, hist[-1][1], "✅ 推荐顺序全部完成：" + _chain(hist)

def update_compare(hist):
    """画面四自动跟随：修复前=原图，修复后=当前结果"""
    if not hist:
        return None, None
    return hist[0][1], hist[-1][1]

def gen_candidates(image, editor_value, use_test_hole, hist, engine, prompt, negative,
                   sd_steps, cn_scale, draft_img, progress=gr.Progress()):
    """多候选：同一掩膜换 3 个种子生成 3 版补全（LaMa 确定性，3 版相同——候选功能面向 SD 引擎）"""
    hist = _ensure_hist(image, editor_value, hist)
    if hist is None:
        return None, [], "⚠️ 请先在画面一上传照片"
    try:
        cap = MAX_SIDE_INPAINT if engine == "LaMa（快）" else MAX_SIDE_SD
        work = fit_max_side(Image.fromarray(hist[-1][1]), cap)
        w8 = max((work.width // 8) * 8, 8)
        h8 = max((work.height // 8) * 8, 8)
        if (w8, h8) != work.size:
            work = work.resize((w8, h8), Image.LANCZOS)
        mask = editor_to_mask(editor_value)
        if mask is not None:
            mask = mask.resize(work.size, Image.NEAREST) if mask.size != work.size else mask
            mask = mask.point(lambda x: 255 if x > 127 else 0)
        elif use_test_hole:
            mask = make_test_hole(work)
        if mask is None:
            return hist, [], "⚠️ 候选生成需要掩膜：请在画面二涂画笔，或勾选'中央测试洞'"
        cands = []
        for i, s in enumerate([42, 137, 2024]):
            progress(0.1 + 0.28 * i, desc=f"候选 {i+1}/3（种子 {s}）…")
            out = _inpaint_once(work, mask, engine, prompt, negative, s, sd_steps, cn_scale, draft_img)
            cands.append(np.array(out))
        return hist, cands, "✅ 已生成 3 版候选（种子 42/137/2024）——点选一张，再按'采用所选候选'"
    except Exception as e:
        import traceback
        return hist, [], "❌ 出错：" + traceback.format_exc().strip().splitlines()[-1]

def adopt_candidate(hist, cands, sel):
    """把选中的候选追加进历史链（成为当前结果，可继续后续步骤）"""
    if not cands:
        return hist, None, "⚠️ 请先生成候选"
    idx = sel if isinstance(sel, int) and 0 <= sel < len(cands) else 0
    hist = hist + [(f"补全·候选{idx+1}", cands[idx])]
    return hist, cands[idx], "✅ 已采用候选 " + str(idx + 1) + "，当前链：" + _chain(hist)

# ---------- 界面 ----------
def build():
    with gr.Blocks(title="AI 照片修复 Demo") as demo:
        gr.Markdown("# AI 照片修复 Demo（第一阶段）\n补全 → 人脸 → 超分 一键管线 | LaMa/SD 双引擎 | 结果可复现")
        with gr.Tab("① 上传照片"):
            with gr.Row():
                inp = gr.Image(label="上传或从示例选择", type="numpy")
                info = gr.Textbox(label="图片信息", interactive=False)
            gr.Examples(examples=[["data/demo_images/old_photo_01.jpg"],
                                  ["data/demo_images/lower_resolution_03.jpg"],
                                  ["data/demo_images/blurred_01.jpg"]],
                        inputs=[inp])
        with gr.Tab("② 破损标注"):
            editor = gr.ImageEditor(label="上传照片并涂抹【要删除重画的内容】（污渍/路人/划痕/缺失处）。\n"
                                            "⚠️ 人脸/画质变清晰请勿涂抹——直接去画面三勾'人脸修复'即可",
                                    sources=["upload"],
                                    brush=gr.Brush(colors=["#ff0000", "#ffff00"], default_color="#ff0000"),
                                    height=480)
            test_hole = gr.Checkbox(label="没有破损？勾选生成中央测试洞", value=False)
            check_btn = gr.Button("🔍 检查标注 —— 看看系统识别到我涂的区域了吗")
            mask_preview = gr.Image(label="系统识别到的破损区域（白=将被修复）", height=280)
            mask_info = gr.Textbox(label="标注状态", interactive=False)
        with gr.Tab("③ 自由修复工作台"):
            gr.Markdown("**自由选择功能与顺序——每次执行都以当前结果为输入。**\n"
                        "推荐顺序：补全 → 去模糊 → 人脸 → 超分（顺序影响效果，如超分后再补全会被拉回工作分辨率）")
            hist_state = gr.State(None)
            cand_state = gr.State([])
            sel_state = gr.State(0)
            fn_choice = gr.Dropdown(["补全缺损", "去模糊", "人脸修复", "超分×4"],
                                    value="补全缺损", label="① 选择要执行的功能")
            with gr.Accordion("参数 · 补全", open=False):
                engine = gr.Radio(["LaMa（快）", "SD Inpainting（自由发挥）", "SD+ControlNet（结构约束）"],
                                  value="LaMa（快）", label="补全引擎")
                cn_scale = gr.Slider(0.0, 1.0, value=0.7, step=0.1,
                                     label="结构约束强度（仅SD+ControlNet生效；0=自由 1=完全遵循，0.7推荐·实测）")
                draft_upload = gr.Image(label="线稿底稿（可选·仅SD+ControlNet生效）：上传修复师线稿/数据集线稿（白底黑线最佳），"
                                          "洞内有真结构线——实验证明优于自动边缘；不上传则自动canny兜底",
                                        type="numpy", height=260)
                btn_multi = gr.Button("🎲 生成3版候选（换种子·建议SD引擎）")
                cand_gallery = gr.Gallery(label="候选结果（点击选中一张）", columns=3, height=280, allow_preview=True)
                btn_adopt = gr.Button("✔ 采用所选候选（进入主流程）")
                with gr.Accordion("SD 参数（仅 SD 引擎生效）", open=False):
                    prompt = gr.Textbox(label="提示词", value="natural, seamless continuation of the surrounding scene, high quality photo")
                    negative = gr.Textbox(label="负面提示词", value="text, watermark, logo, blurry, distortion")
                    seed = gr.Number(label="随机种子（不满意就换数重掷）", value=SD_SEED_DEFAULT, precision=0)
                    sd_steps = gr.Slider(10, 50, value=30, step=5, label="去噪步数")
                gr.Markdown("💡 涂了**人脸缺失区**？引擎选 SD，提示词写 `a clear detailed human face, photorealistic`")
            with gr.Accordion("参数 · 人脸修复", open=False):
                face_engine = gr.Radio(["GFPGAN（质量优先）", "CodeFormer（保真优先）"],
                                       value="GFPGAN（质量优先）", label="人脸引擎")
                with gr.Row():
                    face_weight = gr.Slider(0.2, 0.8, value=0.5, step=0.1,
                                            label="GFPGAN修复强度（低=更保真，高=更清晰）")
                    face_fidelity = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                              label="CodeFormer保真度（高=更像本人）")
            with gr.Row():
                btn_run = gr.Button("▶ ② 执行所选功能", variant="primary")
                btn_undo = gr.Button("↩ 撤销上一步")
                btn_reset = gr.Button("🔄 重置到原图")
            cur_out = gr.Image(label="当前结果（下一步的输入）", height=380)
            status = gr.Textbox(label="执行历史与状态", interactive=False)
            btn_all = gr.Button("🚀 按推荐顺序一键全跑（补全→去模糊→人脸→超分）")
        with gr.Tab("④ 前后对比"):
            with gr.Row():
                before = gr.Image(label="修复前")
                after = gr.Image(label="修复后")

        inp.input(on_load_image, [inp], [editor, info, hist_state])  # 上传后同步编辑器并重置历史
        check_btn.click(on_check_mask, [editor], [mask_preview, mask_info])
        btn_run.click(apply_op, [fn_choice, inp, editor, test_hole, hist_state, engine, prompt,
                                 negative, seed, sd_steps, cn_scale, draft_upload,
                                 face_engine, face_weight, face_fidelity],
                      [hist_state, cur_out, status])
        btn_undo.click(undo_op, [inp, editor, hist_state], [hist_state, cur_out, status])
        btn_reset.click(reset_op, [inp, editor, hist_state], [hist_state, cur_out, status])
        btn_multi.click(gen_candidates, [inp, editor, test_hole, hist_state, engine, prompt,
                                         negative, sd_steps, cn_scale, draft_upload],
                        [hist_state, cand_state, status])
        cand_gallery.select(lambda evt: evt.index, None, [sel_state])
        btn_adopt.click(adopt_candidate, [hist_state, cand_state, sel_state], [hist_state, cur_out, status])
        btn_all.click(run_recommended, [inp, editor, test_hole, hist_state, engine, prompt, negative,
                                        seed, sd_steps, cn_scale, draft_upload,
                                        face_engine, face_weight, face_fidelity],
                      [hist_state, cur_out, status])
        hist_state.change(update_compare, [hist_state], [before, after])  # 画面四自动跟随
    return demo

if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860)
