# core/face_codeformer.py —— CodeFormer 人脸修复（保真优先引擎，与 GFPGAN 互补）
# 与 GFPGAN 的本质区别：CodeFormer 的 w 参数专为"保真 vs 质量"权衡设计——
#   fidelity: 0=质量优先（清晰但可能不像本人） 1=保真优先（更像本人但保留部分退化）
#   ★ 方向与 GFPGAN 的 weight 相反（GFPGAN: 高=更强修复），参数名特意区分防混淆
# 权重依赖：CodeFormer/weights/CodeFormer/codeformer.pth（主模型，显式路径）
#           weights/facelib/{detection_Resnet50_Final, parsing_parsenet}.pth（检测/解析，须在项目根）
import numpy as np
import torch
import cv2
from PIL import Image

_state = {}

def _load():
    if "net" in _state:
        return
    from codeformer.basicsr.archs.codeformer_arch import CodeFormer
    from codeformer.basicsr.utils import img2tensor, tensor2img
    from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                     connect_list=["32", "64", "128", "256"]).to(device)
    ckpt = torch.load("CodeFormer/weights/CodeFormer/codeformer.pth", map_location=device)["params_ema"]
    net.load_state_dict(ckpt)
    net.eval()
    helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1),
                               det_model="retinaface_resnet50", use_parse=True, device=device)
    _state.update(net=net, helper=helper, device=device,
                  img2tensor=img2tensor, tensor2img=tensor2img)

def restore_face(img: Image.Image, fidelity: float = 0.75) -> Image.Image:
    """整图输入：检测人脸 → CodeFormer 修复 → 贴回。无脸时原样返回。"""
    _load()
    from torchvision.transforms.functional import normalize
    net, helper, device = _state["net"], _state["helper"], _state["device"]
    img2tensor, tensor2img = _state["img2tensor"], _state["tensor2img"]

    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    helper.clean_all()
    helper.read_image(bgr)
    n = helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    if n == 0:
        return img
    helper.align_warp_face()
    for cropped in helper.cropped_faces:
        t = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
        normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        t = t.unsqueeze(0).to(device)
        with torch.no_grad():
            out = net(t, w=float(fidelity), adain=True)[0]
        restored = tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
        helper.add_restored_face(restored, cropped)
    helper.get_inverse_affine(None)
    result = helper.paste_faces_to_input_image()
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
