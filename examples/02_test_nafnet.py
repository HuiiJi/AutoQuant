import os
import glob
import cv2
import tqdm
import numpy as np
from PIL import Image
import torch
import onnxruntime
import math
from skimage.metrics import structural_similarity as ssim
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from autoquant import NAFNet_dgf, get_default_qconfig, get_ort_qconfig, get_trt_qconfig, ModelQuantizer, ptq, ONNXExporter, SensitivityAnalyzer, QuantDtype, QScheme, get_qconfig_for_engine
import torch.nn.functional as F

class instanceSegInferenceV2:
    def __init__(self, onnx_model_path=r"C:\Users\75241\Documents\xwechat_files\wxid_7qorwm7awnqp22_977e\msg\file\2026-03\instance_0806.onnx"):
        self.nc = 1
        self.nmsth = 0.6 # 0.45
        self.onnx_model_path = onnx_model_path
        self.c = 32
        self.new_shape = [640, 640]
        self.m_shape = [160, 160]
        self.scoreth = 0.45 # 0.25
        self.maskth = 0.5
        
        # 加载模型
        self.load_model()

    def load_model(self):
        self.sess = onnxruntime.InferenceSession(self.onnx_model_path, providers=['CPUExecutionProvider'])
        self.in_name = [input.name for input in self.sess.get_inputs()][0]
        self.out_name = [output.name for output in self.sess.get_outputs()]

    def xywh2xyxy(self, x):
        y = x.copy()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 2]  # bottom right x
        y[..., 3] = x[..., 3]  # bottom right y

        return y
    
    def custom_round(self, num, lt=False):  
        if lt:
            return math.floor(num)
        else:
            return math.ceil(num) 

    def mask_area(self, mask):
        """计算 mask 的面积."""
        _, binary_mask = cv2.threshold(mask, self.maskth, 1, cv2.THRESH_BINARY)
        return np.count_nonzero(binary_mask)
    
    def mask_iou(self, mask1, mask2, flag='mask'):
        if flag == 'mask':
            _, binary_mask1 = cv2.threshold(mask1, self.maskth, 1, cv2.THRESH_BINARY)
            _, binary_mask2 = cv2.threshold(mask2, self.maskth, 1, cv2.THRESH_BINARY)

            intersection_area = np.logical_and(binary_mask1, binary_mask2).sum()
            union_area = np.logical_or(binary_mask1, binary_mask2).sum()
            iou = intersection_area / union_area
        else:
            # 计算两个边界框的交集部分的坐标
            x1, y1 = max(mask1[0], mask2[0]), max(mask1[1], mask2[1])
            x2, y2 = min(mask1[2], mask2[2]), min(mask1[3], mask2[3])
            # 计算交集的面积
            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
            # 计算两个边界框的并集的面积
            box1_area = (mask1[2] - mask1[0] + 1) * (mask1[3] - mask1[1] + 1)
            box2_area = (mask2[2] - mask2[0] + 1) * (mask2[3] - mask2[1] + 1)
            union_area = box1_area + box2_area - intersection_area
            # 计算IOU
            iou = intersection_area / union_area

        return iou

    def mask_nms(self, img, box, masks, protos,  scores, scoreth, nmsth, mask_weight=0.7, bbox_weight=0.3):
        """基于 mask 面积的非最大抑制（NMS）."""
        # 使用得分阈值过滤掉低于阈值的检测框
        inds = np.array(np.where(scores > scoreth)[0], dtype=int)
        # print('inds:', inds)
        masks = masks[inds]
        _masks = masks @ protos.reshape(self.c, -1)
        _masks = 1 / (1 + np.exp(-_masks))
        masks = _masks.reshape(-1, self.m_shape[0], self.m_shape[1])
        boxmask = np.zeros_like(masks)
        box = box[inds]
        scale = self.m_shape[0]/self.new_shape[0]
        for i, b in enumerate(box):
            b[2] = b[0] + b[2]
            b[3] = b[1] + b[3]
            xmin, ymin, xmax, ymax = int(b[0]*scale), int(b[1]*scale), round(b[2]*scale), round(b[3]*scale)
            # xmin, ymin, xmax, ymax = b.astype(int)
            boxmask[i, ymin:ymax, xmin:xmax] = 1
        masks = boxmask * masks
        scores = scores[inds]
        # 对分数进行排序（降序）  
        sorted_indices = scores.argsort()[::-1]  
        sorted_bboxes = box[sorted_indices]
        sorted_masks = masks[sorted_indices]
        sorted_scores = scores[sorted_indices]

        # 用于存储最终保留的 mask 的索引
        keep = []
        i = 0
        # 遍历排序后的 mask
        while len(sorted_indices) > 0:
            # 选择得分最高的边界框
            idx = sorted_indices[0]
            max_idx = inds[idx]
            keep.append(max_idx)
            bbox_ious = np.array([self.mask_iou(sorted_bboxes[0], sorted_bboxes[i], flag='bbox') for i in range(1, len(sorted_indices))])
            mask_ious = np.array([self.mask_iou(sorted_masks[0], sorted_masks[i], flag='mask') for i in range(1, len(sorted_indices))])
            
            # 标准 NMS：删除与当前高分掩码重叠大于 nmsth 的低分掩码
            # keep_indices = np.where((mask_ious <= nmsth) & (bbox_ious <= nmsth))[0]
            combined_iou = mask_weight * mask_ious + bbox_weight * bbox_ious
            keep_indices = np.where(combined_iou <= nmsth)[0]

            # 更新 sorted_indices 和 sorted_boxes
            sorted_indices = sorted_indices[keep_indices+1]  # 加1是因为从1开始计算IoU
            sorted_masks = sorted_masks[keep_indices+1]
            sorted_bboxes = sorted_bboxes[keep_indices+1]
            sorted_scores = sorted_scores[keep_indices+1]
            i += 1

        return np.array(keep)

    def inference(self, img):
        shape = img.shape[:2]
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LANCZOS4)
        tp, bp = int(round(dh - 0.1)), int(round(dh + 0.1))
        lp, rp = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, tp, bp, lp, rp, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
        vis_img = img.copy()
        img = img.transpose((2, 0, 1))[::-1]
        im = img / 255.
        im = im[None].astype("float32")

        # infer
        y = self.sess.run(self.out_name, {self.in_name: im})

        # 后处理
        prediction = y[0]
        proto = y[1][0]
        nm = prediction.shape[1] - self.nc - 4
        mi = 4 + self.nc
        scores = prediction[0, 4]
        x = prediction[0]

        x = x.transpose(1, 0)
        x = x[scores > self.scoreth]
        box, clss, mask, _ = np.array_split(x, [4, 4 + self.nc, 4 + self.nc + nm], 1)
        score = clss[:, 0]
        box = self.xywh2xyxy(box)

        x = np.concatenate((box, clss[:, 0:1], mask), 1)
        inds = self.mask_nms(vis_img, box.copy(), mask.copy(), proto.copy(), score, self.scoreth, self.nmsth)
        inds = inds.reshape(-1)

        box = box[inds]
        mask = mask[inds]
        score = score[inds]
        for s_box in box:
            s_box[2] = s_box[0] + s_box[2]
            s_box[3] = s_box[1] + s_box[3]

        _mask = mask @ proto.reshape(self.c, -1)
        _mask = 1 / (1 + np.exp(-_mask))
        mask = _mask.reshape(-1, self.m_shape[0], self.m_shape[1])

        scale = self.m_shape[0] / self.new_shape[0]
        dbox = box * scale
        boxmask = np.zeros_like(mask)
        for i, b in enumerate(dbox):
            xmin, ymin, xmax, ymax = b
            xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(round(xmax), self.new_shape[0]), min(round(ymax), self.new_shape[0])
            boxmask[i, ymin:ymax, xmin:xmax] = 1

        mask = boxmask * mask
        mask = mask.transpose(1, 2, 0)

        # +++++++++++++++++++++++++++++++++++++++
        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        # +++++++++++++++++++++++++++++++++++++++
        
        mask = cv2.resize(mask, self.new_shape, interpolation=cv2.INTER_LANCZOS4)

        mask[mask >= self.maskth] = 1
        mask[mask < self.maskth] = 0

        mask = mask[tp: self.new_shape[1] - bp, lp: self.new_shape[0] - rp]

        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)

        if len(mask.shape) == 3:
            mask = mask.transpose((2, 0, 1))
        else:
            mask = mask[None]
            # mask = mask[..., np.newaxis]

        box[:, [0, 2]] -= rp
        box[:, [1, 3]] -= tp
        box[:, :4] /= r
        box[:, :2] = box[:, :2].astype(int)
        box[:, 2:] = np.round(box[:, 2:])
        return box, score, mask



def load_im(img_path: str) -> np.ndarray:
    """加载输入图像并将其转换为神经网络所需的格式。

    Args:
        img_path (str): 图像路径。

    Returns:
        dict: 包含图像数据和尺寸信息的字典。

    Raises:
        ValueError: 如果图像不存在或无法读取。
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Image not found at {img_path}.")

    normalized_path = os.path.normpath(img_path)
    encoded_path = normalized_path.encode("gbk").decode("gbk", "ignore")

    try:
        with open(encoded_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    except:
        img = cv2.imread(img_path)
    return img


def save_from_array(
    inp: np.ndarray,
    output_dir: str,
) -> None:
    """从字典中保存结果到指定的目录下。

    Args:
        output_dir (str): 输出目录路径。
        result_dict (Dict[str, Any]): 结果字典，包含'image', 'prediction', 'gt'三个键。
        image_name (str): 图片名称，用于生成最终保存的文件名。
        suffix (str, optional): 后缀名，用于生成最终保存的文件名。默认是空字符串。

    Returns:
        None
    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)
    if inp.ndim == 2:
        inp_3d = np.repeat(inp, 3, axis=2)
        img = Image.fromarray(inp_3d[...])
    elif inp.ndim == 3:
        img = Image.fromarray(inp[..., ::-1])
    else:
        raise ValueError("Input array must be a 2D or 3D numpy array.")

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    img.save(output_dir, quality=100)

def resize_with_padding(image, target_size):
    height, width = image.shape[:2]
    max_size = max(height, width)
    padded_image = np.zeros((max_size, max_size, 3), dtype=np.uint8)

    pad_top = (max_size - height) // 2
    pad_left = (max_size - width) // 2

    padded_image[pad_top:pad_top + height, pad_left:pad_left + width, :] = image

    # 缩放图像
    # while max_size // 2 > target_size:
    #     max_size = max_size // 2
    #     padded_image = cv2.resize(padded_image, (max_size, max_size))
    padded_image = cv2.resize(padded_image, (target_size, target_size))
    return padded_image


def img_diff(
    ori_img: np.array,
    img: np.array,
    binary_threshold: int = 0,
    need_open: bool = False,
    need_dilate: bool = False,
    name: str = "",
) -> np.array:
    """Image diff, return diff img and binary img.

    Args:
        ori_img (np.array): original image
        img (np.array): diff image
        binary_threshold (int): binary threshold
        name (str): name

    Returns:
        show_img (np.array): show image
        binary (np.array): binary image
        area (int): diff area
    """
    if len(ori_img.shape) == 2:
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
        # ori_img = ori_img[..., None]
        # ori_img = np.repeat(ori_img, 3, axis=2)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img = img[..., None]
        # img = np.repeat(img, 3, axis=2)
    if ori_img.shape != img.shape:
        raise ValueError(
            f"ori_img shape should be equal to img shape, but got {ori_img.shape} and {img.shape}"
        )

    diff_img = cv2.absdiff(ori_img, img)
    diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    binary = np.where(diff_gray > binary_threshold, 255, 0).astype(np.uint8)
    binary_refine = binary.copy()
    if need_open:
        binary_refine = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
    if need_dilate:
        binary_refine = cv2.dilate(
            binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

    contours, _ = cv2.findContours(binary, 1, 2)
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)

    binary_3d = np.repeat(binary_refine[..., None], 3, axis=2)
    # show_img = np.concatenate([ori_img, binary_3d], axis=1)
    # cv2.drawContours(show_img, contours, -1, (0, 0, 255), cv2.FILLED)
    return binary_3d, area

def sort_img_list(root_dir:str):
    # 定义支持的图片扩展名
    img_extensions = ['*.png', '*.jpg', '*.jpeg']
    img_list = []
    # 遍历所有扩展名，递归查找匹配的文件
    for ext in img_extensions:
        img_list.extend(glob.glob(
            os.path.join(root_dir, "**", ext),
            recursive=True
        ))

    # 对找到的文件路径进行排序后返回
    img_list = sorted(img_list)
    return img_list


def save_from_array(
    inp: np.ndarray,
    output_dir: str,
) -> None:
    """从字典中保存结果到指定的目录下。

    Args:
        output_dir (str): 输出目录路径。
        result_dict (Dict[str, Any]): 结果字典，包含'image', 'prediction', 'gt'三个键。
        image_name (str): 图片名称，用于生成最终保存的文件名。
        suffix (str, optional): 后缀名，用于生成最终保存的文件名。默认是空字符串。

    Returns:
        None
    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)
    if inp.ndim == 2:
        img = Image.fromarray(inp[...])
    elif inp.ndim == 3:
        img = Image.fromarray(inp[..., ::-1])
    else:
        raise ValueError("Input array must be a 2D or 3D numpy array.")

    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    img.save(output_dir, quality=100)



def load_pretrained_weights(model, checkpoint_path):
    """智能权重加载方案（支持输入通道变化，如RGB→RGB+mask）"""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取状态字典
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 清理状态字典键名
    new_weights_dict = {}
    for key, value in state_dict.items():
        # 移除编译相关的前缀
        if key.startswith('_orig_mod.'):
            new_weights_dict[key[10:]] = value
        elif key.startswith('module.'):  # 处理DataParallel前缀
            new_weights_dict[key[7:]] = value
        else:
            new_weights_dict[key] = value


    if "intro.weight" in new_weights_dict:
        pretrained_w = new_weights_dict["intro.weight"]
        model_w = model.intro.weight
        if pretrained_w.shape != model_w.shape:
            out_c, in_c, k1, k2 = pretrained_w.shape
            out_c_new, in_c_new, _, _ = model_w.shape
            if out_c == out_c_new and in_c_new == in_c + 1:
                print(f"[INFO] 扩展 intro.weight 通道数: {in_c} -> {in_c_new}")
                new_w = torch.zeros(out_c, in_c_new, k1, k2)
                new_w[:, :in_c, :, :] = pretrained_w  # 拷贝前三个通道
                new_weights_dict["intro.weight"] = new_w

            if out_c == out_c_new and in_c_new == in_c - 1:
                print(f"[INFO] 缩减 intro.weight 通道数: {in_c} -> {in_c_new}")
                new_w = pretrained_w[:, :in_c_new, :, :]
                del new_weights_dict["intro.weight"]
                new_weights_dict["intro.weight"] = new_w
                
            else:
                print(f"[WARN] intro.weight shape mismatch: "
                    f"pretrained={pretrained_w.shape}, model={model_w.shape}")
                # fallback: 使用model自带的初始化，不替换
                del new_weights_dict["intro.weight"]

    if "ending.weight" in new_weights_dict:
        pretrained_w = new_weights_dict["ending.weight"]
        model_w = model.ending.weight
        if pretrained_w.shape != model_w.shape:
            out_c, in_c, k1, k2 = pretrained_w.shape
            out_c_new, in_c_new, _, _ = model_w.shape
            
            if out_c == 3 and out_c_new == 4 and in_c == in_c_new:
                print(f"[INFO] 扩展 ending.weight 输出通道数: {out_c} -> {out_c_new}")
                new_w = torch.zeros(out_c_new, in_c, k1, k2)
                new_w[:out_c, :, :, :] = pretrained_w  # 拷贝前3个输出通道
                # 第4个通道保持为0（随机初始化）
                new_weights_dict["ending.weight"] = new_w

            if out_c == 4 and out_c_new == 3 and in_c == in_c_new:
                print(f"[INFO] 缩减 ending.weight 输出通道数: {out_c} -> {out_c_new}")
                new_w = pretrained_w[:out_c_new, :, :, :]
                del new_weights_dict["ending.weight"]
                new_weights_dict["ending.weight"] = new_w


    if "ending.bias" in new_weights_dict and "ending.weight" in new_weights_dict:
        pretrained_b = new_weights_dict["ending.bias"]
        model_b = model.ending.bias if hasattr(model.ending, 'bias') else None
        
        if model_b is not None and pretrained_b.shape[0] == 3 and model_b.shape[0] == 4:
            print(f"[INFO] 扩展 ending.bias: 3 -> 4")
            new_b = torch.zeros(4)
            new_b[:3] = pretrained_b  # 拷贝前3个偏置
            new_weights_dict["ending.bias"] = new_b
        elif model_b is not None and pretrained_b.shape[0] == 4 and model_b.shape[0] == 3:
            print(f"[INFO] 缩减 ending.bias: 4 -> 3")
            new_b = pretrained_b[:3]
            del new_weights_dict["ending.bias"]
            new_weights_dict["ending.bias"] = new_b

    try:
        model.load_state_dict(new_weights_dict, strict=True)
        print(f"✅ State dict loaded successfully (strict=True) from {checkpoint_path}")

    except Exception as e:
        print(f"⚠️  Strict loading failed: {e}")
        print("🔄 Trying with strict=False...")
        model.load_state_dict(new_weights_dict, strict=False)
        print(f"✅ State dict loaded with strict=False from {checkpoint_path}")

    return model

def resize(img, target_size):
    """
    等比缩放图片，保持宽高比
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def crop_back_tensor(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
@torch.no_grad()
def main_main():
    # 1. 模型加载 (保持不变)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_refine = NAFNet_dgf().to(device)
    model_refine = load_pretrained_weights(model_refine, r"C:\Users\75241\Documents\xwechat_files\wxid_7qorwm7awnqp22_977e\msg\file\2026-03\yiwu_dgf_3c_20260312_duanmian_txue_cuzhezhou_siz512_bs16_ep086_loss0.0133.pth")
    model_refine.eval()
    print("refine")

    model_refine_input_size = (512, 512)

    root_path = r"C:\Users\75241\Desktop\显著色溢"
    new_folder_name = r"C:\Users\75241\Desktop\显著色溢debug"
    os.makedirs(new_folder_name, exist_ok=True)
    
    psd_list = sort_img_list(root_path)
    ins_seg_infer_v2 = instanceSegInferenceV2()

    for idx in tqdm.tqdm(psd_list):
        basename = os.path.basename(idx)
        rel_path = os.path.relpath(idx, root_path)
        new_dir = os.path.join(new_folder_name, os.path.dirname(rel_path))
        os.makedirs(new_dir, exist_ok=True)

        yuantu_img = load_im(idx)
        if yuantu_img is None:
            continue
        
        working_img = yuantu_img.copy()
        # 实例分割提取衣物区域
        ins_boxes, _, ins_mask = ins_seg_infer_v2.inference(yuantu_img)

        for pid, ins_box in enumerate(ins_boxes):
            xmin, ymin, xmax, ymax = ins_box.astype(int)
            # 坐标限制
            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, yuantu_img.shape[1]), min(ymax, yuantu_img.shape[0])
            
            # --- 关键修改 Step 1: 准备数据 ---
            # 原始高清 Crop (H_orig, W_orig, 3)
            img_face_hr = working_img[ymin:ymax, xmin:xmax, :].copy()
            img_face_hr_float = img_face_hr.astype(np.float32) / 255.0
            pad_h, pad_w = img_face_hr.shape[:2]
            target_pad_size = (pad_h, pad_w)

            # Resize 到 512 进模型
            # img_face_resize = resize(img_face_hr, model_refine_input_size[0]) 
            img_face_resize = cv2.resize(img_face_hr, model_refine_input_size, interpolation=cv2.INTER_LINEAR)
            img_tensor_512 = torch.from_numpy(img_face_resize).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():

                res_a_512, res_b_512 = model_refine(img_tensor_512)
                a_up = F.interpolate(res_a_512, size=target_pad_size, mode='bilinear', align_corners=False)
                b_up = F.interpolate(res_b_512, size=target_pad_size, mode='bilinear', align_corners=False)
                a_up = crop_back_tensor(a_up)
                b_up = crop_back_tensor(b_up)

            res_guided_numpy = a_up * img_face_hr_float + b_up
            final_img_base_float = img_face_hr_float + res_guided_numpy 
            final_img_base = np.clip(final_img_base_float * 255.0, 0, 255).round().astype(np.uint8)
            vis_res_base = np.clip(np.abs(res_guided_numpy) * 10 * 255.0, 0, 255).round().astype(np.uint8)

            save_from_array(final_img_base, os.path.join(new_dir, f"{basename}_{pid}_1_Temp.jpg"))
            # save_from_array(vis_res_base, os.path.join(new_dir, f"{basename}_{pid}_3_Res.jpg"))
            save_from_array(img_face_hr, os.path.join(new_dir, f"{basename}_{pid}_Input.jpg"))

def calculate_psnr(img1, img2):
    """计算 PSNR"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """计算 SSIM"""
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    return ssim(img1_gray, img2_gray, data_range=255)


@torch.no_grad()
def quantize_nafnet(
    model_fp32,
    calib_data=None,
    qconfig_type='ort',
    device='cuda'
):
    """
    量化 NAFNet 模型
    
    Args:
        model_fp32: FP32 模型
        calib_data: 校准数据（list of tensors）
        qconfig_type: 'ort' 或 'trt'
        device: 设备
    """
    print("=" * 70)
    print("🔧 开始量化 NAFNet 模型")
    print("=" * 70)
    
    # 准备 QConfig
    if qconfig_type == 'ort':
        qconfig = get_ort_qconfig()
        print(f"✅ 使用 ONNX Runtime 最佳配置")
    elif qconfig_type == 'trt':
        qconfig = get_trt_qconfig()
        print(f"✅ 使用 TensorRT 最佳配置")
    else:
        qconfig = get_default_qconfig()
        print(f"✅ 使用默认配置")
    
    print(f"\n[1/6] 开始全OP敏感度分析 (Engine: {qconfig_type})...")
    analyzer = SensitivityAnalyzer(model_fp32, qconfig)
    dummy_input = [torch.randn(1, 3, 64, 64) for _ in range(10)]
    
    # 分析所有可量化层（全OP分析）
    sensitivity_scores = analyzer.analyze(
        dummy_input,
        calib_data=calib_data
    )
    
    # ========================================================================
    # 步骤 3: 保存完整报表和图表
    # ========================================================================
    print("[2/6] 保存敏感度分析结果...")
    output_dir = os.path.join(project_root, "asset")
    analyzer.save_results(output_dir)
    
    # ========================================================================
    # 步骤 4: 获取自动推荐跳过的层（无需手动设定！）
    # ========================================================================
    quantizable_layers, skip_layers, recommendation_info = analyzer.get_recommended_layers()
    
    # 打印自动推荐信息
    if recommendation_info:
        print(f"\n    🤖 自动推荐结果:")
        print(f"       方法: {recommendation_info.get('description', 'auto')}")
        print(f"       Skip层数: {recommendation_info.get('skip_count', 0)}")
        print(f"       Skip占比: {recommendation_info.get('skip_percent', 0):.1f}%")
        print(f"       覆盖敏感度: {recommendation_info.get('coverage', 0):.1%}")
        
        alternatives = recommendation_info.get('alternatives', {})
    
    # ========================================================================
    # 步骤 5: PTQ 量化
    # ========================================================================
    print("\n[3/6] 准备量化模型...")
    quantizer = ModelQuantizer(model_fp32, qconfig)
    prepared_model = quantizer.prepare(skip_layers=set(skip_layers))
    print("\n" + "=" * 70)
    print("✅ NAFNet 量化完成！")
    print("=" * 70)
    
    return prepared_model


@torch.no_grad()
def compare_models(
    model_fp32,
    model_quantized,
    test_img,
    device='cuda'
):
    """
    对比 FP32 和量化模型的效果
    
    Args:
        model_fp32: FP32 模型
        model_quantized: 量化模型
        test_img: 测试图像 (numpy array)
        device: 设备
    
    Returns:
        dict: 包含对比结果
    """
    print("\n" + "=" * 70)
    print("📊 开始对比 FP32 vs 量化模型")
    print("=" * 70)
    
    # 准备输入
    h, w = test_img.shape[:2]
    img_resized = cv2.resize(test_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # 1. FP32 推理
    print("\n[1/2] FP32 模型推理...")
    model_fp32.eval()
    res_a_fp32, res_b_fp32 = model_fp32(img_tensor)
    a_up_fp32 = F.interpolate(res_a_fp32, size=(h, w), mode='bilinear', align_corners=False)
    b_up_fp32 = F.interpolate(res_b_fp32, size=(h, w), mode='bilinear', align_corners=False)
    a_up_fp32_np = crop_back_tensor(a_up_fp32)
    b_up_fp32_np = crop_back_tensor(b_up_fp32)
    
    # 2. 量化模型推理
    print("\n[2/2] 量化模型推理...")
    model_quantized.eval()
    res_a_quant, res_b_quant = model_quantized(img_tensor)
    a_up_quant = F.interpolate(res_a_quant, size=(h, w), mode='bilinear', align_corners=False)
    b_up_quant = F.interpolate(res_b_quant, size=(h, w), mode='bilinear', align_corners=False)
    a_up_quant_np = crop_back_tensor(a_up_quant)
    b_up_quant_np = crop_back_tensor(b_up_quant)
    
    # 计算差异
    print("\n[对比结果]")
    print("-" * 70)
    
    img_float = test_img.astype(np.float32) / 255.0
    
    # FP32 结果
    res_guided_fp32 = a_up_fp32_np * img_float + b_up_fp32_np
    final_img_fp32 = img_float + res_guided_fp32
    final_img_fp32_uint8 = np.clip(final_img_fp32 * 255.0, 0, 255).round().astype(np.uint8)
    
    # 量化结果
    res_guided_quant = a_up_quant_np * img_float + b_up_quant_np
    final_img_quant = img_float + res_guided_quant
    final_img_quant_uint8 = np.clip(final_img_quant * 255.0, 0, 255).round().astype(np.uint8)
    
    # 计算指标
    psnr_a = calculate_psnr(a_up_fp32_np * 255, a_up_quant_np * 255)
    psnr_b = calculate_psnr(b_up_fp32_np * 255, b_up_quant_np * 255)
    psnr_final = calculate_psnr(final_img_fp32_uint8, final_img_quant_uint8)
    
    ssim_a = calculate_ssim((a_up_fp32_np * 255).astype(np.uint8), (a_up_quant_np * 255).astype(np.uint8))
    ssim_b = calculate_ssim((b_up_fp32_np * 255).astype(np.uint8), (b_up_quant_np * 255).astype(np.uint8))
    ssim_final = calculate_ssim(final_img_fp32_uint8, final_img_quant_uint8)
    
    print(f"   最终图像 PSNR: {psnr_final:.2f} dB")
    print(f"   最终图像 SSIM: {ssim_final:.4f}")
    print("-" * 70)
    
    return {
        'fp32': final_img_fp32_uint8,
        'quantized': final_img_quant_uint8,
        'psnr_final': psnr_final,
        'ssim_final': ssim_final,
    }


def prepare_calib_data_from_images(img_list, device, num_calib=5):
    """
    从图像列表准备校准数据
    
    Args:
        img_list: 图像路径列表
        device: 设备
        num_calib: 校准样本数量
    """
    calib_data = []
    num_used = min(num_calib, len(img_list))
    
    print(f"\n[准备校准数据] 使用 {num_used} 张测试集图像")
    
    for i in range(num_used):
        img_path = img_list[i]
        img = load_im(img_path)
        
        # Resize 到 512x512
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # 转换为 tensor
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        calib_data.append(img_tensor)
        
        print(f"   [{i+1}/{num_used}] {os.path.basename(img_path)}")
    
    return calib_data


@torch.no_grad()
def main_with_quantization():
    """
    主函数：包含完整的量化和对比流程
    """
    print("=" * 70)
    print("🚀 NAFNet 量化与对比完整流程")
    print("=" * 70)
    
    # 1. 模型加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[步骤 1] 加载 FP32 模型到 {device}")
    model_fp32 = NAFNet_dgf().to(device)
    model_fp32 = load_pretrained_weights(
        model_fp32, 
        r"C:\Users\75241\Documents\xwechat_files\wxid_7qorwm7awnqp22_977e\msg\file\2026-03\yiwu_dgf_3c_20260312_duanmian_txue_cuzhezhou_siz512_bs16_ep086_loss0.0133.pth"
    )
    model_fp32.eval()
    
    # 2. 准备校准数据（使用你的测试集！）
    print("\n[步骤 2] 准备校准数据")
    root_path = r"C:\Users\75241\Desktop\显著色溢"
    psd_list = sort_img_list(root_path)
    
    if len(psd_list) == 0:
        print("⚠️  没有找到测试图像！使用 dummy 数据")
        calib_data = [torch.randn(1, 3, 512, 512).to(device) for _ in range(5)]
    else:
        # 使用你的测试集图像进行校准！
        calib_data = prepare_calib_data_from_images(psd_list, device, num_calib=min(10, len(psd_list)))
    
    print(f"   校准数据数量: {len(calib_data)}")
    
    # 3. 量化模型
    model_quantized = quantize_nafnet(
        model_fp32,
        calib_data=calib_data,
        qconfig_type='ort',
        device=device
    )
    
    # 4. 准备测试数据
    print("\n[步骤 3] 准备测试图像")
    root_path = r"C:\Users\75241\Desktop\显著色溢"
    psd_list = sort_img_list(root_path)
    
    if len(psd_list) == 0:
        print("⚠️  没有找到测试图像！使用 dummy 图像")
        test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    else:
        test_img = load_im(psd_list[0])
        print(f"   测试图像: {os.path.basename(psd_list[0])}")
    
    # 5. 对比模型
    results = compare_models(
        model_fp32,
        model_quantized,
        test_img,
        device=device
    )
    
    # 6. 保存结果
    print("\n[步骤 4] 保存对比结果")
    output_dir = r"C:\Users\75241\Desktop\显著色溢_quant_compare"
    os.makedirs(output_dir, exist_ok=True)
    
    save_from_array(results['fp32'], os.path.join(output_dir, "result_fp32.jpg"))
    save_from_array(results['quantized'], os.path.join(output_dir, "result_quantized.jpg"))
    save_from_array(test_img, os.path.join(output_dir, "input.jpg"))
    
    # 保存对比图
    diff_img = np.abs(results['fp32'].astype(np.int32) - results['quantized'].astype(np.int32)).astype(np.uint8)
    diff_img = np.clip(diff_img * 10, 0, 255).astype(np.uint8)
    save_from_array(diff_img, os.path.join(output_dir, "difference.jpg"))
    
    print(f"   结果已保存到: {output_dir}")
    
    # # 7. 可选：导出 ONNX
    # print("\n[可选步骤] 导出 ONNX 模型")
    # try:
    #     dummy_input = torch.randn(1, 3, 512, 512).to(device)
    #     onnx_path = os.path.join(output_dir, "nafnet_quantized.onnx")
    #     ONNXExporter.export(
    #         model_quantized,
    #         dummy_input,
    #         onnx_path,
    #         opset_version=18,
    #         verbose=False
    #     )
    # except Exception as e:
    #     print(f"⚠️  ONNX 导出跳过: {e}")
    
    # print("\n" + "=" * 70)
    # print("✅ 所有流程完成！")
    # print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NAFNet 推理和量化")
    parser.add_argument('--mode', type=str, default='fp32', 
                       choices=['fp32', 'quant', 'compare'],
                       help="运行模式: fp32 (仅FP32), quant (仅量化), compare (对比)")
    
    args = parser.parse_args()
    
    if args.mode == 'fp32':
        print("\n🎯 运行 FP32 推理模式")
        main_main()
    elif args.mode == 'quant':
        print("\n🎯 运行量化模式")
        main_with_quantization()
    elif args.mode == 'compare':
        print("\n🎯 运行对比模式")
        main_with_quantization()

