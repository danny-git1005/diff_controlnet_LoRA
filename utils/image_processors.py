import cv2
import numpy as np
from PIL import Image
import torch
from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector, PidiNetDetector

# 處理器：Canny
class CannyProcessor:
    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 轉灰階
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

# 處理器：Depth
class DepthProcessor:
    def __init__(self):
        self.detector = None

    def __call__(self, image):
        if self.detector is None:
            self.detector = MidasDetector.from_pretrained("lllyasviel/ControlNet")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.detector(image)

# 處理器：Pose
class PoseProcessor:
    def __init__(self):
        self.detector = None

    def __call__(self, image):
        if self.detector is None:
            self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.detector(image)

# 處理器：Segmentation
class SegmentationProcessor:
    def __init__(self):
        self.detector = None

    def __call__(self, image):
        if self.detector is None:
            self.detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.detector(image)

# ✅ 處理器：None（不做處理，原樣返回）
class NoneProcessor:
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return image

# ✅ 使用 Lazy Init 避免初始化模型導致 multiprocessing crash
def get_processor_for_condition(condition_type):
    if condition_type == "canny":
        return CannyProcessor()
    elif condition_type == "depth":
        return DepthProcessor()
    elif condition_type == "pose":
        return PoseProcessor()
    elif condition_type == "seg":
        return SegmentationProcessor()
    elif condition_type == None:  # ✅ 新增支援 "none"
        return NoneProcessor()
    else:
        raise ValueError(f"Unsupported condition type: {condition_type}. Must be one of ['canny', 'depth', 'pose', 'seg', 'none']")
