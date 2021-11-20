from PIL import Image, ImageOps
import numpy as np
import os
import math
import cv2
from matplotlib import pyplot as plt


def show_image(img: np.ndarray, filename: str = "sample", save=False):
    logdir = os.path.join("data", "sample_outputs")
    if img is not None:
        plt.figure(dpi=200)
        plt.tight_layout(pad=0)
        plt.imshow(img.astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        if save:
            plt.savefig(
                os.path.join(logdir, f"{filename}.jpeg"),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.show()


def read_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    np_img = np.array(img)
    return np_img


def resize_img(img: np.ndarray, target_height: int) -> np.ndarray:
    """이미지의 비율을 유지하면서 높이를 target_height으로 고정한다.

    Args:
        img (np.ndarray): 입력 이미지
        target_height (int): 목표 높이

    Returns:
        np.ndarray: 출력 이미지
    """
    h, w, _ = img.shape
    new_height = target_height
    scale = h / new_height
    new_width = math.ceil(w / scale)
    return cv2.resize(img, (new_width, new_height))
