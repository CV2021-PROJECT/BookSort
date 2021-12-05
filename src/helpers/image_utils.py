import numpy as np
import cv2


def crop_polygon(image, vertices):
    h = image.shape[0]
    w = image.shape[1]

    mask = np.zeros((h, w)).astype(np.uint8)
    vertices = np.array([vertices])
    rect = cv2.boundingRect(vertices)

    cv2.fillPoly(mask, vertices, 255)
    cropped = cv2.bitwise_and(image, image, mask=mask)
    cropped = cropped[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    return cropped


def normalize(image, mask):
    if np.all(mask == 0): return image
    
    arr = image[mask>0]
    avg = np.average(arr)
    std = np.std(arr)

    return (image - avg) / (std + 1e-6) * mask


def down_sampling(img, rate):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w//rate, h//rate))

    return img


def resize_img(
    img: np.ndarray, target_height: int, target_width: int = None
) -> np.ndarray:
    """이미지의 비율을 유지하면서 높이를 target_height으로 고정한다.

    Args:
        img (np.ndarray): 입력 이미지
        target_height (int): 목표 높이

    Returns:
        np.ndarray: 출력 이미지
    """
    img = img.astype(np.float32)
    if target_width is not None:
        return cv2.resize(img, (target_width, target_height))
    h, w = img.shape[:2]
    new_height = target_height
    scale = h / new_height
    new_width = math.ceil(w / scale)
    return cv2.resize(img, (new_width, new_height))


def is_identical_image(img1, img2, thr_iou=0.5, thr_rms_diff=0.75):
    """
    Args:
        같은 좌표에 올려져있는 두 개의 이미지
    """
    assert img1.shape == img2.shape, "사이즈 다릅니당~"
    mask1 = np.any(img1 != 0, axis=2)
    mask2 = np.any(img2 != 0, axis=2)
    iou = np.sum(mask1 * mask2) / np.sum(mask1 + mask2)
    
    mask = (mask1 * mask2).astype(np.uint8) # 0 or 1
    mask = np.tile(np.expand_dims(mask, -1), 3)

    img1_norm = normalize(img1 * mask, mask)
    img2_norm = normalize(img2 * mask, mask)

    diff = img1_norm - img2_norm
    rms_diff = np.sqrt(np.sum(diff * diff) / (np.sum(mask) + 1e-6))

    #print(f"iou={iou}, rms_diff={rms_diff}")
    return iou > thr_iou and rms_diff < thr_rms_diff
