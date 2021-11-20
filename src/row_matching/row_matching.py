import sys
sys.path.append("..")

from helpers import *
from models import *


def is_identical_row(img1, img2):
    """
    Args:
        같은 좌표에 올려져있는 두 개의 책장 행 이미지
    """
    assert img1.shape == img2.shape, "사이즈 다릅니당~"
    mask1 = np.any(img1 != 0, axis=2)
    mask2 = np.any(img2 != 0, axis=2)
    iou = np.sum(mask1 * mask2) / np.sum(mask1 + mask2)
    
    mask = (mask1 * mask2).astype(np.uint8) # 0 or 1
    mask = np.tile(np.expand_dims(mask, -1), 3)    
    img1_norm = normalize(img1 * mask)
    img2_norm = normalize(img2 * mask)

    diff = img1_norm - img2_norm
    rms_diff = np.sqrt(np.average(diff * diff))

    print("iou = {}, rms_diff = {}".format(iou, rms_diff))
    return iou > 0.1 and rms_diff < 1

def match_row(
    row1: RowImage,
    row2: RowImage,
    thr_match_nms: float = 0.3,
    thr_inlier_pixel: float = 0.1,
    verbose: bool = False,
    ) -> bool:
    """
    Usage:
        row1, row2가 같은 행이니?

    Args:
        thr_match_nms: Non Maximum Supression 할 때 필요한 값 (0~1, 작을수록 빡세게 supress 시키는거임)
        thr_inlier_pixel: Key Point 비교할 때 동일하다고 판단하기 위해 필요한 값 (0~inf)
    """
    img1 = row1.img
    img2 = row2.img

    kp1, kp2 = get_corr_keypoints(img1, img2, thr=thr_match_nms, verbose=verbose)
    assert len(kp1) == len(kp2), "Key Point matching 잘못된 듯?"

    if len(kp1) < 4: return False

    H_1_to_2 = find_optimal_H(kp2, kp1, thr=thr_inlier_pixel)
    img1_on_2 = warp_image(img1, img2, H_1_to_2)

    if verbose:
        cv2.imshow("img1_on_2", img1_on_2)
        cv2.imshow("img2", img2)
        

    return is_identical_row(img1_on_2, img2)

if __name__ == "__main__":
    row1_1 = RowImage(cv2.imread("./data/row_image/row1_img1.png"), 1)
    row1_2 = RowImage(cv2.imread("./data/row_image/row1_img2.png"), 2)
    row2_1 = RowImage(cv2.imread("./data/row_image/row2_img1.png"), 1)
    row2_2 = RowImage(cv2.imread("./data/row_image/row2_img2.png"), 2)

    print(match_row(row2_1, row2_2, verbose=True))



                               
