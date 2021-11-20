from helpers import *
from models import *


def warp_book_image(img, img_ref, H_to_ref):
    """
    img 를 img_ref 의 좌표 위로 올라가도록 변환하는 함수
    """
    w, h = img_ref.shape[:2]
    return cv2.warpPerspective(img, H_to_ref, dsize=(h, w))


def is_identical_book(img1, img2):
    """
    Args:
        같은 좌표에 올려져있는 두 개의 책 이미지
    """
    assert img1.shape == img2.shape, "사이즈 다릅니당~"
    mask1 = np.any(img1 != 0, axis=2)
    mask2 = np.any(img2 != 0, axis=2)
    mask = (mask1 * mask2).astype(np.uint8) # 0 or 1
    mask = np.tile(np.expand_dims(mask, -1), 3)
    
    img1_norm = normalize(img1 * mask)
    img2_norm = normalize(img2 * mask)

    diff = img1_norm - img2_norm
    rms_diff = np.sqrt(np.average(diff * diff))

    return rms_diff < 1
        

def match_book(
    book1: Book,
    book2: Book,
    thr_match_nms: float = 0.7,
    thr_inlier_pixel: float = 0.1,
    verbose: bool = False,
    ) -> bool:
    """
    Usage:
        book1, book2가 같은 책이니?

    Args:
        thr_match_nms: Non Maximum Supression 할 때 필요한 값 (0~1, 작을수록 빡세게 supress 시키는거임)
        thr_inlier_pixel: Key Point 비교할 때 동일하다고 판단하기 위해 필요한 값 (0~inf)
    """
    img1, _ = book1.rect()
    img2, _ = book2.rect()

    kp1, kp2 = get_corr_keypoints(img1, img2, thr=thr_match_nms, verbose=verbose)
    assert len(kp1) == len(kp2), "Key Point matching 잘못된 듯?"

    if len(kp1) < 4: return False

    H_1_to_2 = find_optimal_H(kp2, kp1, thr=thr_inlier_pixel)
    img1_on_2 = warp_book_image(img1, img2, H_1_to_2)

    if verbose:
        cv2.imshow("img1_on_2", img1_on_2)
        cv2.imshow("img2", img2)
        

    return is_identical_book(img1_on_2, img2)

if __name__ == "__main__":
    image1 = RowImage(cv2.imread("./data/image1_512.jpg"), 0)
    image2 = RowImage(cv2.imread("./data/image2_512.jpg"), 0)
    image3 = RowImage(cv2.imread("./data/image3_512.jpg"), 0)

    book1 = Book(image1, corner=np.array([[210, 46], [272, 48], [254, 494], [202, 488]]))
    book2 = Book(image2, corner=np.array([[214, 64], [272, 64], [274, 500], [204, 494]]))
    book3 = Book(image3, corner=np.array([[190, 148], [268, 156], [264, 468], [216, 472]]))
    book4 = Book(image1, corner=np.array([[93, 34], [178, 35], [170, 488], [102, 482]]))
    book5 = Book(image2, corner=np.array([[127, 67], [194, 76], [169, 507], [86, 497]]))
    book6 = Book(image3, corner=np.array([[58, 134], [154, 136], [194, 462], [138, 462]]))

    print(match_book(book4, book6, 0.5, 0.25, verbose=False))
