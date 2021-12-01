import sys
sys.path.append("..")

from helpers import *
from models import *


def is_identical_image(img1, img2):
    """
    Args:
        같은 좌표에 올려져있는 두 개의 책 이미지
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
    rms_diff = np.sqrt(np.sum(diff * diff) / (np.sum(mask) + 1e-6))

    print("iou = {}, rms_diff = {}".format(iou, rms_diff))
    return iou > 0.5 and rms_diff < 0.75


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
    center_x_diff = book2.get_center_x() - book1.get_center_x()
    if center_x_diff > book1.get_width() / 2 or center_x_diff > book2.get_width() / 2:
        return False
    else:
        return True

    img1, _ = book1.rect()
    img2, _ = book2.rect()

    if not hasattr(book1, "kp"):
        book1.kp, book1.des = get_kp_desc(img1)
    if not hasattr(book2, "des"):
        book2.kp, book2.des = get_kp_desc(img2)

    kp1 = book1.kp
    kp2 = book2.kp
    des1 = book1.des
    des2 = book2.des

    corr_kp1, corr_kp2 = get_corr_keypoints(
        img1, kp1, des1, img2, kp2, des2, thr=thr_match_nms, verbose=False
    )
    assert len(corr_kp1) == len(corr_kp2), "Key Point matching 잘못된 듯?"

    if len(corr_kp1) < 4: return False, None

    H_1_to_2 = find_optimal_H(corr_kp2, corr_kp1, thr=thr_inlier_pixel)

    if type(H_1_to_2) == type(None): return False
    
    img1_on_2 = warp_image(img1, img2, H_1_to_2)

    if verbose:
        cv2.imshow("img1_on_2", img1_on_2)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)

    return is_identical_image(img1_on_2, img2)


def fill_position_info(book_list):
    min_floor = min([book.row_image.absolute_floor for book in book_list]) # should be 0
    max_floor = max([book.row_image.absolute_floor for book in book_list])

    for floor in range(min_floor, max_floor+1):
        books_in_floor = list(filter(lambda book: book.row_image.absolute_floor == floor, book_list))
        books_in_floor.sort(key=lambda book: book.get_center_x())

        # 위치, 이미지 를 둘다 고려해서 중복이라고 판단되는 책들은 1개만 남겨둬
        # 지금은 homography 믿고 위치만 고려하는 중..
        unique_books = [books_in_floor[0]]
        books_in_floor[0].position = (floor, 1)
        
        for book in books_in_floor[1:]:
            book1 = unique_books[-1]
            book2 = book
            if not match_book(book1, book2):
                unique_books.append(book)
            book.position = (floor, len(unique_books))
        # end

def show_grid_books(book_list):
    grid_row = max([book.position[0] for book in book_list])
    grid_col = max([book.position[1] for book in book_list])

    plt.figure()
    for book in book_list:
        i, j = book.position
        plt.subplot(grid_row, grid_col, (i-1) * grid_col + j)
        plt.imshow(book.rect()[0])
        plt.axis('off')
    plt.show()
    plt.close()
