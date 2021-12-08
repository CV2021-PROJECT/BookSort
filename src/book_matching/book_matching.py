import sys
sys.path.append("..")

from helpers import *
from models import *

down_sampling_rate = 2


def match_book_in_row(
    book1: Book,
    book2: Book,
) -> bool:
    center_x_diff = book2.get_center_x() - book1.get_center_x()
    if center_x_diff > book1.get_width() / 2 or center_x_diff > book2.get_width() / 2:
        return False
    else:
        return True


def match_book_by_image(
    book1: Book,
    book2: Book,
    thr_key_point_num: int = 10,
    thr_match_nms: float = 0.8,
    thr_inlier_pixel: float = 4,
    thr_iou: float = 0.5,
    thr_rms_diff: float = 1.0,
    verbose: bool = False,
) -> bool:
    img1, _ = book1.rect()
    img2, _ = book2.rect()

    img1 = down_sampling(img1, down_sampling_rate)
    img2 = down_sampling(img2, down_sampling_rate)

    if verbose:
        cv2.imshow("img1", img1)
        cv2.imshow("img2", img2)

    if not hasattr(book1, "kp"):
        book1.kp, book1.des = get_kp_desc(img1)
    if not hasattr(book2, "kp"):
        book2.kp, book2.des = get_kp_desc(img2)

    kp1 = book1.kp
    kp2 = book2.kp
    des1 = book1.des
    des2 = book2.des

    try:
        corr_kp1, corr_kp2 = get_corr_keypoints(
            img1, kp1, des1, img2, kp2, des2, thr=thr_match_nms, verbose=verbose
        )
    except:
        return False
    assert len(corr_kp1) == len(corr_kp2), "Key Point matching 잘못된 듯?"

    if len(corr_kp1) < 4:
        #print("악 ketpoint 모잘랑")
        return False

    H_1_to_2 = find_optimal_H(corr_kp2, corr_kp1, thr=thr_inlier_pixel)

    #print(H_1_to_2)

    if type(H_1_to_2) == type(None): return False
    
    img1_on_2 = warp_image(img1, img2, H_1_to_2)

    if verbose:
        cv2.imshow("img1_on_2", img1_on_2)

    match_result = is_identical_image(img1_on_2, img2, thr_iou=thr_iou, thr_rms_diff=thr_rms_diff)

    return match_result


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
            if not match_book_in_row(book1, book2):
                unique_books.append(book)
            book.position = (floor, len(unique_books))
        # end


def convert_to_book_group_list(book_list):
    book_group_list = []
    pos_set = set([book.position for book in book_list])
    
    for pos in pos_set:
        books = list(filter(lambda book: book.position == pos, book_list))
        bg = BookGroup(pos, books)
        book_group_list.append(bg)

    return book_group_list


def save_plt_grid_books(book_group_list, filename="grid_books", vanished_coord=None, moved_coord=None):
    grid_row = max([bg.position[0] for bg in book_group_list])
    grid_col = max([bg.position[1] for bg in book_group_list])

    plt.figure()
    for bg in book_group_list:
        i, j = bg.position
        plt.subplot(grid_row, grid_col, (i-1) * grid_col + j)
        img = bg.get_first().rect()[0]
        pad = int(0.03 * img.shape[0]) 
        if vanished_coord != None and bg.position in vanished_coord:
            imgPad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, (255, 0, 0))
            plt.imshow(imgPad)
        elif moved_coord != None and bg.position in moved_coord:
            imgPad = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, None, (0, 255, 0))
            plt.imshow(imgPad)
        else:
            plt.imshow(img)
        plt.axis('off')
    plt.savefig(f"log/{filename}.jpeg", bbox_inches="tight", pad_inches=0)