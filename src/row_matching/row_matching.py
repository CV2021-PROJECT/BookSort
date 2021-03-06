import sys
import cv2
sys.path.append("..")

from helpers import *
from models import *
import matplotlib.pyplot as plt
from tqdm import tqdm

down_sampling_rate = 4
    
def scale_matrix(s):
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
        ])

def match_row(
    row1: RowImage,
    row2: RowImage,
    thr_match_nms: float = 0.5,
    thr_inlier_pixel: float = 3,
    verbose: bool = False,
    ) -> (bool, np.ndarray):
    """
    Usage:
        row1, row2가 같은 행이니?

    Args:
        thr_match_nms: Non Maximum Supression 할 때 필요한 값 (0~1, 작을수록 빡세게 supress 시키는거임)
        thr_inlier_pixel: Key Point 비교할 때 동일하다고 판단하기 위해 필요한 값 (0~inf)
    """
    img1 = down_sampling(row1.img, down_sampling_rate)
    img2 = down_sampling(row2.img, down_sampling_rate)

    if not hasattr(row1, "kp"):
        row1.kp, row1.des = get_kp_desc(img1)
    if not hasattr(row2, "des"):
        row2.kp, row2.des = get_kp_desc(img2)

    kp1 = row1.kp
    kp2 = row2.kp
    des1 = row1.des
    des2 = row2.des

    corr_kp1, corr_kp2 = get_corr_keypoints(
        img1, kp1, des1, img2, kp2, des2, thr=thr_match_nms, verbose=False
    )
    assert len(corr_kp1) == len(corr_kp2), "Key Point matching 잘못된 듯?"

    if len(corr_kp1) < 4: return False, None

    H_1_to_2 = find_optimal_H(corr_kp2, corr_kp1, thr=thr_inlier_pixel)

    if type(H_1_to_2) == type(None): return False, None

    img1_on_2 = warp_image(img1, img2, H_1_to_2)

    if verbose:
        cv2.imshow("img1_on_2", img1_on_2)
        cv2.imshow("img2", img2)
        cv2.waitKey(0)

    # undo down sampling
    H_1_to_2 = scale_matrix(down_sampling_rate) @ H_1_to_2 @ scale_matrix(1/down_sampling_rate)

    return is_identical_image(img1_on_2, img2, thr_iou=0.1, thr_rms_diff=0.75), H_1_to_2

def fill_matching_info(row_image_list):
    """
    Usage:
        RowImage list 넣으면,
        absolute floor 찾아서 무료로 채워드립니다.
        homography in row 찾아서 무료로 채워드립니다.
    """
    sources = [] # all sources in row_image_list
    for row_image in row_image_list:
        source = row_image.source
        if not source in sources:
            sources.append(source)

    vertical_relation_table = [[None for _ in range(len(sources))] for _ in range(len(sources))] # 방향그래프 인접 테이블 // source{i}_floor0의 세로 위치 = source{j}_floor0의 세로 위치 + Edge(i,j)
    horizontal_relation_table = [[None for _ in range(len(row_image_list))] for _ in range(len(row_image_list))] # 방향그래프 인접 테이블 // row_image{i}를 row_image{j}의 좌표계 위로 옮기는 homography H = Edge(i,j)

    # extract info. from matching image-pairs...
    for i in range(len(row_image_list)):
        for j in range(i):
            row_i = row_image_list[i]
            row_j = row_image_list[j]

            isMatch, H = match_row(row_i, row_j)

            if isMatch:
                source_i_index = sources.index(row_i.source)
                source_j_index = sources.index(row_j.source)
                relative_floor_i = row_i.relative_floor
                relative_floor_j = row_j.relative_floor
                vertical_relation_table[source_i_index][source_j_index] = relative_floor_j - relative_floor_i
                vertical_relation_table[source_j_index][source_i_index] = relative_floor_i - relative_floor_j
                horizontal_relation_table[i][j] = H
                horizontal_relation_table[j][i] = get_inverse(H)

    source_floor_map = dict() # Source -> int
    flag_list = [False for _ in range(len(sources))]

    start_index = 0
    queue = [start_index]
    flag_list[start_index] = True
    source_floor_map[sources[start_index]] = 0
    
    while len(queue) > 0:
        curr_index = queue.pop()

        for i in range(len(sources)):
            if vertical_relation_table[i][curr_index] == None : continue
            if flag_list[i] : continue
            
            source_floor_map[sources[i]] = source_floor_map[sources[curr_index]] + vertical_relation_table[i][curr_index]
            flag_list[i] = True
            queue.append(i)

    assert all([flag for flag in flag_list]), "매칭 실패!! ㅠㅠ"

    for row_image in row_image_list:
        row_image.absolute_floor = source_floor_map[row_image.source] + row_image.relative_floor

    min_abs_floor = min([row_image.absolute_floor for row_image in row_image_list])
    for row_image in row_image_list:
        row_image.absolute_floor = row_image.absolute_floor - min_abs_floor + 1

    same_floor_group = dict() # absolute floor -> [Int, ...]

    for i in range(len(row_image_list)):
        row_i = row_image_list[i]
        if not row_i.absolute_floor in same_floor_group:
            same_floor_group[row_i.absolute_floor] = []
        
        same_floor_group[row_i.absolute_floor].append(i)

    flag_list = [False for _ in range(len(row_image_list))]

    for position, group in same_floor_group.items():
        start_index = group[0]
        queue = [start_index]
        flag_list[start_index] = True
        row_image_list[start_index].homography_in_row = np.eye(3)

        while len(queue) > 0:
            curr_index = queue.pop()

            for gi in group:
                if type(horizontal_relation_table[gi][curr_index]) == type(None) : continue
                if flag_list[gi] : continue

                row_image_list[gi].homography_in_row = row_image_list[curr_index].homography_in_row @ horizontal_relation_table[gi][curr_index]
                flag_list[gi] = True
                queue.append(gi)

def display_vertical_matching_result(row_image_list):
    row_group = dict() # absolute floor -> [RowImage, ...]
    for row_image in row_image_list:
        floor = row_image.absolute_floor
        if not floor in row_group:
            row_group[floor] = []
        row_group[floor].append(row_image)

    grid_row = len(row_group)
    grid_col = max([len(group) for _, group in row_group.items()])

    plt.figure()
    for i, (_, group) in enumerate(row_group.items()):
        for j, row_image in enumerate(group):
            plt.subplot(grid_row, grid_col, i*grid_col + j + 1)
            plt.imshow(row_image.img)
            plt.axis('off')
    plt.show()
    plt.close()

def display_horizontal_matching_result(row_image_list):
    row_group = dict() # absolute floor -> [RowImage, ...]
    for row_image in row_image_list:
        floor = row_image.absolute_floor
        if not floor in row_group:
            row_group[floor] = []
        row_group[floor].append(row_image)

    grid_row = len(row_group)
    
    plt.figure()
    for i, (_, group) in enumerate(row_group.items()):
        img_list = []
        H_list = []
        
        for row_image in group:
            img = row_image.img
            H = row_image.homography_in_row
            if type(H) != type(None):
                img_list.append(img)
                H_list.append(H)
                
        merged = merge_images(img_list, H_list)
        
        plt.subplot(grid_row, 1, i+1)
        plt.imshow(merged)
        plt.axis('off')
    plt.show()
    plt.close()
        

if __name__ == "__main__":
    source_list = []
    row_image_list = []
    
    for i in range(7):
        path = "./data/source/source{}.png".format(i)
        img = cv2.imread(path)
        source = Source(img, path)
        source_list.append(img)

        for j in range(2):
            img = cv2.imread("./data/row_image/source{}_floor{}.png".format(i, j))
            row_image = RowImage(img, source, j)
            row_image_list.append(row_image)
        
    fill_matching_info(row_image_list)

    display_horizontal_matching_result(row_image_list)
