import numpy as np
import cv2
import math
import time


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

sift = cv2.xfeatures2d.SIFT_create()


def tuple_to_homogeneous(t):
    return np.expand_dims(np.array(t + (1,)), -1)


def homogeneous_to_tuple(h):
    assert len(h.shape) in [1, 2]
    if len(h.shape) == 1:
        return h[:-1] / h[-1]
    else:
        return np.squeeze(h[:-1], -1) / h[-1]


def get_inverse(H):
    return np.linalg.inv(H)


def get_H(p1, p2):
    A = []
    zero3 = np.zeros((1, 3))

    for (ui, vi), (xi, yi) in zip(p1, p2):
        xy_hmg_row = tuple_to_homogeneous((xi, yi)).transpose()

        row1 = [xy_hmg_row, zero3, -ui * xy_hmg_row]
        row2 = [zero3, xy_hmg_row, -vi * xy_hmg_row]

        A.append(row1)
        A.append(row2)

    A = np.block(A)

    u, s, v = np.linalg.svd(A)
    return v[-1].reshape((3, 3))


def get_inliers_error(p1, p2, H, thr):
    try:
        H_inv = get_inverse(H)
    except:
        return [], [], float("inf")

    p1 = np.array(p1).reshape(-1, 1, 2)
    p2 = np.array(p2).reshape(-1, 1, 2)

    H_p1 = cv2.perspectiveTransform(p1, H_inv)
    H_p2 = cv2.perspectiveTransform(p2, H)

    e1 = np.square(H_p1 - p2)
    e1 = np.sum(e1, axis=(1, 2))
    e2 = np.square(H_p2 - p1)
    e2 = np.sum(e2, axis=(1, 2))
    e = np.sqrt((e1 + e2) / 2)

    inlier_index = np.where(e < thr)[0]
    inliers_1 = p1[inlier_index]
    inliers_2 = p2[inlier_index]

    inliers_1 = [(pt[0,0], pt[0,1]) for pt in inliers_1]
    inliers_2 = [(pt[0,0], pt[0,1]) for pt in inliers_2]

    return inliers_1, inliers_2, np.sum(e)


def find_optimal_H(p1, p2, thr, p_ransac=0.99, runtime_bound=1000):
    H_optimal = None
    N, count = float("inf"), 0
    num_of_inliers_max = -float("inf")
    best_inliers_1, best_inliers_2 = None, None
    error = float("inf")

    while count < N:
        indices = np.random.choice(len(p1), 4, replace=False)
        p1_sub = [p1[i] for i in indices]
        p2_sub = [p2[i] for i in indices]

        H = get_H(p1_sub, p2_sub)
        inliers_1, inliers_2, e = get_inliers_error(p1, p2, H, thr)

        n_in = np.shape(inliers_1)[0]  # = np.shape(iinliers_2)[0]
        if n_in > num_of_inliers_max or (n_in == num_of_inliers_max and e < error):
            num_of_inliers_max = n_in
            best_inliers_1 = inliers_1.copy()
            best_inliers_2 = inliers_2.copy()
            H_optimal = H
            error = e

        eps = 1 - n_in / len(p1)
        N = (
            math.log(1 - p_ransac) / math.log(1 - (1 - eps) ** 4)
            if 0 < eps < 1
            else np.float('inf')
        )
        count += 1

        if count > runtime_bound: return None

    #print("hello1", count, N)

    return get_H(best_inliers_1, best_inliers_2)


def get_kp_desc(img):
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def get_corr_keypoints(img1, kp1, des1, img2, kp2, des2, thr, verbose=False):
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < thr * n.distance:
            matchesMask[i] = [1, 0]

    if verbose:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=0,
        )
        match = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        cv2.imshow("match", match)
        cv2.waitKey(0)

    matched_kp1 = []
    matched_kp2 = []
    for i, (m, _) in enumerate(matches):
        if matchesMask[i][0] == 1:
            new_kp1 = kp1[m.queryIdx].pt
            new_kp2 = kp2[m.trainIdx].pt
            flag = False
            for old_kp1, old_kp2 in zip(matched_kp1, matched_kp2):
                if old_kp1 == new_kp1 and old_kp2 == new_kp2:
                    flag = True
                    break
            if not flag:
                matched_kp1.append(new_kp1)
                matched_kp2.append(new_kp2)

    return matched_kp1, matched_kp2

def warp_image(img, img_ref, H_to_ref):
    """
    img 를 img_ref 의 좌표 위로 올라가도록 변환하는 함수
    """
    w, h = img_ref.shape[:2]
    return cv2.warpPerspective(img, H_to_ref, dsize=(h, w))


def merge_images(img_list, H_list):
    all_corners = []
    for img, H in zip(img_list, H_list):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1,1,2)
        corners = cv2.perspectiveTransform(corners, H)
        all_corners.append([[corners]])

    all_corners = np.block(all_corners)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    Ht = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    result = np.zeros((ymax-ymin, xmax-xmin, 3)).astype(np.uint8)
    for img, H in zip(img_list, H_list):
        warped = cv2.warpPerspective(img, Ht @ H, dsize=(xmax-xmin, ymax-ymin))
        intersection = np.logical_and(warped != 0, result != 0)
        error_rms = np.sqrt(np.sum(np.power(result[intersection] - warped[intersection], 2)) / (np.sum(intersection) + 1e-6))
        if error_rms < 10:
            result[warped != 0] = warped[warped != 0]

    return result
