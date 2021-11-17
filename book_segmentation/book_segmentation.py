#%%
from typing import List, Tuple
import cv2
import numpy as np
import math
from utils import *


def trim_lines(points: list, y_max: int, x_max: int):
    """이미지의 범위를 넘어가지 않도록 선을 다듬는 함수

    Args:
        points (list): Hough Line을 나타내는 ((x1, y1), (x2, y2)) 형태의 선분 리스트
        y_max (int): 세로 방향 범위
        x_max (int): 가로 방향 범위

    Returns:
        list: 범위를 넘어가지 않도록 다듬은 선분들의 리스트
    """
    slope_threshold = y_max / x_max
    shortened_points = []
    for point in points:
        ((x1, y1), (x2, y2)) = point

        # Slope
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            shortened_points.append(((x1, y_max), (x1, 0)))
            continue

        if abs(m) > slope_threshold:
            # 위아래로 상한
            # x = (y-y1)/m + x1 임을 이용하여 새로운 x값 계산

            # let y = y_max
            new_x1 = math.ceil(((y_max - y1) / m) + x1)
            start_point = (abs(new_x1), y_max)
            # let y = 0
            new_x2 = math.ceil(((0 - y1) / m) + x1)
            end_point = (abs(new_x2), 0)
        elif abs(m) < slope_threshold:
            # y = m * (x - x1) + y1 임을 이용하여 새로운 y값 계산

            # let x = x_max
            new_y1 = math.ceil(m * (x_max - x1) + y1)
            start_point = (x_max, abs(new_y1))
            # let x = 0
            new_y2 = math.ceil(m * (0 - x1) + y1)
            end_point = (0, abs(new_y2))
        else:
            # 정확히 대각선 (가능성 거의 없음)
            if m > 0:
                start_point = (0, y_max)
                end_point = (x_max, 0)
            else:
                start_point = (0, 0)
                end_point(x_max, y_max)
        shortened_points.append((start_point, end_point))
    return shortened_points


def remove_duplicate_lines(sorted_points, horizontal=False):
    """
    Serches for the lines that are drawn
    over each other in the image and returns
    a list of non duplicate line co-ordinates
    """
    last_x1 = 0
    last_y1 = 0
    non_duplicate_points = []
    for point in sorted_points:
        ((x1, y1), (x2, y2)) = point
        if not horizontal:
            if last_x1 == 0:
                non_duplicate_points.append(point)
                last_x1 = x1

            elif abs(last_x1 - x1) >= 25:
                non_duplicate_points.append(point)
                last_x1 = x1
        else:
            if last_y1 == 0:
                non_duplicate_points.append(point)
                last_y1 = y1

            elif abs(last_y1 - y1) >= 25:
                non_duplicate_points.append(point)
                last_y1 = y1

    return non_duplicate_points


def get_points_in_x_and_y(
    hough_lines: np.ndarray,
    max_length: int = 2000,
) -> List[Tuple]:
    """rho, theta로 표현된 Hough Line을 ((x1, y1), (x2, y2)) 꼴의 선분 집합으로 변환한다.

    Args:
        hough_lines (np.ndarray): rho, theta로 표현된 Hough Line들의 리스트
        max_length (int, optional): 선분의 최대 길이. Defaults to 2000.

    Returns:
        List[Tuple]: 변환된 선분들의 집합
    """
    points = []
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 - max_length * b)
        y1 = int(y0 + max_length * a)
        x2 = int(x0 + max_length * b)
        y2 = int(y0 - max_length * a)
        start = (x1, y1)
        end = (x2, y2)
        points.append((start, end))
    return points


def draw_spine_lines(img: np.ndarray, horizontal: bool = False) -> np.ndarray:
    """이미지에 Hough Line을 그린다.

    Args:
        img (np.ndarray): 입력 이미지
        horizontal (bool, optional): 탐지할 Hough Line의 방향. Defaults to False.

    Returns:
        np.ndarray: 출력 이미지
    """
    final_points = detect_spines(img, horizontal=horizontal)
    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        final_image = cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return final_image


def leave_horizontals_only(lines: np.ndarray) -> np.ndarray:
    """Vertical Hough Line을 제거한다.

    Args:
        lines (np.ndarray): 전체 Hough Line 선분의 집합

    Returns:
        np.ndarray: 기울기가 10도 이하인 선분
    """
    horizontals = []
    for line in lines:
        ((x1, y1), (x2, y2)) = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if 0 <= abs(angle) < 10:
            horizontals.append(line)
    return horizontals


def leave_verticals_only(lines: np.ndarray) -> np.ndarray:
    """Horizontal Hough Line을 제거한다.

    Args:
        lines (np.ndarray): 전체 Hough Line 선분의 집합

    Returns:
        np.ndarray: 기울기가 90도 (오차범위 10도) 이내인 선분
    """
    error = 10
    verticals = []
    for line in lines:
        ((x1, y1), (x2, y2)) = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if 90 - error < abs(angle) < 90 + error:
            verticals.append(line)
    return verticals


def detect_spines(img: np.ndarray, horizontal: bool = False):
    """
    Returns a list of lines seperating
    the detected spines in the image
    """
    height, width, _ = img.shape
    blur = cv2.GaussianBlur(img, (11, 11), 2)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 100)
    kernel = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    if horizontal:
        kernel = kernel.T
    img_erosion = cv2.erode(edge, kernel, iterations=1)
    lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 200 if horizontal else 90)
    if lines is None:
        return []
    points = get_points_in_x_and_y(lines)
    points.sort(key=lambda val: val[0][0])
    points = remove_duplicate_lines(points, horizontal=horizontal)

    if horizontal:
        points = leave_horizontals_only(points)
    else:
        points = leave_verticals_only(points)

    points = trim_lines(points, height, width)

    return points


#%%
if __name__ == "__main__":
    paths = [
        "sample_inputs/sample1.jpeg",
        "sample_inputs/sample2.jpeg",
        "sample_inputs/sample3.jpeg",
    ]
    for i, path in enumerate(paths, 1):
        np_image = read_image(path)
        resized = resize_img(np_image, target_height=1000)
        img_with_line = draw_spine_lines(resized, horizontal=False)
        show_image(img_with_line, filename=f"out{i}.jpeg", save=True)
