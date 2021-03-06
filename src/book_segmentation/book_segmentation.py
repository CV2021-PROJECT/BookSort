#%%
import sys

sys.path.append("..")
from typing import List, Tuple
import cv2
import numpy as np
import math
from models import RowImage, Book
from helpers import read_image, show_image


def trim_lines(points: list, y_max: int, x_max: int):
    """이미지의 범위를 넘어가지 않도록 선을 다듬는 함수

    Args:
        points (list): Hough Line을 나타내는 ((x1, y1), (x2, y2)) 형태의 선분 리스트
        y_max (int): 세로 방향 범위
        x_max (int): 가로 방향 범위

    Returns:
        list: 범위를 넘어가지 않도록 다듬은 선분들의 리스트
    """
    slope_threshold = y_max / x_max
    shortened_points = []
    for point in points:
        ((x1, y1), (x2, y2)) = point

        # Slope
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            shortened_points.append(((x1, 0), (x1, y_max)))
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
            print("대각선")
            if m > 0:
                start_point = (0, y_max)
                end_point = (x_max, 0)
            else:
                start_point = (0, 0)
                end_point(x_max, y_max)
        if (
            start_point[0] > x_max
            or end_point[0] > x_max
            or start_point[1] > y_max
            or end_point[1] > y_max
        ):
            continue

        if start_point[1] < end_point[1]:
            shortened_points.append((start_point, end_point))
        else:
            shortened_points.append((end_point, start_point))
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

            elif abs(last_x1 - x1) >= 15:
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


def convert_to_xy(
    hough_lines: np.ndarray,
    max_length: int = 2000,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
        point1 = (x1, y1)
        point2 = (x2, y2)
        if y1 < y2:
            points.append((point1, point2))
        else:
            points.append((point2, point1))
    return points


def draw_hough_lines(img: np.ndarray, horizontal: bool = False) -> np.ndarray:
    """이미지에서 Hough Line을 찾고 그린다.

    Args:
        img (np.ndarray): 입력 이미지
        horizontal (bool, optional): 탐지할 Hough Line의 방향. Defaults to False.

    Returns:
        np.ndarray: 출력 이미지
    """
    final_points = get_hough_lines(img, horizontal=horizontal)
    if len(final_points) == 0:
        print("hough line을 찾을 수 없어요.")
        return img
    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        final_image = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return final_image


def draw_lines_on_image(img: np.ndarray, lines: np.ndarray) -> np.ndarray:
    canvas = img.copy()
    for line in lines:
        ((x1, y1), (x2, y2)) = line
        canvas = cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4)
    return canvas


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
        if 0 <= abs(angle) < 5:
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


def get_hough_lines(img: np.ndarray, horizontal: bool = False):
    """
    Returns a list of lines seperating
    the detected spines in the image
    """
    height, width, _ = img.shape
    blur = cv2.GaussianBlur(img, (11, 11), 2)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 50, 100)
    show_image(edge)
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
    show_image(img_erosion)
    if horizontal:
        lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 15)
    else:
        lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 90)

    if lines is None:
        return []
    points = convert_to_xy(lines)
    points.sort(key=lambda val: val[0][0])

    if not horizontal:
        points = remove_duplicate_lines(points, horizontal=horizontal)

    if horizontal:
        points = leave_horizontals_only(points)
    else:
        points = leave_verticals_only(points)

    points = trim_lines(points, height, width)

    if not horizontal:
        points = finalize_lines(points)

    return points


def finalize_lines(lines: list) -> list:
    """Book Spine 경계를 확정한다.

    Args:
        lines (list): 선분들의 집합

    Returns:
        list: 선분들의 집합. 경계 당 딱 하나만.
    """
    finalized = []

    lines.sort(key=lambda val: val[0][0])  # 시작점(위쪽)의 x좌표 기준으로 정렬
    for i in range(len(lines) - 1):
        line1, line2 = lines[i], lines[i + 1]
        upper_x_gap = line2[0][0] - line1[0][0]
        lower_x_gap = line2[1][0] - line1[1][0]
        if upper_x_gap * lower_x_gap < 0:
            continue  # 선이 교차하는 경우
        if upper_x_gap < 10 or lower_x_gap < 10:
            continue  # 선이 너무 가깝게 붙어있는 경우
        finalized.append(line1)
    finalized.append(lines[-1])

    # 한번 더 필터링
    finalized2 = []
    for i in range(len(finalized) - 1):
        line1, line2 = finalized[i], finalized[i + 1]
        upper_x_gap = line2[0][0] - line1[0][0]
        lower_x_gap = line2[1][0] - line1[1][0]
        if upper_x_gap < 20 or lower_x_gap < 20:
            continue  # 선이 너무 가깝게 붙어있는 경우
        ratio = upper_x_gap / lower_x_gap
        if ratio < 1:
            ratio = 1 / ratio
        if ratio > 1.7:
            continue
        finalized2.append(line1)
    finalized2.append(finalized[-1])
    return finalized2


def line(p1, p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def get_books_list(image_list: List[RowImage]) -> List[Book]:
    books = []
    for row_image in image_list:
        lines = get_hough_lines(row_image.get_resized_img(), horizontal=False)
        image_with_line = draw_lines_on_image(
            img=row_image.get_resized_img(), lines=lines
        )
        show_image(image_with_line)
        for i in range(len(lines) - 1):
            line1, line2 = lines[i], lines[i + 1]
            corner = np.vstack((np.array(line1), np.array(line2)))
            corner[[2, 3]] = corner[[3, 2]]  # 반시계방향으로 변경
            new_book = Book(row_image=row_image, corner=corner)

            # 위아래 부분을 자른다.
            rect, _ = new_book.rect(use_resized_img=True)
            horizontal_lines = get_hough_lines(rect, horizontal=True)
            if len(horizontal_lines) == 0:
                books.append(new_book)
                continue

            horizontal_lines.sort(key=lambda val: val[0][1])  # y좌표 기준으로 정렬
            first_y = horizontal_lines[1][0][1]
            last_y = horizontal_lines[-1][0][1]
            first_line = line([-2000, first_y], [2000, first_y])
            last_line = line([-2000, last_y], [2000, last_y])
            line1 = line(*line1)
            line2 = line(*line2)
            lu = intersection(first_line, line1)
            ld = intersection(last_line, line1)
            rd = intersection(last_line, line2)
            ru = intersection(first_line, line2)
            new_book.corner = np.array([lu, ld, rd, ru]).astype(int)
            books.append(new_book)
    return books


#%%
if __name__ == "__main__":
    # Prepare RowImages
    paths = [
        # "./sample_inputs/sample1.jpeg",
        "./sample_inputs/sample2.jpeg",
        # "./sample_inputs/sample3.jpeg",
        # "./sample_inputs/sample4.jpeg",
        # "./sample_inputs/sample5.jpg",
    ]
    row_images = []
    for i, path in enumerate(paths, 1):
        np_image = read_image(path)
        row_images.append(RowImage(np_image, relative_floor=1))

    books = get_books_list(row_images)
    # for book in books[10:20]:
    #     rect, _ = book.rect(use_resized_img=True)
    # show_image(rect)
