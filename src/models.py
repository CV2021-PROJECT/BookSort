from typing import List, Tuple
import numpy as np
import cv2
from helpers import crop_polygon, resize_img

class Source:
    """촬영한 한 장의 이미지"""

    def __init__(self, img: np.ndarray, path: str):
        self.img = img
        self.path = path

    def __eq__(self, other):
        return other.path == self.path

    def __hash__(self):
        return hash(self.path)


class RowImage:
    """책장의 한 행을 나타내는 이미지 모델"""

    def __init__(self, img: np.ndarray, source: Source, relative_floor: int) -> None:
        """
        Args:
            img (np.ndarray): numpy 배열로 나타낸 이미지 파일
            source (Source): 원본 이미지
            relative_floor (int): 상대적 층수
            absolute_floor (int): 절대적 층수
            homography_in_row (np.ndarray): 기준 좌표계로의 homography
        """
        self.img = img
        self.source = source
        self.resized_img: np.ndarray = None
        self.relative_floor = relative_floor
        self.absolute_floor = None  # 생성 시점에서는 absolute_floor 정보를 알 수 없다.
        self.homography_in_row = None # 마찬가지로, 생성 시점에서는 모르니까

    def __str__(self):
        return "고향: {}, 상대적 위치: {}, 절대적 위치: {}"\
               .format(self.source.path, self.relative_floor, self.absolute_floor)

    def update_absolute_floor(self, value: int):
        self.absolute_floor = value

    def get_resized_img(self):
        if self.resized_img is None:
            self.resized_img = resize_img(self.img, target_height=1000)
            return self.resized_img
        return self.resized_img


class Book:
    def __init__(
        self,
        row_image: RowImage,
        corner: np.ndarray,
    ) -> None:
        """
        Args:
            row_image (RowImage): 책의 출처 이미지를 나타낸다. 높이는 1000으로 고정되어 있다.
            corner (np.ndarray): 책의 꼭지점 4개의 좌표를 (반)시계방향으로 (4, 2) 행렬로 나타낸다.
            position (int, int): 책의 위치(가로, 세로)
        """
        self.row_image = row_image
        self.corner = corner
        self.position = None

    def get_global_corner(self):
        """
        Returns:
            global_position: row 안에서의 위치
        """
        if not hasattr(self, "global_corner"):
            self.global_corner = cv2.perspectiveTransform(
                self.corner.reshape(-1, 1, 2).astype(np.float32),
                self.row_image.homography_in_row.astype(np.float32),
            )
        return self.global_corner.reshape(-1, 2)

    def get_center_x(self):
        center = np.average(self.get_global_corner(), axis=0)
        return center[0]

    def get_width(self):
        x_min = np.min(self.get_global_corner()[:, 0])
        x_max = np.max(self.get_global_corner()[:, 0])
        return x_max - x_min

    def rect(self):
        """
        Returns:
            crop: RGB 이미지
            mask: 0/255 이미지
        """
        img = self.row_image.img
        if not hasattr(self, "crop"):
            self.crop = crop_polygon(img, self.corner)
        if not hasattr(self, "mask"):
            self.mask = crop_polygon(np.ones(img.shape).astype(np.uint8) * 255, self.corner)

        return self.crop, self.mask


class BookGroup:
    def __init__(self, position, book_list):
        self.position = position
        self.book_list = book_list

    def get_first(self):
        return self.book_list[0]

    def add_book(self, book):
        self.book_list.append(book)


class BookTrajectory:
    def __init__(self, bg_before, bg_after):
        self.bg_before = bg_before
        self.bg_after = bg_after

    def __str__(self):
        if type(self.bg_before) != type(None):
            before_text = str(self.bg_before.position)
        else:
            before_text = "?"
            
        if type(self.bg_after) != type(None):
            after_text = str(self.bg_after.position)
        else:
            after_text = "?"

            
        return "{} -> {}".format(before_text, after_text)
