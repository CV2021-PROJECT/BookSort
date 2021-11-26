from typing import List, Tuple
import numpy as np
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
            horizontal_position (float): 같은 층들끼리의 상대적인 가로 위치
        """
        self.img = img
        self.source = source
        self.resized_img: np.ndarray = None
        self.relative_floor = relative_floor
        self.absolute_floor = None  # 생성 시점에서는 absolute_floor 정보를 알 수 없다.
        self.horizontal_position = None # 마찬가지로, 생성 시점에서는 모르니까

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
        """
        self.row_image = row_image
        self.corner = corner

    def rect(self):
        """
        Returns:
            crop: RGB 이미지
            mask: 0/255 이미지
        """
        img = self.row_image.img
        crop = crop_polygon(img, self.corner)
        mask = crop_polygon(np.ones(img.shape).astype(np.uint8) * 255, self.corner)

        return crop, mask
