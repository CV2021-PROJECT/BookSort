from typing import List, Tuple
import numpy as np
from helpers import crop_polygon

class RowImage:
    """책장의 한 행을 나타내는 이미지 모델"""

    def __init__(self, img: np.ndarray, relative_floor: int) -> None:
        """
        Args:
            img (np.ndarray): numpy 배열로 나타낸 이미지 파일
            relative_floor (int): 상대적 층수
        """
        self.img = img
        self.relative_floor = relative_floor
        self.absolute_floor = None  # 생성 시점에서는 absolute_floor 정보를 알 수 없다.

    def update_absolute_floor(self, value: int):
        self.absolute_floor = value


class Book:
    def __init__(
        self,
        row_image: RowImage,
        corner: np.ndarray,
    ) -> None:
        """
        Args:
            row_image (RowImage): 책의 출처 이미지를 나타낸다.
            corner (np.ndarray): 책의 꼭지점 4개의 좌표를 (4, 2) 행렬로 나타낸다.
        """
        self.row_image = row_image
        self.corner = corner

    def rect(self):
        """
        Return:
            crop: RGB 이미지
            mask: 0/255 이미지
        """
        img = self.row_image.img
        crop = crop_polygon(img, self.corner)
        mask = crop_polygon(np.ones(img.shape).astype(np.uint8) * 255, self.corner)

        return crop, mask

