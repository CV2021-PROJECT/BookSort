from typing import List, Tuple
from row_image import RowImage
import numpy as np


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
