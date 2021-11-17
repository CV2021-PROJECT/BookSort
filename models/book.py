from typing import List, Tuple
from image import RowImage
import numpy as np


class Book:
    def __init__(
        self,
        src_image: RowImage,
        image_coordinates: np.ndarray,
    ) -> None:
        """
        Args:
            src_image (RowImage): 책의 출처 이미지를 나타낸다.
            image_coordinates (np.ndarray): 책의 꼭지점 4개의 좌표를 (4, 2) 행렬로 나타낸다.
        """
        self.src_image = src_image
        self.image_coordinates = image_coordinates
