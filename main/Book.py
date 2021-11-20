import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'utils')))


from typing import List, Tuple
from RowImage import RowImage
from ImageUtils import crop_polygon
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
