import numpy as np


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
