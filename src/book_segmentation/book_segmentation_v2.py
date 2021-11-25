#%%
import sys

sys.path.append("..")

from matplotlib import pyplot as plt
import cv2
from typing import List, Tuple
import numpy as np
import scipy.ndimage
from scipy.ndimage.measurements import label
import scipy.stats
from helpers import read_image, resize_img, show_image
from models import RowImage


class Line(object):
    """
    Simple class that holds the information related to a line;
    i.e., the slope, y-intercept, and center point along the line
    """

    vertical_threshold = 30

    def __init__(self, m, b, center, min_x, max_x, min_y, max_y):
        """
        m: slope
        b: y-intercept
        center: center point along the line (tuple)
        """

        self.m = m
        self.b = b

        self.center = center

        self.min_x = min_x
        self.max_x = max_x

        self.min_y = min_y
        self.max_y = max_y

    def y(self, x):
        """
        Returns the y-value of the line at position x.
        If the line is vertical (i.e., slope is close to infinity), the y-value
        will be returned as None
        """

        # Line is vertical
        if self.m > self.vertical_threshold:
            return None

        else:
            return self.m * x + self.b

    def x(self, y):
        """
        Returns the x-value of the line at posiion y.
        If the line is vertical (i.e., slope is close to infinity), will always
        return the center point of the line
        """

        # Line is vertical
        if self.m > self.vertical_threshold:
            return self.center[0]

        # Line is not vertical
        else:
            return (y - self.b) / self.m


class BookSpines:
    def __init__(self, row_images: List[RowImage], verbose=False) -> None:
        self.row_images = row_images
        self.verbose = verbose

    @staticmethod
    def plot_image(img: np.ndarray):
        plt.figure(figsize=(16, 12))
        plt.imshow(img, cmap="gray", interpolation="none")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def show_if_verbose(self, img: np.ndarray, label: str):
        if self.verbose:
            print(f"-------------{label}-------------")
            self.plot_image(img)

    def gaussian_blur(self, img: np.ndarray, sigma=3) -> np.ndarray:
        blur = scipy.ndimage.filters.gaussian_filter(img, sigma=(sigma, sigma))
        self.show_if_verbose(blur, "gaussian_blur")
        return blur

    def downsample(self, img: np.ndarray, iterations: int = 2) -> np.ndarray:
        downsampled = img.copy()
        for _ in range(iterations):
            downsampled = scipy.ndimage.interpolation.zoom(downsampled, 0.5)
        self.show_if_verbose(downsampled, "downsample")
        return downsampled

    def upsample(self, img: np.ndarray, upsample_factor: int = 4):
        upsampled = img.repeat(upsample_factor, axis=0).repeat(upsample_factor, axis=1)
        self.show_if_verbose(upsampled, "upsample")
        return upsampled

    def sobel_vertical(self, img: np.ndarray) -> np.ndarray:
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0) ** 2
        self.show_if_verbose(sobel, "sobel_vertical")
        return sobel

    def normalize(self, img: np.ndarray) -> np.ndarray:
        imin, imax = np.min(img), np.max(img)
        normalized = (img - imin) / (imax - imin)
        self.show_if_verbose(normalized, "normalize")
        return normalized

    def adaptive_binarize(self, img: np.ndarray) -> np.ndarray:
        cutoff = 0
        img = img.copy()
        for i in range(20):
            cutoff = i * 0.01
            bright_pixel_ratio = len(np.where(img > cutoff)[0]) / (
                img.shape[0] * img.shape[1]
            )
            if bright_pixel_ratio <= 0.4:
                break

        img[img > cutoff] = 1
        img[img <= cutoff] = 0
        self.show_if_verbose(img, "adaptive_binarize")
        return img

    def vertical_erode(
        self, img: np.ndarray, structure_length: int = 200, iterations: int = 8
    ) -> np.ndarray:
        structure = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) * structure_length
        eroded = scipy.ndimage.morphology.binary_erosion(img, structure, iterations)
        self.show_if_verbose(eroded, "vertical_erode")
        return eroded

    def vertical_dilate(
        self, img: np.ndarray, structure_length: int = 500, iterations: int = 10
    ) -> np.ndarray:
        structure = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]) * structure_length
        dilated = scipy.ndimage.morphology.binary_dilation(img, structure, iterations)
        self.show_if_verbose(dilated, "vertical_dilate")
        return dilated

    def group_lines(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        grouped_img, n_features = scipy.ndimage.label(img, structure=np.ones((3, 3)))
        self.show_if_verbose(grouped_img, "group_lines")
        return grouped_img, n_features

    def remove_short_lines_vertical(self, img: np.ndarray, n_features: int):
        drop_values = []
        ptps = []

        # Calculate peak-to-peak height of line
        labels = list(range(1, n_features + 1))
        for label in labels:
            bright_pixels = np.where(img == label)
            ptp = np.ptp(bright_pixels[0])
            ptps.append(ptp)

        # Determine which lines to drop
        threshold = np.max(ptps) / 2.0
        for i in range(len(ptps)):
            if ptps[i] < threshold:
                drop_values.append(labels[i])

        # Drop the lines
        for drop_value in drop_values:
            img[img == drop_value] = 0
        self.show_if_verbose(img, "remove_short_lines_vertical")
        return img

    def binarize_to_one(self, img: np.ndarray) -> np.ndarray:
        img[img > 0] = 1
        self.show_if_verbose(img, "binarize_to_one")
        return img

    def get_lines_from_img(self, img: np.ndarray, n_features: int) -> List[Line]:
        lines = []
        for level in range(1, n_features + 1):
            line = np.where(img == level)
            xs = line[1]
            ys = line[0]
            center = [np.mean(xs), np.mean(ys)]

            min_x = np.min(xs)
            max_x = np.max(xs)
            min_y = np.min(ys)
            max_y = np.max(ys)

            spread = (np.max(ys) - np.min(ys)) / (np.max(xs) - np.min(xs))

            # Line is vertical
            if spread > 10:
                line = Line(1000, 0, center, min_x, max_x, min_y, max_y)

            # Line is not vertical
            else:
                m, b, r, p, std = scipy.stats.linregress(xs, ys)
                line = Line(m, b, center, min_x, max_x, min_y, max_y)

            lines.append(line)

        # Sort the lines by their center x positions
        lines.sort(key=lambda line: line.center[0])

        return lines

    def get_book_spines(self, img: np.ndarray):
        _img = np.mean(img, axis=2)
        img_final = np.zeros_like(_img)
        _img = self.gaussian_blur(_img)
        _img = self.sobel_vertical(_img)
        _img = self.downsample(_img)
        _img = self.normalize(_img)
        _img = self.adaptive_binarize(_img)
        _img = self.vertical_erode(_img)
        _img = self.vertical_dilate(_img)
        _img, n_features = self.group_lines(_img)
        _img = self.remove_short_lines_vertical(_img, n_features=n_features)
        _img = self.upsample(_img)

        _img.resize(img.shape[:2])
        img_final += _img

        img_final = self.binarize_to_one(img_final)
        img_final, n_features = self.group_lines(img_final)
        lines = self.get_lines_from_img(img_final, n_features=n_features)

        if self.verbose:
            new_img = img.copy()
            plt.imshow(new_img, cmap="gray", interpolation="none")
            for line in lines:
                y0 = line.min_y
                y1 = line.max_y
                x0 = line.x(y0)
                x1 = line.x(y1)
                plt.plot([x0, x1], [y0, y1], color=np.array([0, 169, 55]) / 255.0, lw=6)
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0)
            plt.xticks([])
            plt.yticks([])
            plt.show()

    def main(self):
        for row_image in self.row_images:
            self.get_book_spines(img=row_image.img)


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

    book_spines = BookSpines(row_images=row_images, verbose=True)
    book_spines.main()
