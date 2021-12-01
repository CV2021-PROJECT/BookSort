#%%
import sys

sys.path.append("..")
# from .book_segmentation import convert_to_xy, draw_lines_on_image
from book_segmentation import convert_to_xy, draw_lines_on_image, trim_lines

from matplotlib import pyplot as plt
import cv2
from typing import List, Tuple
import numpy as np
import scipy.ndimage
from scipy.ndimage.measurements import label
import scipy.stats
from helpers import read_image, resize_img, show_image
from models import RowImage, Book, Source


class Line:
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
        self.books = []
        self.hough_lines = []

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

    def gaussian_blur(self, img: np.ndarray, sigma=1) -> np.ndarray:
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
        sobel = np.sqrt(cv2.Sobel(img, cv2.CV_64F, 1, 0)**2)
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

        threshold = np.max(ptps) / 2.0
        for i in range(len(ptps)):
            if ptps[i] < threshold:
                drop_values.append(labels[i])

        # Drop the lines
        for drop_value in drop_values:
            img[img == drop_value] = 0
        self.show_if_verbose(img, "remove_short_lines_vertical")
        return img, n_features

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

            # # Line is vertical
            # if spread > 10:
            #     line = Line(1000, 0, center, min_x, max_x, min_y, max_y)

            # # Line is not vertical
            # else:
            m, b, r, p, std = scipy.stats.linregress(xs, ys)
            line = Line(m, b, center, min_x, max_x, min_y, max_y)
            # if line.max_y < 1500:
            #     continue

            # if abs(line.m) <= 5:
            #     continue
            lines.append(line)

        # Sort the lines by their center x positions
        lines.sort(key=lambda line: line.center[0])
        # for line in lines:
        #     if line.m == 1000:
        #         print("vertical")
        #     else:
        #         print("not vertical")

        return lines

    def get_hough_line_per_group(self, img: np.ndarray, n_features: int):
        # lines = []
        lrho = []
        ltheta = []
        for level in range(1, n_features + 1):
            line = np.where(img == level)
            ys = line[0]
            max_y = np.max(ys)
            if max_y < img.shape[0] * 7 / 10:
                # 전체 높이의 10분의 6지점
                continue

            temp = img.copy()
            temp[level != img] = 0
            rho_thetas = cv2.HoughLines(temp.astype(np.uint8), 1, np.pi / 180, 0)
            self.hough_lines = rho_thetas
            if rho_thetas is None:
                continue
            # line = convert_to_xy(rho_thetas[:1], max_length=4000)
            # lines.extend(line)
            lrho.append(rho_thetas[0][0][0])
            ltheta.append(rho_thetas[0][0][1])
        return lrho, ltheta

    def GetStartEndPoints(self, rho, theta):
        length = 4000
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int(x0 - length * b)
        y1 = int(y0 + length * a)
        x2 = int(x0 + length * b)
        y2 = int(y0 - length * a)
        return np.array([(x1, y1), (x2, y2)])

    def GetAllPointsBetween(self, start: np.ndarray, end: np.ndarray):
        x_diff = abs(end[0] - start[0])
        y_diff = abs(end[1] - start[1])

        def _arange(p1, p2):
            if p2 <= p1:
                return np.arange(p2, p1)
            return np.arange(p1, p2)

        # special case
        if x_diff == 0:
            y = _arange(start[1], end[1])
            x = np.full(len(y), start[0])
            print('special case', x, y)
            return zip(x.astype(int), y.astype(int))

        m = (end[1] - start[1]) / (end[0] - start[0])
        c = end[1] - start[1] * m
        if x_diff >= y_diff:
            x = _arange(start[0], end[0])
            y = m * (x - start[0]) + start[1]
        else:

            x = _arange(start[0], end[0])
            y = m * x + 1
            # x = (y - start[1]) / m + start[0]
        return zip(x.astype(int), y.astype(int))  # [(x0,y0), (x1,y1), ...]

    def HoughLineSegments(self, lrho, ltheta, binary_img: np.ndarray):
        ret_list = []
        self.segments = []
        for i, (rho, theta) in enumerate(zip(lrho, ltheta)):
            start_coord, end_coord = self.GetStartEndPoints(rho, theta)
            self.segments.append(dict(start=start_coord, end=end_coord))
            temp_list = []
            seg_dict = dict()
            for (x, y) in self.GetAllPointsBetween(start_coord, end_coord):
                # if not (
                #     binary_img.shape[1] >= x >= 1 and binary_img.shape[0] >= y >= 1
                # ):
                #     continue
                try:
                    if np.max(binary_img[y - 4 : y + 5, x - 4 : x + 5]) == 1:
                        this_coord = np.array([x, y])
                        if "start" not in seg_dict:
                            seg_dict["start"] = this_coord
                        seg_dict["end"] = this_coord
                    else:
                        if "start" in seg_dict and "end" in seg_dict:
                            temp_list.append(seg_dict)
                            seg_dict = dict()
                except (IndexError, ValueError) as e:
                    if "start" in seg_dict and "end" in seg_dict:
                        temp_list.append(seg_dict)
                        seg_dict = dict()
                    continue
            if len(temp_list) == 0:
                continue
            ret_list.append(temp_list[0])
        return ret_list

    def FindLongestLine(self, line_list: list):
        max_len = -1
        max_dict = None
        for line_dict in line_list:
            new_len = np.hypot(*(line_dict["start"] - line_dict["end"]))
            if new_len > max_len:
                max_len = new_len
                max_dict = line_dict
        return max_dict

    def get_book_spines(self, row_image: RowImage):
        img = row_image.img
        _img = np.mean(img, axis=2)
        self.verbose = False
        _img = self.gaussian_blur(_img)
        _img = self.sobel_vertical(_img)
        _img = self.downsample(_img, iterations=1)
        _img = self.normalize(_img)
        _img = self.adaptive_binarize(_img)
        _img = self.vertical_erode(_img)
        _img = self.vertical_dilate(_img)
        self.verbose = True
        _img, n_features = self.group_lines(_img)
        _img, n_features = self.remove_short_lines_vertical(_img, n_features=n_features)
        _img = resize_img(_img, target_height=img.shape[0], target_width=img.shape[1])
        binary_img = self.binarize_to_one(_img)
        _img, n_features = self.group_lines(binary_img)

        lrho, ltheta = self.get_hough_line_per_group(_img, n_features=n_features)
        segments = self.HoughLineSegments(lrho, ltheta, binary_img)
        segments.sort(key=lambda x: x["start"][0])

        for i in range(len(segments) - 1):
            seg1, seg2 = segments[i], segments[i + 1]
            lu, ld, rd, ru = seg1["start"], seg1["end"], seg2["end"], seg2["start"]
            corner = np.array([lu, ld, rd, ru]).astype(int)
            new_book = Book(row_image=row_image, corner=corner)
            self.books.append(new_book)

        # if self.verbose:
        new_img = img.copy()
        plt.figure(figsize=(16, 12))
        plt.imshow(new_img, cmap="gray", interpolation="none")
        for segment in segments:
            ((x0, y0), (x1, y1)) = (segment["start"], segment["end"])
            plt.plot([x0, x1], [y0, y1], color=np.array([0, 169, 55]) / 255.0, lw=5)
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def get_books(self) -> List[Book]:
        for row_image in self.row_images:
            self.get_book_spines(row_image=row_image)
        return self.books


#%%
if __name__ == "__main__":
    # Prepare RowImages
    paths = [
        # "./sample_inputs/sample1.jpeg",
        # "./sample_inputs/sample2.jpeg",
        # "./sample_inputs/sample3.jpeg",
        # "./sample_inputs/sample4.jpeg",
        # "./sample_inputs/sample5.jpg",
        # "./sample_inputs/sample6.jpg",
        # "./sample_inputs/sample7.jpg",
        # "./sample_inputs/row1.png",
        # "./sample_inputs/row2.png",
        "./sample_inputs/row3.png",
    ]
    row_images = []
    for i, path in enumerate(paths, 1):
        np_image = read_image(path)
        row_images.append(
            RowImage(np_image, source=Source(np_image, path), relative_floor=1)
        )

    book_spines = BookSpines(row_images=row_images, verbose=True)
    books = book_spines.get_books()
    # for book in books:
    #     rect, _ = book.rect()
    #     show_image(rect)
