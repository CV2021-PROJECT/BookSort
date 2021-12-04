#%%
import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import tqdm

from models import *
from helpers import show_image_grid, show_image, read_image, resize_img, normalize
from book_rectification import rectify, generate_row_image
from row_matching import fill_matching_info, display_vertical_matching_result, display_horizontal_matching_result
from book_segmentation import BookSpines
from book_matching import fill_position_info, save_plt_grid_books, match_book_by_image, convert_to_book_group_list


def get_books_from_directory(source_dir, log_prefix="", verbose=False):
    source_list = []
    row_image_list = []
    book_list = None

    for file_name in os.listdir(source_dir):
        # build source
        source_path = os.path.join(source_dir, file_name)
        source = read_image(source_path)
        source = rectify(source)
        if type(source) != type(None):
            source = Source(source, source_path)
            source_list.append(source)

    print("# of source = {}".format(len(source_list)))

    # build row images
    for source in source_list:
        row_image_list.extend(generate_row_image(source))

    print("# of row-image = {}".format(len(row_image_list)))

    # fill info in each row images
    fill_matching_info(row_image_list)

    if verbose:
        display_vertical_matching_result(row_image_list)
        display_horizontal_matching_result(row_image_list)

    row_image_list = list(filter(
        lambda row_image: type(row_image.homography_in_row) != type(None),
        row_image_list
    ))

    print("# of matched row-image = {}".format(len(row_image_list)))

    # segment books from each row images
    book_spines = BookSpines(row_image_list, verbose=False)
    book_list = book_spines.get_books()

    if verbose:
        show_image_grid(
            [book.rect()[0] for book in book_list],
            int(math.sqrt(len(book_list) * 10))
        )
        
    print("# of book spine = {}".format(len(book_list)))

    # positioning books
    fill_position_info(book_list)

    # show grid image
    book_group_list = convert_to_book_group_list(book_list)
    save_plt_grid_books(book_group_list, filename=f"{log_prefix}_book_grid")

    print("# of book = {}".format(len(book_group_list)))

    return book_group_list


def query_book_group(bg_list, bg_in, fast_match, params_dict):
    for bg in bg_list:
        if fast_match:
            book1 = bg.get_first()
            book2 = bg_in.get_first()
            match_result = match_book_by_image(book1, book2, **params_dict)
            if match_result:
                return bg
        else:
            for book1 in bg.book_list:
                for book2 in bg_in.book_list:
                    match_result = match_book_by_image(book1, book2, **params_dict)
                    if match_result:
                        return bg

    return None


def compare_book_order(bg_list_before, bg_list_after):
    # call by value
    bg_list_before = bg_list_before.copy()
    bg_list_after = bg_list_after.copy()

    # what to find
    trajectory_list = []

    # params list
    params_dict_list = [
        {
            "thr_key_point_num": 10,
            "thr_inlier_pixel": 8,
            "thr_iou": 0.5,
            "thr_rms_diff": 1.0,
            "verbose": False
        },
        {
            "thr_key_point_num": 5,
            "thr_inlier_pixel": 10,
            "thr_iou": 0.4,
            "thr_rms_diff": 1.35,
            "verbose": False
        }
    ]

    for params_dict in params_dict_list:
        _bg_list_before = bg_list_before.copy()
        for bg_before in _bg_list_before:
            bg_after = query_book_group(
                bg_list_after,
                bg_before,
                params_dict != params_dict_list[-1],
                params_dict=params_dict
            )
            print("query on {} -> {}".format(bg_before.position, bg_after))
            if type(bg_after) != type(None):
                trajectory_list.append(BookTrajectory(bg_before, bg_after))
                bg_list_before.remove(bg_before)
                bg_list_after.remove(bg_after)

    for bg_before_remained in bg_list_before:
        trajectory_list.append(BookTrajectory(bg_before_remained, None))

    for bg_after_remained in bg_list_after:
        trajectory_list.append(BookTrajectory(None, bg_after_remained))

    return trajectory_list
    


data_dir = os.path.join(os.path.dirname(__file__), "data")
source_before_dir = os.path.join(data_dir, "source-before")
source_after_dir = os.path.join(data_dir, "source-after")
# source_before_dir = os.path.join(data_dir, "서가5_시점t_세로만")
# source_after_dir = os.path.join(data_dir, "서가5_시점t+1_세로만")
# source_before_dir = os.path.join(data_dir, "서가6_시점t_가로세로둘다")
# source_after_dir = os.path.join(data_dir, "서가6_시점t+1_가로세로둘다")

bg_list_before = get_books_from_directory(source_before_dir, log_prefix="before", verbose=False)
bg_list_after = get_books_from_directory(source_after_dir, log_prefix="after", verbose=False)

trajectory_list = compare_book_order(bg_list_before, bg_list_after)
for t in trajectory_list:
    print(t)
