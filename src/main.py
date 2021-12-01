import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from models import *
from helpers import show_image_grid, show_image, read_image, resize_img
from book_rectification import rectify, generate_row_image
from row_matching import fill_matching_info, display_vertical_matching_result
from book_segmentation import BookSpines


def get_books_from_directory(source_dir, verbose=True):
    source_list = []
    row_image_list = []
    book_list = None
    
    for file_name in os.listdir(source_dir):
        # build source
        source_path = os.path.join(source_dir, file_name)
        source = read_image(source_path)
        source = resize_img(source, 1000).astype(np.uint8)
        # TODO> 재윤님이 rectify를 고쳐주신다면,, 그건 어떤 기분일까?
        #source = rectify(source)
        
        if type(source) != type(None):
            source = Source(source, source_path)
            source_list.append(source)

    print("# of source = {}".format(len(source_list)))

    # build row images
    for source in source_list:
        row_image_list.extend(generate_row_image(source))

    print("# of row-image = {}".format(len(row_image_list)))

    # segment books from each row images
    book_spines = BookSpines(row_image_list, verbose=False)
    books = book_spines.get_books()

    print("# of book = {}".format(len(books)))

    if verbose:
        show_image_grid([book.rect()[0] for book in books], 30)

    # fill info in each row images
    fill_matching_info(row_image_list)

    if verbose:
        display_vertical_matching_result(row_image_list)
        




data_dir = os.path.join(os.path.dirname(__file__), "data")
source_before_dir = os.path.join(data_dir, "source-before")
source_after_dir = os.path.join(data_dir, "source-after")


get_books_from_directory(source_before_dir)