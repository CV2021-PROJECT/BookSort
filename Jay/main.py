import numpy as np
import cv2
from utils import *

class Book:
    def __init__(self, image, corner=None, mask=None):
        self.image = image
        self.corner = corner
        self.mask = mask

    def rect(self):
        if type(self.corner) == type(None) and type(self.mask) == type(None):
            return None, None
        elif type(self.corner) != type(None):            
            rect = crop_polygon(self.image, self.corner)
            mask = crop_polygon(np.ones(self.image.shape).astype(np.uint8) * 255, self.corner)
            return rect, mask
        elif type(self.mask) != type(None):
            t = np.where(np.any(self.mask>128, axis=1))[0][0]
            b = np.where(np.any(self.mask>128, axis=1))[0][-1]
            l = np.where(np.any(self.mask>128, axis=0))[0][0]
            r = np.where(np.any(self.mask>128, axis=0))[0][-1]

            print(t,b,l,r)
            
            rect = self.image[t:b+1, l:r+1]
            return rect, self.mask
        

def warp_book(book, book_ref, H):
    image_in, mask_in = book.rect()
    image_ref, _ = book_ref.rect()
    
    image_warped = cv2.warpPerspective(image_in, H, dsize=(image_ref.shape[1], image_ref.shape[0]))
    mask_warped = cv2.warpPerspective(mask_in, H, dsize=(image_ref.shape[1], image_ref.shape[0]))

    cv2.imshow("image_warped", image_warped)
    cv2.imshow("mask_warped", mask_warped)

    return Book(image_warped, mask=mask_warped)


def match_book(book, book_ref, thr_match_nms=0.7, thr_inlier_pixel=0.1, verbose=False):
    img, _ = book.rect()
    img_ref, _ = book_ref.rect()
    
    kp, kp_ref = get_corr_keypoints(img, img_ref, thr=thr_match_nms, verbose=verbose)

    if len(kp) < 4: return False
    
    H = find_optimal_H(kp_ref, kp, thr=thr_inlier_pixel)

    warped_book = warp_book(book, book_ref, H)

    img = warped_book.rect()[0]

    cv2.imshow("img", img)

    


def main():
    image1 = cv2.imread("./data/image1_512.jpg")
    image2 = cv2.imread("./data/image2_512.jpg")
    image3 = cv2.imread("./data/image3_512.jpg")

    book1 = Book(image1, corner=np.array([[210, 46], [272, 48], [254, 494], [202, 488]]))
    book2 = Book(image2, corner=np.array([[214, 64], [272, 64], [274, 500], [204, 494]]))
    book3 = Book(image3, corner=np.array([[190, 148], [268, 156], [264, 468], [216, 472]]))
    book4 = Book(image1, corner=np.array([[93, 34], [178, 35], [170, 488], [102, 482]]))
    book5 = Book(image2, corner=np.array([[127, 67], [194, 76], [169, 507], [86, 497]]))
    book6 = Book(image3, corner=np.array([[58, 134], [154, 136], [194, 462], [138, 462]]))

    match_book(book2, book1, 0.5, 0.25, verbose=True)
    cv2.waitKey(0)
    match_book(book6, book4, 0.5, 0.25, verbose=True)
    cv2.waitKey(0)
    match_book(book4, book1, 0.5, 0.25)
    cv2.waitKey(0)
    match_book(book5, book1, 0.5, 0.25)
    cv2.waitKey(0)
    match_book(book6, book1, 0.5, 0.25)
    cv2.waitKey(0)
    assert False

main()
