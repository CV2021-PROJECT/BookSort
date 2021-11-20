import numpy as np
import cv2



def crop_polygon(image, vertices):
    h = image.shape[0]
    w = image.shape[1]

    mask = np.zeros((h, w)).astype(np.uint8)
    vertices = np.array([vertices])
    rect = cv2.boundingRect(vertices)
    
    cv2.fillPoly(mask, vertices, 255)
    cropped = cv2.bitwise_and(image, image, mask = mask)
    cropped = cropped[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    return cropped

def normalize(image):
    avg = np.average(image)
    std = np.std(image)

    return (image - avg) / std
