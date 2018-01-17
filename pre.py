import cv2
import numpy as np

def preprocess(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    #cv2.imshow('1', dilated)
    diff1 = 255 - cv2.subtract(dilated, gray)
    #cv2.imshow('2', diff1)

    median = cv2.medianBlur(dilated, 15)
    #cv2.imshow('3', median)
    diff2 = 255 - cv2.subtract(median, gray)
    #cv2.imshow('4', diff2)

    normed = cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow('5', normed)

    #res = np.hstack((gray, dilated, diff1, median, diff2, normed))
    cv2.waitKey(1)
    return normed
