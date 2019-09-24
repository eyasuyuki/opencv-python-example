import datetime

from imutils import contours
import cv2
import numpy as np
from datetime import datetime

DIGITS_LOOKUP = DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def gray(img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("grayed.jpg", grayed)
        return grayed

def denoise(img):
        denoised = cv2.fastNlMeansDenoising(img, 10, 10, 7, 21)
        cv2.imwrite("denoised.jpg", denoised)
        return denoised

def lut(img, min_table, max_table):
        diff_table = max_table - min_table
        lookup_table = np.arange(256, dtype="uint8")

        for i in range(0, len(lookup_table)):
                if (i < min_table):
                        lookup_table[i] = 0
                elif (i >= min_table and i <= max_table):
                        n = 255 * (i - min_table) / diff_table
                        #print(f"{i}, {n}")
                        lookup_table[i] = n
                elif (i > max_table):
                        lookup_table[i] = 255

        contrast = cv2.LUT(img.copy(), lookup_table)
        cv2.imwrite("contrast.jpg", contrast)
        return contrast

def canny(img):
        edged = cv2.Canny(img, 20, 80, 255)
        cv2.imwrite("edged.jpg", edged)
        return edged

def filter_noise(img):
        # find contors

        cnts0 = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        # filter noize

        thCnts = []
        for c in cnts0:
                (x, y, z, w) = cv2.boundingRect(c)
                a = cv2.contourArea(c)
                # print(f"{x}, {y}, {z}, {w}, {a}")
                if (a >= 10):
                        thCnts.append(c)

        background = np.zeros_like(img, np.uint8)

        edged_cnts = cv2.drawContours(background, thCnts, -1, (255, 255, 255), 1)
        cv2.imwrite("edged_cnts.jpg", edged_cnts)
        return edged_cnts

def closing(img):
        # https://github.com/DevashishPrasad/LCD-OCR/blob/master/code.py

        dilate = cv2.dilate(edged, None, iterations=4)
        cv2.imwrite("dilate.jpg", dilate)

        erode = cv2.erode(dilate, None, iterations=4)
        cv2.imwrite("erode.jpg", erode)

        return dilate, erode

def adaptive_threshold(img):
        ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        now = datetime.utcnow()
        cv2.imwrite(now.strftime('th.jpg'), th4)
        # TODO recognize


# load image
image = cv2.imread("example.jpg")

# resize

# image = imutils.resize(image, height=500)

# gray
grayed = gray(image)

# denoizing

denoized = denoise(grayed)

# lut

contrast = lut(denoized, 60, 195)

# canny

edged = canny(contrast)

# filter noize

edged_cnts = filter_noise(edged)

# open

dilate, erode = closing(edged_cnts)

# find contuors

# mask2 = np.ones(image.shape[:2], dtype="uint8") * 255

cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# sort contours

sorted_cnts = contours.sort_contours(cnts, method="left-to-right")[0]

orig = contrast.copy()
for c in sorted_cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = orig[y:y + h, x:x + w]
        adaptive_threshold(roi)
        print(f"{x}, {y}, {w}, {h}")
        orig = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)


