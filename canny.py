from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
from matplotlib import pyplot as plt
import numpy as np

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
        cv2.imwrite("test.jpg", img)
        grayed = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite("grayed.jpg", grayed)
        return grayed

def blurr(img):
        blurred = cv2.GaussianBlur(img, (9, 9), 10, 10)
        cv2.imwrite("blurred.jpg", blurred)
        return blurred

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

# load image
image = cv2.imread("example.jpg")

# resize

# image = imutils.resize(image, height=500)

# gray
grayed = gray(image)

# blur

blurred = blurr(grayed)

# lut

contirast = lut(blurred, 60, 195)

# adaptive threshold

ret,th1 = cv2.threshold(contrast,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(contrast,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(contrast,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(contrast,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite("th1.jpg", th1)
cv2.imwrite("th2.jpg", th2)
cv2.imwrite("th3.jpg", th3)
cv2.imwrite("th4.jpg", th4)

# canny

edged = cv2.Canny(contrast, 20, 80, 255)
cv2.imwrite("edged.jpg", edged)

# find contors

cnts0 = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

# filter noize

thCnts = []
for c in cnts0:
        (x, y, z, w) = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        print(f"{x}, {y}, {z}, {w}, {a}")
        if (a >= 5):
                thCnts.append(c)

background = np.zeros_like(image, np.uint8)

edged_cnts = cv2.drawContours(background, thCnts, -1, (255, 255, 255), 1)
edged_cnts = cv2.cvtColor(edged_cnts, cv2.COLOR_BGR2GRAY);
cv2.imwrite("edged_cnts.jpg", edged_cnts)

# https://github.com/DevashishPrasad/LCD-OCR/blob/master/code.py

dilate = cv2.dilate(edged, None, iterations=16)
cv2.imwrite("dilate.jpg", dilate)

erode = cv2.erode(dilate, None, iterations=16)
cv2.imwrite("erode.jpg", erode)

# another dilate/erode

d2 = cv2.dilate(edged, None, iterations=16)
cv2.imwrite("d2.jpg", d2)

e2 = cv2.erode(d2, None, iterations=16)
cv2.imwrite("e2.jpg", e2)

# find contuors

mask2 = np.ones(image.shape[:2], dtype="uint8") * 255

cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured = cv2.drawContours(image.copy(), cnts, -1, (0, 255, 0), 3)
cv2.imwrite("contoured.jpg", contoured)

orig = image.copy()
for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        #print(f"{x}, {y}, {w}, {h}")
        if cv2.contourArea(c) < 900:
                cv2.drawContours(mask2, [c], -1, 0, -1)
                continue

newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
cv2.imwrite("newimage1.jpg", newimage)

newimage = cv2.dilate(newimage,None, iterations=16)
newimage = cv2.erode(newimage,None, iterations=16)
cv2.imwrite("newimage2.jpg", newimage)

""" ret,newimage = cv2.threshold(newimage,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imwrite("newimage3.jpg", newimage)
 """
cnts2 = cv2.findContours(newimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
digitCnts = []

""" for c in cnts2:
    (x, y, w, h) = cv2.boundingRect(c)
    print(f"{x}, {y}, {w}, {h}")
    if (w >= 20 and w <= 200) and h >= 70:
        digitCnts.append(c)
 """
#digits = cv2.drawContours(image.copy(), digitCnts, -1, (0, 255, 0), 3)
digits = cv2.drawContours(image.copy(), cnts2, -1, (0, 255, 0), 3)
cv2.imwrite("digits.jpg", digits)
