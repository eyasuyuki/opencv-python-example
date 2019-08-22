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

# load image
image = cv2.imread("example.jpg")

# pre-process
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 10, 10)
cv2.imwrite("blurred.jpg", blurred)

# canny

edged = cv2.Canny(blurred, 20, 60, 255)
cv2.imwrite("edged.jpg", edged)

# find contors

cnts0 = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# filter noize
thCnts = []
for c in cnts0:
        (x, y, z, w) = cv2.boundingRect(c)
        a = cv2.contourArea(c)
        print(f"{x}, {y}, {z}, {w}, {a}")
        if (a >= 20):
                thCnts.append(c)

background = np.zeros_like(image, np.uint8)

edged_cnts = cv2.drawContours(background, thCnts, -1, (255, 255, 255), 1)
edged_cnts = cv2.cvtColor(edged_cnts, cv2.COLOR_BGR2GRAY);
cv2.imwrite("edged_cnts.jpg", edged_cnts)

# https://github.com/DevashishPrasad/LCD-OCR/blob/master/code.py

dilate = cv2.dilate(edged_cnts, None, iterations=16)
cv2.imwrite("dilate.jpg", dilate)

erode = cv2.erode(dilate, None, iterations=16)
cv2.imwrite("erode.jpg", erode)

mask2 = np.ones(image.shape[:2], dtype="uint8") * 255

# find contuors

cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
contoured = cv2.drawContours(image.copy(), cnts, -1, (0, 255, 0), 3)
cv2.imwrite("contoured.jpg", contoured)

orig = image.copy()
for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        print(f"{x}, {y}, {w}, {h}")
        if cv2.contourArea(c) < 900:
                cv2.drawContours(mask2, [c], -1, 0, -1)
                continue

newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
cv2.imwrite("newimage1.jpg", newimage)

newimage = cv2.dilate(newimage,None, iterations=7)
newimage = cv2.erode(newimage,None, iterations=5)
cv2.imwrite("newimage2.jpg", newimage)

""" ret,newimage = cv2.threshold(newimage,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imwrite("newimage3.jpg", newimage)
 """
cnts2 = cv2.findContours(newimage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
digitCnts = []

for c in cnts2:
    (x, y, w, h) = cv2.boundingRect(c)
    print(f"{x}, {y}, {w}, {h}")
    if (w >= 20 and w <= 200) and h >= 70:
        digitCnts.append(c)

digits = cv2.drawContours(image.copy(), digitCnts, -1, (0, 255, 0), 3)
#digits = cv2.drawContours(image.copy(), cnts2, -1, (0, 255, 0), 3)
cv2.imwrite("digits.jpg", digits)
