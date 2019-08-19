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

# http://ni4muraano.hatenablog.com/entry/2017/05/20/215135
element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)
cv2.imwrite("background.jpg", background)

foreground = cv2.absdiff(gray, background)
cv2.imwrite("foreground.jpg", foreground)

#blurred = cv2.GaussianBlur(foreground, (5, 5), 0)
blurred = cv2.blur(foreground, (5, 5))
cv2.imwrite("blurred.jpg", blurred)

edged = cv2.Canny(blurred, 20, 60, 255)
cv2.imwrite("edged.jpg", edged)
