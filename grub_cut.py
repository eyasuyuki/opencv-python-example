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
#image = imutils.resize(image, height=500)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#edged = cv2.Canny(blurred, 50, 200, 255)
#cv2.imwrite("edged.jpg", edged)

# find contours
#cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#displayCnt = None

# loop over the contour
#for c in cnts:
#    # approximate the contour
#    peri = cv2.arcLength(c, True)
#    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#
#    # if the contour has four vertices, when we have found the lcd
#    print(len(approx))
#    if len(approx) == 4:
#        displayCnt = approx
#        break;

# extract the digit    
#wrapped = four_point_transform(gray, displayCnt.reshape(4, 2))
#output = four_point_transform(image, displayCnt.reshape(4, 2))
#wrapped = gray
#cv2.imwrite("gray.jpg", gray)
#output = image

# threshold the wrapped image
#thresh = cv2.threshold(wrapped, 0, 255,
#                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#cv2.imwrite("thresh.jpg", thresh)

# http://ni4muraano.hatenablog.com/entry/2017/05/20/215135
#element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (200, 200))
#cv2.imwrite("element.jpg", element)

#background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)
#cv2.imwrite("background.jpg", background)

#foreground = cv2.absdiff(gray, background)
#cv2.imwrite("foreground.jpg", foreground)

# http://ina17.hatenablog.jp/entry/2017/11/28/093146
mask = np.zeros(image.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64) # background
fgdModel = np.zeros((1, 65), np.float64)

rect = (22, 13, 200, 230) # x, y, w, h

cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = image * mask2[:, :, np.newaxis]

cv2.imwrite("img.jpg", img) # debug

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite("blurred.jpg", blurred)

# threshold the wrapped image
thresh = cv2.threshold(blurred, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imwrite("thresh.jpg", thresh)
