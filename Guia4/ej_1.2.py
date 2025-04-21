import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\rosas.jpg')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)

h = (h + 70) % 180  # Shift hue by 70 degrees
s=-s  # Invert saturation
v = -v  # Invert value
img_hsv = cv.merge((h, s, v))  # Merge the channels back
img_result = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

cv.imshow('Modified Image', img_result)
cv.waitKey(0)
cv.destroyAllWindows()