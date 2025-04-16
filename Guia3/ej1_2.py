import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\cameraman.tif", cv.IMREAD_GRAYSCALE)
gauss3=cv.GaussianBlur(img, (7,7), 1)
gauss5=cv.GaussianBlur(img, (5,5), 1)
gauss3sig=cv.GaussianBlur(img, (7,7), 3)
gauss5sig=cv.GaussianBlur(img, (5,5), 3)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

ax[0,0].imshow(gauss3, cmap='gray')
ax[0,0].set_title("Gauss 3x3 sigma=1")
ax[0,1].imshow(gauss5, cmap='gray')
ax[0,1].set_title("Gauss 5x5 sigma=1")
ax[1,0].imshow(gauss3sig, cmap='gray')
ax[1,0].set_title("Gauss 3x3 sigma=3")
ax[1,1].imshow(gauss5sig, cmap='gray')
ax[1,1].set_title("Gauss 5x5 sigma=3")
plt.show()

