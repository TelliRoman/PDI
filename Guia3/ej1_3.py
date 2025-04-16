import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\cameraman.tif", cv.IMREAD_GRAYSCALE)
media=cv.medianBlur(img, 3)
media5=cv.medianBlur(img, 5)
media7=cv.medianBlur(img, 7)
media9=cv.medianBlur(img, 9)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

ax[0,0].imshow(media, cmap='gray')
ax[0,0].set_title("Media 3")
ax[0,1].imshow(media5, cmap='gray')
ax[0,1].set_title("Media 5")
ax[1,0].imshow(media7, cmap='gray')
ax[1,0].set_title("Media 7")
ax[1,1].imshow(media9, cmap='gray')
ax[1,1].set_title("Media 9")
plt.show()

# Aplica desenfoque gaussiano con kernel 5x5 y sigmaX = 1
# sigmaX controla la "anchura" de la campana gaussiana en el eje X (horizontal)
# A mayor sigmaX, mayor desenfoque (más se mezclan los píxeles vecinos)