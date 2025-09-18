import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\hubble.tif',cv.IMREAD_GRAYSCALE)
img_suavizada = cv.boxFilter(img,-1,(7,7),normalize=True)
img_umblar = np.where(img_suavizada > 130 , 255, 0).astype(np.uint8)
plt.figure(0)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(img_suavizada, cmap='gray')
plt.title('Suavizada con Box 7x7')
plt.subplot(1, 3, 3)
plt.imshow(img_umblar, cmap='gray')
plt.title('Umbralizada')
plt.show()