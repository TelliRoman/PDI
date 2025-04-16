import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\cameraman.tif", cv.IMREAD_GRAYSCALE)
box=cv.boxFilter(img, -1, (5,5), normalize=True)
kernel_cruz=np.array([[0,1,0],[1,1,1],[0,1,0]])
kernel_cruz=kernel_cruz/9
cruz=cv.filter2D(img, -1, kernel_cruz)
mask_promedio=np.ones((3,3), np.float32)/9
mask_promedo=cv.filter2D(img, -1, mask_promedio)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_title("Imagen Original")
ax[0,1].imshow(box, cmap='gray')
ax[0,1].set_title("Filtro Box")
ax[1,0].imshow(cruz, cmap='gray')
ax[1,0].set_title("Filtro Cruz")
ax[1,1].imshow(mask_promedo, cmap='gray')
ax[1,1].set_title("Filtro Promedio")
plt.show()

