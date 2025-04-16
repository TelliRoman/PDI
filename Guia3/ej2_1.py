import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\cameraman.tif", cv.IMREAD_GRAYSCALE)
mask=np.array([[0,-1,0],
               [-1,5,-1],
               [0,-1,0]])

mask1=np.array([[0,-1,0],
               [-1,8,-1],
               [0,-1,0]])

mask2=np.array([[-1,-1,-1],
               [-1,7,-1],
               [-1,-1,-1]])

mask4=np.array([[-1,-1,-1],
               [-1,8,-1],
               [-1,-1,-1]])

pasa_alto1=cv.filter2D(img, -1, mask)
pasa_alto2=cv.filter2D(img, -1, mask1)
pasa_alto3=cv.filter2D(img, -1, mask2)
pasa_alto4=cv.filter2D(img, -1, mask4)
fig,ax=plt.subplots(nrows=2, ncols=2, figsize=(12,12))

ax[0,0].imshow(pasa_alto4, cmap='gray')
ax[0,0].set_title("Filtro 4")
ax[0,1].imshow(pasa_alto1, cmap='gray')
ax[0,1].set_title("Fitros 1")
ax[1,0].imshow(pasa_alto2, cmap='gray')
ax[1,0].set_title("Filtro 2")
ax[1,1].imshow(pasa_alto3, cmap='gray')
ax[1,1].set_title("Filtro 3")
plt.show()