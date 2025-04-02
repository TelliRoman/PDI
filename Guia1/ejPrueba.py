import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

print("Python version %s / OpenCV version %s " %(sys.version,cv.__version__))

img_gray = cv.imread('Guia1/camino.tif', cv.IMREAD_GRAYSCALE)
img_color = cv.imread('Guia1/camino.tif') # color

# cuatro imagenes en una figura
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

# Por defecto, imshow toma imágenes HXCX3 como RGB, mientras que opencv usa BGR
ax[0,0].imshow(img_color)
ax[0,0].set_title("BGR")
ax[0,1].imshow(img_color[:,:,::-1]) # invierte los canales, ::-1 indica que se
# recorre desde el ultimo elemento al primero
ax[0,1].set_title("RGB")

# Si es de un solo canal, y no se especifica nada mas, se dibuja una imagen con un mapa de color por defecto, asignando a cada nivel de gris un color RGB
ax[1,0].imshow(img_gray)
ax[1,0].set_title("Color map por default")
# para representar correctamente una imagen de grises, usar el mapa da color
# "gray" y explicitar el valor minimo y máximo de intensidad.
ax[1,1].imshow(img_gray,cmap="gray",vmin=0,vmax=255)
ax[1,1].set_title("Grises")

plt.show()

cv.imshow('Prueba-Gris',img_gray)
cv.imshow('Prueba-Color',img_color)
cv.waitKey(0)
cv.destroyAllWindows()
