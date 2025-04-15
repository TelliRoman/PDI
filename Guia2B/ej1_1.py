import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

imagen = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\patron.tif', cv.IMREAD_GRAYSCALE)
imagen2 = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\patron2.tif',cv.IMREAD_GRAYSCALE)

#en la imagen 1 espero obtener un histograma en donde haya pixeles equilibrados en toda las intensidades de grises menos en las intensidades mas cercanas a 255
#en la imagen 2 espero obtener un histograma en donde haya la mitad de los pixeles en la intensidad 0 y en la 255
histr = cv.calcHist([imagen], [0], None, [256], [0, 256])
histr2 = cv.calcHist([imagen2], [0], None, [256], [0, 256])
plt.figure(0)
plt.imshow(imagen2, 'gray')
plt.title('Imagen 2')
plt.figure(1)
plt.imshow(imagen, 'gray')
plt.title('Imagen 1')
plt.figure(2)
plt.plot(histr, color='black')
plt.figure(3)
plt.plot(histr2, color='red')
plt.show()
