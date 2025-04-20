import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\camaleon.tif',cv.IMREAD_GRAYSCALE)
A = 1.5  # Factor de amplificación

img_PB = cv.boxFilter(img , -1, (3,3),normalize=True)
img_AltaPotencia = cv.addWeighted(img, A, img_PB, -1, 0)
img_AltaPotencia = cv.normalize(img_AltaPotencia, None, 0, 255, cv.NORM_MINMAX)
img_AltaPotencia = img_AltaPotencia.astype(np.uint8)

img_PB_Gauss = cv.GaussianBlur(img, (3, 3), 1)
img_AltaPotencia_Gauss = cv.addWeighted(img, A, img_PB_Gauss, -1, 0)
img_AltaPotencia_Gauss = cv.normalize(img_AltaPotencia_Gauss, None, 0, 255, cv.NORM_MINMAX)
img_AltaPotencia_Gauss = img_AltaPotencia_Gauss.astype(np.uint8)

plt.figure(0, figsize=(12, 6))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(img_PB, cmap='gray')
plt.title('Suavizada con Box 3x3')
plt.subplot(1,3,3)
plt.imshow(img_AltaPotencia, cmap='gray')
plt.title('Alta Potencia')

plt.figure(1, figsize=(12, 6))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(img_PB_Gauss, cmap='gray')
plt.title('Suavizada Gauss 3x3')
plt.subplot(1,3,3)
plt.imshow(img_AltaPotencia_Gauss, cmap='gray')
plt.title('Alta Potencia')

plt.show()

'''
El filtrado de alta potencia es una técnica que realza los detalles finos (como bordes y texturas) en una imagen sin eliminar la información de baja frecuencia (fondos suaves, formas grandes). 
Es como afilar una foto, pero manteniendo los tonos y estructuras originales.

Comparación con otros métodos
Técnica	Resultado
Suavizado (blur)	Se pierden detalles, imagen borrosa
Máscara difusa	    Solo se ven los detalles (alta frecuencia)
Alta potencia	    Se realzan detalles sin perder la imagen base ✔'''