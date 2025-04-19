import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\esqueleto.tif',cv.IMREAD_GRAYSCALE)

plt.figure(0)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.figure(1)
plt.subplot(1,3,1)
#img_suave = cv.GaussianBlur(img_eq, (5, 5), 1)
img_suave = cv.boxFilter(img,-1,(7,7),normalize=True)
plt.imshow(img_suave, cmap='gray')
plt.title('Suave')

kernel0 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)
# Aplicar filtro
img_filtro0 = cv.filter2D(img_suave, -1, kernel0)
plt.subplot(1,3,2)
plt.imshow(img_filtro0, cmap='gray')
plt.title('Pasa-altos (suma=0)')

A = 1.5
img_AltaPotencia = cv.addWeighted(img, A, img_filtro0, 1, 0)
img_AltaPotencia = cv.normalize(img_AltaPotencia, None, 0, 255, cv.NORM_MINMAX)
img_AltaPotencia = img_AltaPotencia.astype(np.uint8)
plt.subplot(1,3,3)
plt.imshow(img_AltaPotencia, cmap='gray')
plt.title('Alta Potencia')

plt.show()


'''
objtivo realzar los huesos. 
1ro se suaviza la imagen, para eliminar el ruido y detalles finos.
2do se aplica un filtro pasa-altos (suma=0) para resaltar los bordes y detalles.
3ro se aplica un filtro de alta potencia para realzar los detalles(huesos) sin perder la imagen base.
4to se normaliza la imagen resultante para que tenga un rango de valores adecuado.
5to se convierte a uint8 para que sea compatible con la visualización y procesamiento de imágenes.
6to se muestra la imagen original, la imagen suavizada, la imagen filtrada y la imagen de alta potencia.
'''