import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\esqueleto.tif', cv.IMREAD_GRAYSCALE)
A = 2  # Factor de amplificación

img=cv.boxFilter(img, -1, (7, 7), normalize=True)
mask3=np.array([[-2,-1,-1],
               [-1,10,-1],
               [-1,-1,-2]])
# Aplicar filtro
img2 = cv.filter2D(img, -1, mask3)
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\esqueleto.tif', cv.IMREAD_GRAYSCALE)
img_PB = cv.boxFilter(img2, -1, (3, 3), normalize=True)
img_AltaPotencia = cv.addWeighted(img, A, img_PB, -1, 0)
img_AltaPotencia = cv.normalize(img_AltaPotencia, None, 0, 255, cv.NORM_MINMAX)
img_AltaPotencia = img_AltaPotencia.astype(np.uint8)



# Mostrar la imagen resultante con matplotlib
plt.imshow(img_AltaPotencia, cmap='gray')
plt.title('Resultado Binario: Negro y Blanco')
plt.axis('off')
plt.show()
