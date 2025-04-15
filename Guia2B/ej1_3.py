import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\earth.bmp', cv.IMREAD_GRAYSCALE)

# Ecualizar el histograma
img_eq = cv.equalizeHist(img)

# Calcular histogramas
hist_orig = cv.calcHist([img], [0], None, [256], [0,256])
hist_eq = cv.calcHist([img_eq], [0], None, [256], [0,256])

# Mostrar todo en una figura
plt.figure()

# Imagen original
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Imagen original")
plt.axis('off')

# Histograma original
plt.subplot(2,2,2)
plt.plot(hist_orig, color='black')
plt.title("Histograma original")
plt.xlim([0,256])

# Imagen ecualizada
plt.subplot(2,2,3)
plt.imshow(img_eq, cmap='gray')
plt.title("Imagen ecualizada")
plt.axis('off')

# Histograma ecualizado
plt.subplot(2,2,4)
plt.plot(hist_eq, color='black')
plt.title("Histograma ecualizado")
plt.xlim([0,256])

plt.tight_layout()
plt.show()

''' ¿Qué diferencias se observan?
Desde la teoría:

La ecualización busca redistribuir los niveles de gris para usar más eficientemente el rango dinámico (0–255).

Idealmente, el histograma ecualizado debería ser uniforme (todos los niveles con frecuencia similar).

En la práctica:

El histograma no siempre es plano. La ecualización depende del contenido de la imagen.

Se nota una mejor distribución de intensidad, especialmente en imágenes oscuras o con poco contraste.

El contraste mejora visiblemente, resaltando detalles que antes estaban apagados.'''