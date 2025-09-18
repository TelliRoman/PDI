import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\estanbul.tif',cv.IMREAD_GRAYSCALE)

kernel = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

img_filtrada = cv.filter2D(img, -1, kernel)

# Mostrar
plt.figure(0)
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Pasa-altos (suma=1)")
plt.imshow(img_filtrada, cmap='gray')

# Máscara pasa-alto con suma 0 (realce puro)
kernel0 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Aplicar filtro
img_filtro0 = cv.filter2D(img, -1, kernel0)
# Mostrar
plt.figure(1)
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Pasa-altos (suma=0)")
plt.imshow(img_filtro0, cmap='gray')
plt.show()
'''
Un filtro pasa-altos deja pasar los detalles finos (bordes, texturas) y atenúa 
las áreas suaves o uniformes (como el cielo en una foto). 
Se usa para realzar bordes o detalles.

Suma=1: realce de altas frecuencias sin alterar las bajas frecuencias.
Realzan detalles, pero mantienen algo del contenido original → imagen más "nítida".

Suma=0: extracción de altas frecuencias, eliminando las bajas frecuencias.
eliminan el contenido plano (DC), y solo muestran bordes y detalles → imagen tipo "esqueleto", sin tonos base.
'''