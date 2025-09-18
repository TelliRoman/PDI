import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\camaleon.tif',cv.IMREAD_GRAYSCALE)

'''
Un filtro de mediana es un tipo de filtro no lineal muy usado en procesamiento de imágenes para:
Reducir el ruido impulsivo (también llamado “sal y pimienta”).
Preservar bordes mejor que los filtros lineales como el promedio (blur).

¿Cómo funciona?
1-Se toma un vecindario alrededor de cada píxel (por ejemplo, una ventana de 3x3).
2-Se ordenan los valores de esos píxeles.
3-Se reemplaza el valor central por la mediana de esos valores.'''

mediana = cv.medianBlur(img, 3)  # Tamaño 3x3

# Mostrar
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(mediana, cmap='gray')
plt.title('Filtrada con Mediana')

plt.tight_layout()
plt.show()
