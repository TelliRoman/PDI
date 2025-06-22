import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''
La función cv.Canny detecta bordes en una imagen usando el **algoritmo de Canny**,
que es uno de los métodos más populares y robustos para detección de bordes.
### ¿Cómo funciona Canny?
1. **Suavizado:** Aplica un filtro Gaussiano para reducir el ruido.
2. **Gradiente:** Calcula el gradiente de intensidad en cada píxel (usando Sobel).
3. **No-maximum suppression:** Afina los bordes, dejando solo los máximos locales.
4. **Umbralización doble:** Usa dos umbrales (`threshold1` y `threshold2`) para clasificar los bordes como fuertes, débiles o no bordes.
5. **Hysteresis:** Conecta los bordes débiles a los fuertes si están conectados, descartando los débiles aislados.
### Parámetros principales
- **src:** Imagen de entrada (en escala de grises).
- **threshold1:** Primer umbral para la función de histéresis (bajo).
- **threshold2:** Segundo umbral para la función de histéresis (alto).
- **edges:** Imagen de salida (opcional).
- **apertureSize:** Tamaño del kernel Sobel usado para calcular el gradiente (por defecto 3).
- **L2gradient:** Si es True, usa la norma L2 para el gradiente (más preciso, pero más lento).
**Devuelve:**  
Una imagen binaria donde los bordes detectados son blancos (255) y el resto negro (0).'''

def func_trackbar(x=None):
    umbralbajo = cv.getTrackbarPos('Bajo','Canny')
    umbralalto = cv.getTrackbarPos('Alto','Canny')
    checkbox = cv.getTrackbarPos('L2gradient','Canny')
    l2gradient = True if checkbox == 1 else False
    
    bordes = cv.Canny(img,umbralbajo,umbralalto,apertureSize=3,L2gradient=l2gradient)
    cv.imshow('Canny', bordes)

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\estanbul.tif',cv.IMREAD_GRAYSCALE)

cv.namedWindow('Canny')
cv.createTrackbar('Bajo', 'Canny', 0, 500, func_trackbar)
cv.createTrackbar('Alto', 'Canny', 0, 500, func_trackbar)
cv.createTrackbar('L2gradient', 'Canny', 0, 1, func_trackbar)
func_trackbar()
cv.waitKey(0)
cv.destroyAllWindows()