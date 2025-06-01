import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''
### ¿Qué tipo de imágenes son apropiadas para utilizar con TH?
- Imágenes binarizadas o con bordes resaltados (por ejemplo, tras aplicar TH).
## ¿Qué preprocesos serían útiles?
- **Detección de bordes** (TH, Sobel, etc.) para resaltar los contornos.
- **Binarización** para separar fondo y objetos.
- **Filtrado** (Gaussiano o Mediana) para reducir el ruido antes de detectar bordes.
- **Normalización** para mejorar el contraste si es necesario
### ¿Qué se obtiene (en el espacio transformado) al aplicar TH a un punto?
- Al aplicar la TH a un punto de la imagen, se obtiene una **curva (senoide) en el espacio de parámetros** (por ejemplo, en el espacio (ρ, θ) para líneas).
- Cada punto de la imagen genera una curva en el espacio de Hough.
### ¿Qué particularidad presentan (en el espacio transformado) los puntos colineales?
- Los puntos colineales en la imagen generan **curvas que se cruzan en un mismo punto** del espacio de Hough.
- Ese punto de intersección representa la línea común a todos esos puntos en la imagen original
### ¿Qué espera y qué no, como resultado del proceso de la TH?
**Se espera:**
- Detectar líneas (o círculos, dependiendo de la variante) presentes en la imagen.
- Obtener los parámetros (por ejemplo, ρ y θ) de las líneas detectadas.
- Que las líneas más votadas correspondan a las más evidentes en la imagen.
**No se espera:**
- Detectar líneas si los bordes no están bien definidos o hay mucho ruido.
- Distinguir líneas muy cercanas o con pocos puntos de soporte.
- Detectar formas complejas que no se ajustan al modelo buscado (por ejemplo, líneas curvas si se busca líneas rectas).
**Resumen:**  
La Transformada de Hough es ideal para detectar líneas (o círculos) en imágenes con bordes claros y poco ruido,
tras un buen preprocesamiento. En el espacio de Hough, los puntos colineales generan intersecciones que permiten identificar las líneas presentes en la imagen

La función cv.HoughLines detecta líneas rectas en una imagen binaria usando la Transformada de Hough.
### Parámetros principales
lines = cv.HoughLines(src, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]])

- **src**: Imagen de entrada (debe ser binaria, normalmente salida de TH).
- **rho**: Resolución en píxeles del parámetro ρ (distancia al origen).
- **theta**: Resolución en radianes del parámetro θ (ángulo de la línea).
- **threshold**: Número mínimo de votos (intersecciones en el espacio de Hough) para considerar que hay una línea.
- **lines** (opcional): Salida para las líneas detectadas.
- **srn** y **stn** (opcional): Usados para la Transformada de Hough multi-escala (normalmente se dejan en 0).
- **min_theta** y **max_theta** (opcional): Rango de ángulos a buscar (en radianes).

### ¿Qué devuelve?
Una lista de líneas detectadas, cada una representada por un par (ρ, θ).
**Resumen:**  
cv.HoughLines busca líneas rectas en una imagen binaria, devolviendo los parámetros
de cada línea detectada en el espacio (ρ, θ).
Si necesitas los votos, debes usar la función cv.HoughLinesWithAccumulator (no siempre disponible en todas las versiones de OpenCV), que devuelve [ρ, θ, votes].
'''

def func_trackbar(x=None):
    min_theta = cv.getTrackbarPos('Min Theta','TH') - 90
    max_theta = cv.getTrackbarPos('Max Theta','TH') - 90
    accumulator = cv.getTrackbarPos('Umbral para acumulador','TH')
    # Convertir grados a radianes
    min_theta = np.deg2rad(min_theta)
    max_theta = np.deg2rad(max_theta)
    # Detección de bordes
    bordes = cv.Canny(img, 70, 200, apertureSize=3, L2gradient=True)
    # Aplicar HoughLines
    lines = cv.HoughLines(bordes, 1, np.pi/180, accumulator, min_theta=min_theta, max_theta=max_theta)
    # Dibujar líneas sobre una copia de la imagen original
    img_color = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\snowman.png')
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #Convierte los parámetros polares (rho, theta) a coordenadas cartesianas.
            #(x0, y0) es un punto sobre la línea, a una distancia rho del origen, en la dirección theta.
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow('TH', img_color)
    

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\snowman.png',cv.IMREAD_GRAYSCALE)

cv.namedWindow('TH')
cv.createTrackbar('Min Theta', 'TH', 0, 180, func_trackbar)
cv.createTrackbar('Max Theta', 'TH', 0, 180, func_trackbar)
cv.createTrackbar('Umbral para acumulador', 'TH', 1, 300, func_trackbar)
func_trackbar()
cv.waitKey(0)
cv.destroyAllWindows()


'''
Para pasar del espacio de Hough (ρ, θ) al espacio cartesiano (x, y) y dibujar una línea, se usan las siguientes ecuaciones y pasos matemáticos:
---
### 1. **Ecuación de la recta en el espacio de Hough**
La ecuación de una línea en el espacio de Hough es:
ρ = x·cos(θ) + y·sin(θ)
donde:
- **ρ**: distancia perpendicular desde el origen al punto más cercano de la línea,
- **θ**: ángulo entre el eje x y la perpendicular a la línea.
### 2. **Encontrar puntos (x, y) de la línea**
Para dibujar la línea en la imagen, se buscan dos puntos suficientemente alejados sobre la línea.  
Se despeja (x, y) para dos valores arbitrarios (por ejemplo, usando un parámetro t):
a = np.cos(theta)
b = np.sin(theta)
x0 = a * rho
y0 = b * rho
- (x0, y0) es un punto sobre la línea, a distancia ρ del origen en la dirección θ.
Luego, para obtener dos puntos extremos sobre la línea:
x1 = int(x0 + 1000 * (-b))
y1 = int(y0 + 1000 * (a))
x2 = int(x0 - 1000 * (-b))
y2 = int(y0 - 1000 * (a))
Esto genera dos puntos (x1, y1) y (x2, y2) que están lejos entre sí y garantizan que la línea cruce toda la imagen.
### 3. **Resumen de pasos matemáticos*
1. **Obtener (ρ, θ) de la Transformada de Hough.**
2. **Calcular:**
   - a = cos(θ)
   - b = sin(θ)
   - x0 = a·ρ
   - y0 = b·ρ
3. **Obtener dos puntos extremos:**
   - (x1, y1) = (x0 + 1000·(-b), y0 + 1000·a)
   - (x2, y2) = (x0 - 1000·(-b), y0 - 1000·a)
4. **Dibujar la línea entre (x1, y1) y (x2, y2) en la imagen.**

**Así se pasa del espacio de Hough (ρ, θ) al espacio cartesiano (x, y) para graficar la línea detectada.**'''