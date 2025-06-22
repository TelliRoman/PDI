import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
'''
La función cv.HoughLinesP de OpenCV implementa la Transformada de Hough Probabilística para la
detección de líneas. Es más eficiente que la Hough clásica porque devuelve segmentos de línea en vez de líneas infinitas.

lines = cv.HoughLinesP(src, rho, theta, threshold[, minLineLength[, maxLineGap]])
Parámetros principales
src: Imagen de entrada (debe ser binaria, normalmente salida de Canny).
rho: Resolución en píxeles del parámetro ρ.
theta: Resolución en radianes del parámetro θ.
threshold: Número mínimo de votos para considerar que hay una línea.
minLineLength (opcional): Longitud mínima de línea aceptada.
maxLineGap (opcional): Máxima distancia permitida entre segmentos para unirlos en una sola línea.

¿Qué devuelve?
Devuelve un array de segmentos de línea, donde cada segmento está representado por sus dos puntos extremos:
[ [x1, y1, x2, y2], ... ]
Cada línea es un vector de 4 valores: (x1, y1) y (x2, y2).
cv.HoughLinesP detecta segmentos de línea, no líneas infinitas.
Los resultados se devuelven como un array de puntos extremos de cada segmento.
Es más eficiente y útil para aplicaciones prácticas donde se necesitan líneas finitas.
'''

def func_trackbar(x=None):
    minLinelength = cv.getTrackbarPos('minLineLength','TH')
    maxLinegap = cv.getTrackbarPos('maxLineGap','TH')
    accumulator = cv.getTrackbarPos('Umbral para acumulador','TH')
    # Detección de bordes
    bordes = cv.Canny(img, 70, 200, apertureSize=3, L2gradient=True)
    # Aplicar HoughLines
    lines = cv.HoughLinesP(bordes, 1, np.pi/180, accumulator, minLineLength=minLinelength,maxLineGap=maxLinegap)
    # Dibujar líneas sobre una copia de la imagen original
    img_color = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\snowman.png')
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow('TH', img_color)
    

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\snowman.png',cv.IMREAD_GRAYSCALE)

cv.namedWindow('TH')
cv.createTrackbar('minLineLength', 'TH', 0, 180, func_trackbar)
cv.createTrackbar('maxLineGap', 'TH', 0, 180, func_trackbar)
cv.createTrackbar('Umbral para acumulador', 'TH', 1, 300, func_trackbar)
func_trackbar()
cv.waitKey(0)
cv.destroyAllWindows()