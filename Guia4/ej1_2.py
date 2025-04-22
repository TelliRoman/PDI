import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def complementario_bgr(img):
    b, g, r = cv.split(img)
    b = 255 - b
    g = 255 - g
    r = 255 - r
    return cv.merge((b, g, r))
def complementario_hsv(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)
    h = (h + 90) % 180  # Rotar el matiz 180 GRADOS es decir 90 en hsv
    return cv.merge((h, s, v))

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\rosas.jpg')
img_comp_bgr = complementario_bgr(img)
img_comp_hsv = complementario_hsv(img)

plt.figure(0)
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img_comp_bgr, cv.COLOR_BGR2RGB))
plt.title('Complementario BGR')
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img_comp_hsv, cv.COLOR_HSV2RGB))
plt.title('Complementario HSV')
plt.show()

'''
1. Complementario BGR:
Estás haciendo una inversión directa de cada canal:
R = 255 - R, G = 255 - G, B = 255 - B.
Esto genera una inversión global de los colores en el espacio RGB, 
sin respetar matices perceptuales.
El resultado puede parecer raro o poco natural (por ejemplo, 
los fondos quedan negros si eran blancos).

2. Complementario HSV:
Estás rotando el matiz (H) 180°, lo cual significa:
El rojo se vuelve cian, el azul se vuelve amarillo, etc.
Mantenés saturación y valor (S y V), por lo que los colores resultantes:
Tienen el mismo brillo e intensidad, pero un matiz opuesto.
El resultado es más fiel al concepto perceptual de "complementario", 
por eso la imagen parece más armoniosa y natural.'''