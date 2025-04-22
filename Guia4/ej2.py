import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def func_trackbar(x=None):
    a = cv.getTrackbarPos('Valor A', 'Ventana')
    mask = img < a
    # Convertir imagen a BGR (color)
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_color[mask] = [0, 255, 255]
    cv.imshow('Ventana', img_color)

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\rio.jpg', cv.IMREAD_GRAYSCALE)

histo = cv.calcHist([img], [0], None, [256], [0, 256])

plt.figure(0)
plt.plot(histo, color='blue', label='Histograma')
plt.title('Histograma de la imagen')

plt.figure(1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')

cv.namedWindow('Ventana')
cv.createTrackbar('Valor A', 'Ventana', 1, 255, func_trackbar)  # Valor inicial 0
func_trackbar()
'''plt.figure(2)
plt.imshow(cv.cvtColor(img_color, cv.COLOR_BGR2RGB))
plt.title('Imagen con valores menores a 2 en amarillo')'''
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()