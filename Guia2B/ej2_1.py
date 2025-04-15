
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

cuadros = cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\cuadros.tif",cv.IMREAD_GRAYSCALE)

def mostrar_histograma(imagen, titulo):
    hist = cv.calcHist([imagen], [0], None, [256], [0, 256])
    plt.figure()
    plt.title(titulo)
    plt.plot(hist, color='gray')
    plt.xlim([0, 256])
    plt.xlabel("Nivel de Intensidad")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.show()

#mostrar_histograma(cuadros, "Histograma - Cuadros")

clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
img_clahe = clahe.apply(cuadros)

cv.imshow('Resultado con máscara', img_clahe)
cv.waitKey(0)
cv.destroyAllWindows()
