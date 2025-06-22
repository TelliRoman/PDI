import cv2 as cv
import numpy as np

# Cargar la imagen
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\createch01.png', cv.IMREAD_GRAYSCALE)

# Binarizar (si tu imagen ya es binaria, podés saltear esto)
_, img_bin = cv.threshold(img, 245, 255, cv.THRESH_BINARY)

# Crear un elemento estructurante
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # o (5,5)

# Calcular el gradiente morfológico
gradiente = cv.morphologyEx(img_bin, cv.MORPH_GRADIENT, kernel)

# Mostrar resultado
cv.imshow("Original", img)
cv.imshow("Gradiente Morfológico", gradiente)
cv.waitKey(0)
cv.destroyAllWindows()
