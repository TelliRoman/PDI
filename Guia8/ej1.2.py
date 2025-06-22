import cv2 as cv
import numpy as np

#Kernerl para detectar lineas verticales
# 3. Crear kernels orientados
kernelh = cv.getStructuringElement(cv.MORPH_RECT, (30, 3))  # Línea horizontal
kernelv  = cv.getStructuringElement(cv.MORPH_RECT, (3, 30))  # Línea vertica
img_original= cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\fosforos.jpg', cv.IMREAD_GRAYSCALE)

_, img_original = cv.threshold(img_original, 245, 255, cv.THRESH_BINARY)
#invertir los colores de la imagen
img_original = cv.bitwise_not(img_original)

#Dejar solo verticales
fosforos_verical = cv.morphologyEx(img_original, cv.MORPH_ERODE, kernelv)
fosforos_verical = cv.morphologyEx(fosforos_verical, cv.MORPH_CLOSE, kernelv)

#Dejar solo horizontales
fosforos_horizontal = cv.morphologyEx(img_original, cv.MORPH_ERODE, kernelh)
fosforos_horizontal = cv.morphologyEx(fosforos_horizontal, cv.MORPH_CLOSE, kernelh)


#Mostrar imagen original y procesada
cv.imshow("Imagen original", img_original)
cv.imshow("Imagen procesada", fosforos_horizontal)
cv.waitKey(0)
cv.destroyAllWindows()

