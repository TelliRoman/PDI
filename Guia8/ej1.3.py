import numpy as np
import cv2 as cv

def actualizar(val):
    size = cv.getTrackbarPos('Tamaño', 'Imagen ajustable')
    if size < 1:
        size = 1
    if size % 2 == 0:
        size += 1  # Para asegurar tamaño impar (opcional)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))

    # Reprocesar la imagen original (sin modificarla directamente)
    _, img_bin = cv.threshold(img_gray, 245, 255, cv.THRESH_BINARY)
    img_inv = cv.bitwise_not(img_bin)
    img_eroded = cv.morphologyEx(img_inv, cv.MORPH_ERODE, kernel)

    cv.imshow('Imagen ajustable', img_eroded)

# Cargar la imagen original en escala de grises
img_gray = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\createch01.png', cv.IMREAD_GRAYSCALE)

# Validar si la imagen se cargó correctamente
if img_gray is None:
    print("No se pudo cargar la imagen.")
    exit()

# Crear ventana
cv.namedWindow('Imagen ajustable')

# Crear trackbar
cv.createTrackbar('Tamaño', 'Imagen ajustable', 17, 100, actualizar)

# Mostrar la imagen inicial
actualizar(0)  # Llamada inicial para mostrar con tamaño por defecto

cv.waitKey(0)
cv.destroyAllWindows()

# Crear el EE grande
kernelGRANDE = cv.getStructuringElement(cv.MORPH_RECT, (53, 53))

# Cargar la imagen y preprocesar
img_gray = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\createch01.png', cv.IMREAD_GRAYSCALE)
_, img_bin = cv.threshold(img_gray, 245, 255, cv.THRESH_BINARY)
img_inv = cv.bitwise_not(img_bin)
img_eroded = cv.morphologyEx(img_inv, cv.MORPH_ERODE, kernelGRANDE)
img=cv.morphologyEx(img_eroded,cv.MORPH_DILATE, kernelGRANDE)
img=cv.bitwise_not(img)
img=cv.bitwise_and(img_inv,img)


kernelGRANDE = cv.getStructuringElement(cv.MORPH_RECT, (53, 51))
img_eroded = cv.morphologyEx(img, cv.MORPH_ERODE, kernelGRANDE)
img_dilatada=cv.morphologyEx(img_eroded,cv.MORPH_DILATE, kernelGRANDE)
img_dilatad=cv.bitwise_not(img_dilatada)
img=cv.bitwise_and(img_dilatad,img)

cv.imshow('Imagen con EE grande', img)
cv.waitKey(0)