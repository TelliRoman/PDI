import numpy as np
import cv2 as cv

# === 1. Cargar imagen en escala de grises ===
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\lluviaEstrellas.jpg', cv.IMREAD_GRAYSCALE)

# === 2. Binarizar la imagen ===
# Se aplica un umbral: todo píxel mayor a 50 se vuelve 255 (blanco), el resto 0 (negro)
_, img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)

# === 3. Definición de kernels ===
# Kernel rectangular (1x1): efecto neutro, se usa más adelante para limpieza
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))

# Kernel en forma de diagonal (↘), diseñado para resaltar o preservar estructuras diagonales
mykernel = np.array([[0,0,0,0,1],
                     [0,0,0,1,0],
                     [0,0,1,0,0],
                     [0,1,0,0,0],
                     [1,0,0,0,0]], dtype=np.uint8)

# === 4. Erosión con kernel personalizado ===
# Se usa para eliminar estructuras que no coincidan con la forma del kernel (como ruido o estrellas no fugaces)
img = cv.morphologyEx(img, cv.MORPH_ERODE, mykernel)

# === 5. Apertura morfológica para limpiar ruido residual ===
# La apertura (erosión seguida de dilatación) elimina objetos pequeños aislados
img_limpia = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# === 6. Dilatación para resaltar los elementos que sobrevivieron (la estrella fugaz) ===
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
img_limpia = cv.morphologyEx(img_limpia, cv.MORPH_DILATE, kernel)

# === 7. Cargar la imagen original nuevamente para usarla como máscara base ===
img_original = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\lluviaEstrellas.jpg', cv.IMREAD_GRAYSCALE)

# === 8. Aplicar la máscara para dejar visibles solo los píxeles de la estrella fugaz ===
img_solo_estrellas = cv.bitwise_and(img_original, img_limpia)

# === 9. Mostrar el resultado ===
cv.imshow('Imagen solo estrella fugaz', img_solo_estrellas)
cv.waitKey(0)
