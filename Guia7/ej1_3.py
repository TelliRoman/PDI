'''
La función cv.Laplacian de OpenCV calcula la derivada de segundo orden
(Laplaciano) de una imagen, lo que permite detectar bordes y zonas donde la 
intensidad cambia rápidamente.
dst = cv.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
src: Imagen de entrada (generalmente en escala de grises).
ddepth: Profundidad de la imagen de salida (cv.CV_8U, cv.CV_16S, cv.CV_64F, etc.).
dst (opcional): Imagen de salida. Si no se especifica, se crea una nueva.
ksize (opcional): Tamaño del kernel (debe ser impar y positivo, típicamente 1, 3, 5, 7). Si es 1, se usa el kernel Laplaciano más simple.
scale (opcional): Factor de escala para los resultados (por defecto es 1).
delta (opcional): Valor agregado a los resultados antes de almacenarlos en la imagen de salida (por defecto es 0).
borderType (opcional): Tipo de borde usado para el padding (por defecto es cv.BORDER_DEFAULT).
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\estanbul.tif',cv.IMREAD_GRAYSCALE)

ksizes = [1, 3, 5, 7]  # Tamaños de kernel a analizar

plt.figure(figsize=(16, 4))
for i, k in enumerate(ksizes, 1):
    lap = cv.Laplacian(img, cv.CV_64F, ksize=k)
    lap_abs = np.abs(lap)
    lap_uint8 = np.uint8(np.clip(lap_abs, 0, 255))
    _, lap_bin = cv.threshold(lap_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.subplot(2, len(ksizes), i)
    plt.title(f'Laplacian ksize={k}')
    plt.imshow(lap_uint8, cmap='gray')
    plt.axis('off')
    plt.subplot(2, len(ksizes), i + len(ksizes))
    plt.title(f'Binarizado ksize={k}')
    plt.imshow(lap_bin, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
