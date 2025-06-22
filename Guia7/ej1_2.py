import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ---------- Cargar imagen ----------
#img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\patron_bordes.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\estanbul.tif',cv.IMREAD_GRAYSCALE)
if img is None:
    raise IOError("No se pudo cargar la imagen.")

# ---------- Mostrar perfiles de intensidad ----------
plt.figure(figsize=(8, 3))
plt.title("Perfil de intensidad (línea horizontal en el medio)")
plt.plot(img[img.shape[0] // 2])
plt.xlabel("Columnas")
plt.ylabel("Intensidad")
plt.grid()
plt.show()

# ---------- Parámetros a evaluar ----------
ddepths = [cv.CV_8U, cv.CV_64F]
ksizes = [3, 5, 7,-1]  # -1 para Scharr
'''
Los kernels de Scharr son matrices utilizadas para calcular derivadas en X e Y, 
optimizadas para mejorar la precisión respecto a Sobel, especialmente en imágenes
pequeñas.
Scharr X (detecta bordes verticales):
[[-3,  0,  3],
 [-10, 0, 10],
 [-3,  0,  3]]'''
dx_dy = [(1, 0), (0, 1), (1, 1)]  # x, y, ambos

# ---------- Procesamiento y visualización ----------
for ddepth in ddepths:
    for ksize in ksizes:
        plt.figure(figsize=(12, 4))
        plt.suptitle(f'ddepth={ddepth}, ksize={ksize}')

        for i, (dx, dy) in enumerate(dx_dy, 1):
            if (dx == dy and dx == 1):
                resultadox = cv.Sobel(img, ddepth, dx, 0, ksize)
                resultadoy = cv.Sobel(img, ddepth, 0, dy, ksize)
                # Asegura el tipo correcto para cv.magnitude
                if resultadox.dtype != np.float32 and resultadox.dtype != np.float64:
                    resultadox = resultadox.astype(np.float32)
                    resultadoy = resultadoy.astype(np.float32)
                resultado = cv.magnitude(resultadox, resultadoy)
            else:
                resultado = cv.Sobel(img, ddepth, dx, dy, ksize)
            # Siempre tomar valor absoluto y convertir a uint8 antes de umbralizar
            resultado = np.abs(resultado)
            resultado = np.uint8(np.clip(resultado, 0, 255))
            #_, resultado = cv.threshold(resultado, 30, 255, cv.THRESH_BINARY)
            _, resultado = cv.threshold(resultado, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            titulo = f"Bordes dx={dx}, dy={dy}"
            plt.subplot(1, 3, i)
            plt.imshow(resultado, cmap='gray')
            plt.title(titulo)
            plt.axis('off')

        plt.tight_layout()
        plt.show()