import numpy as np
import cv2 as cv

# Cargar imagen en escala de grises
img = cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\earth.bmp", cv.IMREAD_GRAYSCALE)

# Valores iniciales de A y C
a_init = 10  # Inicial en 1.0 (10/10)
c_init = 0

# Función para generar la LUT
def gen_LUT(a, c):
    r = np.arange(256)
    LUT = np.clip(a * r + c, 0, 255).astype(np.uint8)
    return LUT

# Función para actualizar y mostrar la imagen con los valores actuales de A y C
def mostrar_imagen():
    a = cv.getTrackbarPos("Valor de A", "Ajuste") / 1  # Escalado para valores más finos
    c = cv.getTrackbarPos("Valor de C", "Ajuste")

    LUT = gen_LUT(a, c)
    img_modificada = cv.LUT(img, LUT)

    cv.imshow("Ajuste", img_modificada)

# Crear ventana
cv.namedWindow("Ajuste")

# Crear trackbars
cv.createTrackbar("Valor de A", "Ajuste", a_init, 100, mostrar_imagen)
cv.createTrackbar("Valor de C", "Ajuste", c_init + 255, 510, mostrar_imagen)  # Rango de -255 a 255

# Mostrar imagen inicial
mostrar_imagen()

# Esperar interacción
cv.waitKey(0)
cv.destroyAllWindows()
