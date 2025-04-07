import numpy as np
import cv2 as cv

def gen_LUT_dos_tramos(a1, c1, x1t1, a2, c2):
    r1 = np.arange(0, x1t1)
    r2 = np.arange(x1t1, 256)

    LUT1 = np.clip(a1 * r1 + c1, 0, 255).astype(np.uint8)
    LUT2 = np.clip(a2 * r2 + c2, 0, 255).astype(np.uint8)
    LUT = np.concatenate((LUT1, LUT2))
    return LUT

# Cargar imagen en escala de grises
img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\earth.bmp', cv.IMREAD_GRAYSCALE)
if img is None:
    print("❌ Error: No se pudo cargar la imagen. Verifica la ruta.")
    exit()

windowname = 'Ventana'
cv.namedWindow(windowname)

# Valores iniciales corregidos (Deben ser positivos)
a1_init = 1  # En OpenCV no puede ser negativo, después lo ajustamos en la función
c1_init = 255  # Para que el rango real sea -255 a 255
x1t1_init = 128
a2_init = 1
c2_init = 0

def func_trackbar(x=None):
    a1 = cv.getTrackbarPos('Valor A1', windowname) 
    c1 = cv.getTrackbarPos('Valor C1', windowname) - 255  # Rango real: -255 a 255
    x1t1 = cv.getTrackbarPos('Valor X1t1', windowname)
    
    lut = gen_LUT_dos_tramos(a1, c1, x1t1, a2_init, c2_init)
    img_modificada = cv.LUT(img, lut)
    cv.imshow(windowname, img_modificada)

# Crear trackbars con valores iniciales corregidos
cv.createTrackbar('Valor A1', windowname, a1_init, 100, func_trackbar)  
cv.createTrackbar('Valor C1', windowname, c1_init, 510, func_trackbar)  
cv.createTrackbar('Valor X1t1', windowname, x1t1_init, 255, func_trackbar)

func_trackbar()

cv.waitKey(0)
cv.destroyAllWindows()




