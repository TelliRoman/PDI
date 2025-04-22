import numpy as np
import cv2 as cv

def seleccionar_roi(img):
    img_copia = img.copy()
    rect = cv.selectROI("Seleccioná la ROI y presioná ENTER o SPACE", img_copia, showCrosshair=True)
    cv.destroyWindow("Seleccioná la ROI y presioná ENTER o SPACE")
    x, y, w, h = rect
    roi = img[y:y+h, x:x+w]
    return roi, rect

def definir_elipsoide(roi, img, a, b, c):
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    b_roi, g_roi, r_roi = cv.split(roi)
    b_p = np.mean(b_roi)
    g_p = np.mean(g_roi)
    r_p = np.mean(r_roi)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i, j]
            if ((r - r_p)**2 / (a**2 + 1e-5) +
                (g - g_p)**2 / (b**2 + 1e-5) +
                (b - b_p)**2 / (c**2 + 1e-5)) <= 1:
                mask[i, j] = 255
            else:
                mask[i, j] = 0
    return mask

def definir_rectangulo(roi, img, a, b):
    """
    Segmenta la imagen usando un subespacio rectangular en HSV basado en H y S.

    Parámetros:
        roi: Región de interés seleccionada (en BGR)
        img: Imagen original (en BGR)
        a: Tolerancia para H
        b: Tolerancia para S

    Retorna:
        mask: Máscara binaria donde los píxeles dentro del rango H-S son 255
    """
    # Convertimos ROI e imagen completa a HSV
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Calculamos los promedios de H y S en la ROI
    h_roi, s_roi, _ = cv.split(hsv_roi)
    h_p = np.mean(h_roi)
    s_p = np.mean(s_roi)

    # Definimos los rangos con tolerancias a y b
    h_min = max(0, int(h_p - a))
    h_max = min(179, int(h_p + a))  # H está entre 0 y 179
    s_min = max(0, int(s_p - b))
    s_max = min(255, int(s_p + b))  # S está entre 0 y 255

    # Creamos la máscara usando solo H y S
    lower = np.array([h_min, s_min, 0])
    upper = np.array([h_max, s_max, 255])
    mask = cv.inRange(hsv_img, lower, upper) 
    #Para cada píxel (h, s, v) de hsv_img, comprueba: 
    #H_min ≤ h ≤ H_max  AND  S_min ≤ s ≤ S_max  AND  V_min ≤ v ≤ V_max
    #wSi todas las comparaciones son verdaderas, la máscara en esa posición vale 255; si alguna falla, vale 0.

    return mask


def nothing(x):
    pass

# Cargar imagen y seleccionar ROI
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\futbol.jpg')  
roi, rect = seleccionar_roi(img)

# Crear ventana con trackbars
cv.namedWindow("Segmentación Rectángulo (HSV)")
cv.createTrackbar("a (H)", "Segmentación Rectángulo (HSV)", 100, 300, nothing)
cv.createTrackbar("b (S)", "Segmentación Rectángulo (HSV)", 100, 300, nothing)

cv.namedWindow("Segmentación Elipsoide (RGB)")
cv.createTrackbar("a (R)", "Segmentación Elipsoide (RGB)", 100, 300, nothing)
cv.createTrackbar("b (G)", "Segmentación Elipsoide (RGB)", 100, 300, nothing)
cv.createTrackbar("c (B)", "Segmentación Elipsoide (RGB)", 100, 300, nothing)

while True:
    # Obtener los valores de los trackbars
    a_h = cv.getTrackbarPos("a (H)", "Segmentación Rectángulo (HSV)")  # H
    b_s = cv.getTrackbarPos("b (S)", "Segmentación Rectángulo (HSV)")  # S
    a_r = cv.getTrackbarPos("a (R)", "Segmentación Elipsoide (RGB)")  # R
    b_g = cv.getTrackbarPos("b (G)", "Segmentación Elipsoide (RGB)")  # G
    c_b = cv.getTrackbarPos("c (B)", "Segmentación Elipsoide (RGB)")  # B

    # Calcular la máscara para el rectángulo en HSV
    mask_rect = definir_rectangulo(roi, img, a_h, b_s)
    resultado_rect = cv.bitwise_and(img, img, mask=mask_rect)

    # Calcular la máscara para el elipsoide en RGB
    mask_ellipse = definir_elipsoide(roi, img, a_r, b_g, c_b)
    resultado_ellipse = cv.bitwise_and(img, img, mask=mask_ellipse)

    # Mostrar ambas imágenes: una con la segmentación HSV y otra con la segmentación RGB
    cv.imshow("Segmentación Rectángulo (HSV)", resultado_rect)
    cv.imshow("Segmentación Elipsoide (RGB)", resultado_ellipse)
    
    key = cv.waitKey(1)
    if key == 27:  # ESC para salir
        break

cv.destroyAllWindows()
