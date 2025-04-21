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

def nothing(x):
    pass

# Cargar imagen y seleccionar ROI
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\futbol.jpg')  
roi, rect = seleccionar_roi(img)

# Crear ventana con trackbars
cv.namedWindow("Segmentación")
cv.createTrackbar("a", "Segmentación", 100, 300, nothing)
cv.createTrackbar("b", "Segmentación", 100, 300, nothing)
cv.createTrackbar("c", "Segmentación", 100, 300, nothing)

while True:
    a = cv.getTrackbarPos("a", "Segmentación")
    b = cv.getTrackbarPos("b", "Segmentación")
    c = cv.getTrackbarPos("c", "Segmentación")

    mask = definir_elipsoide(roi, img, a, b, c)
    resultado = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("Segmentación", resultado)
    
    key = cv.waitKey(1)
    if key == 27:  # ESC para salir
        break

cv.destroyAllWindows()
