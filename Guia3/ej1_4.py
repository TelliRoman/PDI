import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\hubble.tif", cv.IMREAD_GRAYSCALE)

mask=np.ones((5,5), np.float32)/25
suavizado=cv.filter2D(img, -1, mask)
cv.imshow("Original", img)
cv.waitKey(0)
cv.imshow("Suavizado", suavizado)
cv.waitKey(0)

def gen_LUT_un_tramo(a, c):
    r = np.arange(256)
    LUT = np.clip(a * r + c, 0, 255).astype(np.uint8)
    return LUT

umbral = 100
a = 1000
c = -a * umbral
LUT = gen_LUT_un_tramo(a, c)
img2 = cv.LUT(suavizado, LUT)
cv.imshow("Binario", img2)
cv.waitKey(0)