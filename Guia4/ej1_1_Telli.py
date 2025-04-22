import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def spli_hsi(img):
    b,g,r = cv.split(img)    
    i =  (b + g + r) / 3
    h,s,v = cv.split(img_hsv)
    return h,s,i

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\patron.tif')
b,g,r = cv.split(img)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h,s,v = cv.split(img_hsv)

plt.figure(0)
plt.imshow(img_rgb)
plt.title('Original')

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(r, cmap='gray')
plt.title('Canal R')

plt.subplot(1,3,2)
plt.imshow(g, cmap='gray')
plt.title('Canal G')

plt.subplot(1,3,3)
plt.imshow(b, cmap='gray')
plt.title('Canal B')

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(h, cmap='gray')
plt.title('Canal H')

plt.subplot(1,3,2)
plt.imshow(s, cmap='gray')
plt.title('Canal S')

plt.subplot(1,3,3)
plt.imshow(v, cmap='gray')
plt.title('Canal V')

h = 120 - h # Al ir de 0 a 120 (rojo a azul) y vamos de 120 a 0 (azul a rojo)
# el 120 es el azul pq hsv usa 179 como maximo y mapea la vuelta de 0 a 360 en 0 a 179
# por lo tanto el azul que cae a los 240 grados es 120 en hsv
#h = np.linspace(120, 0, img.shape[0]).astype(np.uint8)  # Horizontal gradiente
#h = np.tile(h, (img.shape[1], 1))  # Repetimos por filas para que cubra toda la imagen
s = np.ones(s.shape, dtype=np.uint8) * 255
v = np.ones(s.shape, dtype=np.uint8) * 255
img_modificada = cv.merge((h,s,v))
img_modificada_RGB = cv.cvtColor(img_modificada, cv.COLOR_HSV2RGB)
plt.figure(3)
plt.imshow(img_modificada_RGB)
plt.title('Imagen Modificada')

# HSV
plt.figure(4)
plt.subplot(1,3,1)
plt.imshow(h, cmap='gray')
plt.title('Canal H modificado')
plt.subplot(1,3,2)
plt.imshow(s, cmap='gray')
plt.title('Saturación')
plt.subplot(1,3,3)
plt.imshow(v, cmap='gray')
plt.title('Brillo')

# RGB
r_mod, g_mod, b_mod = cv.split(img_modificada_RGB)
plt.figure(5)
plt.subplot(1,3,1)
plt.imshow(r_mod, cmap='gray')
plt.title('Canal R')
plt.subplot(1,3,2)
plt.imshow(g_mod, cmap='gray')
plt.title('Canal G')
plt.subplot(1,3,3)
plt.imshow(b_mod, cmap='gray')
plt.title('Canal B')

plt.show()