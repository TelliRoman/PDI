import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils 
img_org=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\frecuencia.png",cv.IMREAD_GRAYSCALE)
# Transformada de Fourier de la imagen original y la imagen rotada
img = np.fft.fft2(img_org) # Aplica la transformada de Fourier
img = np.fft.fftshift(img) # Desplaza el cero al centro de la imagen
img = np.log(20*np.abs(img) + 1)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='jet'), plt.title('Transformada de Fourier Cuadros')
plt.subplot(2, 2, 2), plt.imshow(img_org, cmap='gray'), plt.title('Imagen Original Cuadros')
plt.tight_layout()
plt.show()