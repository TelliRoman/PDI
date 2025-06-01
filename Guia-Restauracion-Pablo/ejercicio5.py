import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\img_degradada.tif", cv.IMREAD_GRAYSCALE)

transformada = np.fft.fft2(img)
transformada = np.fft.fftshift(transformada) # Desplaza el cero al centro de la imagen
transformada = np.log(20*np.abs(transformada) + 1) 

plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(transformada, cmap='gray')
plt.title('Transformada de Fourier')
plt.axis('off')
plt.colorbar()
plt.show()
