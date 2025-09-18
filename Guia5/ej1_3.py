import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils 

img_lineav=np.zeros((512,512),np.uint8)
cv.line(img_lineav,(0,256),(512,256),255,1) # Dibuja una línea vertical en la imagen
img_lineav_rotada=imutils.rotate(img_lineav, 20) # Rota la imagen 20 grados
h,w=img_lineav.shape
cx,cy=w//2,h//2 # Coordenadas del centro de la imagen
roi_size=256
mitad_size=roi_size//2
roi_lineav=img_lineav[cy-mitad_size:cy+mitad_size,cx-mitad_size:cx+mitad_size] # Region de interes
roi_lineav_rotada=img_lineav_rotada[cy-mitad_size:cy+mitad_size,cx-mitad_size:cx+mitad_size] # Region de interes
# Transformada de Fourier de la imagen original y la imagen rotada
transformada_lineav = np.fft.fft2(roi_lineav)
transformada_lineav = np.fft.fftshift(transformada_lineav) # Desplaza el cero al centro de la imagen
transformada_lineav = np.log(20*np.abs(transformada_lineav) + 1) # Aplica logaritmo para mejorar la visualización

transformada_lineav_rotada = np.fft.fft2(roi_lineav_rotada)
transformada_lineav_rotada = np.fft.fftshift(transformada_lineav_rotada) # Desplaza el cero al centro de la imagen
transformada_lineav_rotada = np.log(20*np.abs(transformada_lineav_rotada) + 1) # Aplica logaritmo para mejorar la visualización

#graficar la transformada de Fourier de la imagen original y la transformada de Fourier
# de la imagen transformada
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(roi_lineav, cmap='gray'), plt.title('Imagen Original Linea Vertical')
plt.subplot(2, 2, 2), plt.imshow(transformada_lineav, cmap='jet'), plt.title('Transformada de Fourier Linea Vertical')
plt.subplot(2, 2, 3), plt.imshow(roi_lineav_rotada, cmap='gray'), plt.title('Imagen Original Linea Vertical Rotada')
plt.subplot(2, 2, 4), plt.imshow(transformada_lineav_rotada, cmap='jet'), plt.title('Transformada de Fourier Linea Vertical Rotada')
plt.tight_layout() 
plt.show()

# Al rotar la línea, esta ya no está alineada con los ejes de la imagen (ni horizontal ni vertical).
# La Transformada de Fourier ya no muestra una única línea en una dirección,
# sino que aparecen múltiples componentes diagonales.

# ¿Por qué? Porque la línea inclinada introduce información de frecuencia
# en *ambas direcciones* (u y v) del dominio de Fourier.

# Además, como la línea es una figura finita y discreta (píxeles), 
# no se puede representar con una sola frecuencia pura, por lo tanto se descompone
# en una suma de muchas frecuencias → aparecen varios picos en la transformada.

# Esto es un ejemplo visual de cómo la Transformada de Fourier refleja
# las *direcciones* y *frecuencias espaciales* presentes en la imagen original.



