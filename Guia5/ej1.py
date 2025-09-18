import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils 


img_lineah=np.zeros((100,100),np.uint8)
cv.line(img_lineah,(0,50),(100,50),255,10) # Dibuja una línea horizontal en la imagen
img_lineav=np.zeros((100,100),np.uint8)
cv.line(img_lineav,(50,0),(50,100),255,10) # Dibuja una línea vertical en la imagen
img_circulo=np.zeros((100,100),np.uint8)
cv.circle(img_circulo,(50,50),10,255,-1) # Dibuja un círculo en la imagen
img_rectangulo_centrado=np.zeros((100,100),np.uint8)
cv.rectangle(img_rectangulo_centrado,(30,30),(70,70),255,-1) # Dibuja un rectángulo centrado en la imagen

#En la transformada de la linea horizonal, la línea se convierte en una línea vertical en el dominio de la frecuencia, y viceversa para la línea vertical. 
#El círculo y el rectángulo centrado tienen un comportamiento diferente, ya que su transformada de Fourier muestra patrones más complejos debido a su forma y simetría.
#El circulo, su transformada de Fourier mostrar un patron radial, mientras mas chico el circulo, mas concentrado el patron radial.
#El rectángulo centrado, su transformada de Fourier mostrara un patron rectangular, y al igual que el circulo, mientras mas chico el rectangulo, mas concentrado el patron rectangular.

transformada_lineah = np.fft.fft2(img_lineah)
transformada_lineah = np.fft.fftshift(transformada_lineah) # Desplaza el cero al centro de la imagen
transformada_lineah = np.log(20*np.abs(transformada_lineah) + 1) # Aplica logaritmo para mejorar la visualización

transformada_lineav = np.fft.fft2(img_lineav)
transformada_lineav = np.fft.fftshift(transformada_lineav) # Desplaza el cero al centro de la imagen
transformada_lineav = np.log(20*np.abs(transformada_lineav) + 1) # Aplica logaritmo para mejorar la visualización

transformada_circulo = np.fft.fft2(img_circulo)
transformada_circulo = np.fft.fftshift(transformada_circulo) # Desplaza el cero al centro de la imagen
transformada_circulo = np.log(20*np.abs(transformada_circulo) + 1) # Aplica logaritmo para mejorar la visualización

transformada_rectangulo = np.fft.fft2(img_rectangulo_centrado)
transformada_rectangulo = np.fft.fftshift(transformada_rectangulo) # Desplaza el cero al centro de la imagen
transformada_rectangulo = np.log(20*np.abs(transformada_rectangulo) + 1) # Aplica logaritmo para mejorar la visualización
#graficar la transformada de Fourier de la imagen original y la transformada de Fourier
# de la imagen transformada
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img_lineah, cmap='gray'), plt.title('Imagen Original Linea Horizontal')
plt.subplot(2, 2, 2), plt.imshow(transformada_lineah, cmap='jet'), plt.title('Transformada de Fourier Linea Horizontal')
plt.subplot(2, 2, 3), plt.imshow(img_lineav, cmap='gray'), plt.title('Imagen Original Linea Vertical')
plt.subplot(2, 2, 4), plt.imshow(transformada_lineav, cmap='jet'), plt.title('Transformada de Fourier Linea Vertical')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img_circulo, cmap='gray'), plt.title('Imagen Original Linea Horizontal')
plt.subplot(2, 2, 2), plt.imshow(transformada_circulo, cmap='jet'), plt.title('Transformada de Fourier Linea Horizontal')
plt.subplot(2, 2, 3), plt.imshow(img_rectangulo_centrado, cmap='gray'), plt.title('Imagen Original Linea Vertical')
plt.subplot(2, 2, 4), plt.imshow(transformada_rectangulo, cmap='jet'), plt.title('Transformada de Fourier Linea Vertical')
plt.tight_layout()
plt.show()
