import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
##1
##Cargar
imagen = cv.imread("Guia1\camino.tif")
##Visualizar
##Opcion 1
cv.imshow("Prueba",imagen)
cv.waitKey(0)
cv.destroyAllWindows
##Opcion 2
plt.imshow(imagen[:,:,::-1])
plt.show()
##2
print("Tamaño de la imagen: ",imagen.shape)
print("Tipo de dato de la imagen: ",imagen.dtype)
##3
print("El pixel",200,115, " es: ",imagen[200,115,:])
##5
##Opcion 1 sin mascara
H = imagen.shape[0]
W = imagen.shape[1]
plt.imshow(imagen[int(H/3):int(2*H/3),int(W/3):int(2*W/3),::-1])
plt.title("Parte de la Imagen")
plt.show()
##Opcion 2 con mascara
mask = np.zeros(imagen.shape,imagen.dtype)
mask[int(H/3):int(2*H/3),int(W/3):int(2*W/3)]=255 # Máscara rectangular
# operación AND pixel a pixel
img_recorte=cv.bitwise_and(imagen,mask)
plt.imshow(img_recorte)
plt.show()
##6
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,12))

# Por defecto, imshow toma imágenes HXCX3 como RGB, mientras que opencv usa BGR
ax[0].imshow(imagen)
ax[0].set_title("BGR")
ax[1].imshow(imagen[:,:,::-1]) # invierte los canales, ::-1 indica que se
ax[1].set_title("RGB")
plt.show()

##7
imagen2 = imagen.copy()
cv.line(imagen2,(0,0),(imagen.shape[1],imagen.shape[0]),(0,255,0),2)
plt.imshow(imagen2[:,:,[2,1,0]])
plt.show()
cv.circle(imagen,(int((imagen.shape[1])/2),int((imagen.shape[0]/2))),50,(0,255,0),2)
plt.imshow(imagen[:,:,[2,1,0]])
plt.show()
