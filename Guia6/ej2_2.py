import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ej2_1 import filtro_media_geom
from ej2_1 import filtro_contraarmonica
from ej1_1 import ruido_exp
from ej1_1 import ruido_sp

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

if __name__ == "__main__":
    img_sangre = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\sangre.jpg',cv.IMREAD_GRAYSCALE)

    ruido_gauss = cv.randn( np.zeros(img_sangre.shape, dtype=np.uint8) , 0, 20)
    img_sangre_ruido = img_sangre + ruido_gauss
    img_sangre_ruido = ruido_sp(img_sangre_ruido.copy(), 0.02, 0.02, 255, 0)

    img_media_geom = filtro_media_geom(img_sangre_ruido,3,3)
    img_contraar = filtro_contraarmonica(img_sangre_ruido,2,3,3)

    error_img_MD_Limpia = mse(img_sangre,img_media_geom)
    error_img_CA_Limpia = mse(img_sangre,img_contraar)

    error_img_Ruido_Limpia = mse(img_sangre_ruido,img_sangre)

    print('El error MSE de la imagen limpia comparada con la imagen filtrada por media geométrica es: ', error_img_MD_Limpia)
    print('El error MSE de la imagen limpia comparada con la imagen filtrada contra-armónica es: ', error_img_CA_Limpia)
    print('El error MSE de la imagen limpia comparada con la imagen ruidosa: ', error_img_Ruido_Limpia)    
    
    plt.figure(1,figsize=(12,10))
    plt.subplot(121)
    plt.imshow(img_sangre,cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Limpia')
    plt.subplot(122)
    plt.imshow(img_sangre_ruido,cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen con Ruido')

    plt.figure(2,figsize=(12,10))
    plt.subplot(121)
    plt.imshow(img_media_geom,cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtrada Media Geom')
    plt.subplot(122)
    plt.imshow(img_contraar,cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen Filtrada Contra Ar')

    plt.show()


'''Filtro de la media geométrica:
 Suaviza la imagen con menor pérdida de detalles de la imagen.
 Bueno para ruido gaussiano, falla con ruido impulsivo.'''

'''bFiltro de la media contra-armónica:
 Q: orden del filtro.
 Bueno para ruido sal y pimienta. Q > 0: elimina pimienta, Q < 0:
elimina sal.
 Q = 0: media aritmética, Q = −1: media armónica.
'''