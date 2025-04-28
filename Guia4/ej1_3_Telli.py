import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def perfil_intensidad(img,filocol,num):
    if(filocol == 'fila'):
        b = img[num,:,0]
        g = img[num,:,1]
        r = img[num,:,2]
        img_aux = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h = img_aux[num,:,0]
        s = img_aux[num,:,1]
        v = img_aux[num,:,2]
    else: 
        b = img[:,num,0]
        g = img[:,num,1]
        r = img[:,num,2]
        img_aux = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h = img_aux[:,num,0]
        s = img_aux[:,num,1]
        v = img_aux[:,num,2]
    plt.figure(2)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Imagen Original')

    plt.figure(0)
    plt.plot(b, color='blue', label='B')
    plt.plot(g, color='green', label='G')
    plt.plot(r, color='red', label='R')
    plt.title('Perfil de Intensidad RGB')
    plt.legend()

    plt.figure(1)
    plt.plot(h, color='blue', label='H')
    plt.plot(s, color='green', label='S')
    plt.plot(v, color='red', label='V')
    plt.title('Perfil de Intensidad HSV')
    plt.legend()
    plt.show()
    return None

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\patron.tif')
perfil_intensidad(img,'fila',img.shape[0]//2)