import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def trazar_perfiles_rgb_hsv(img):
    # Extraemos las dimensiones
    alto, ancho, _ = img.shape
    fila = alto // 2  # Tomamos la fila central

    # Dividimos en canales RGB
    b, g, r = cv.split(img)

    # Convertimos a HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)

    # Tomamos los perfiles de la fila central
    perfil_r = r[fila, :]
    perfil_g = g[fila, :]
    perfil_b = b[fila, :]

    perfil_h = h[fila, :]
    perfil_s = s[fila, :]
    perfil_v = v[fila, :]

    # Graficamos perfiles RGB
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(perfil_r, color='red', label='R')
    plt.plot(perfil_g, color='green', label='G')
    plt.plot(perfil_b, color='blue', label='B')
    plt.title('Perfil de Intensidad - RGB')
    plt.xlabel('Columna')
    plt.ylabel('Intensidad')
    plt.legend()
    plt.grid(True)

    # Graficamos perfiles HSV
    plt.subplot(1, 2, 2)
    plt.plot(perfil_h, color='orange', label='H (Hue)')
    plt.plot(perfil_s, color='purple', label='S (Saturation)')
    plt.plot(perfil_v, color='gray', label='V (Value)')
    plt.title('Perfil de Intensidad - HSV')
    plt.xlabel('Columna')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\patron.tif')
trazar_perfiles_rgb_hsv(img)
