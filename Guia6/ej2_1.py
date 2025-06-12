import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
#Filtros optimizados para media geometrica
#Matemáticamente: se usa el producto.
#Computacionalmente: se usa logaritmo y exponencial para evitar problemas numéricos, pero el resultado es el mismo.
def _media_geom_func_log(pixels):
    # Evita valores 0 para evitar log(0)
    pixels = np.where(pixels == 0, 1e-5, pixels)
    # Calcula la media geométrica usando logaritmos para evitar underflow
    return np.exp(np.mean(np.log(pixels)))

def filtro_media_geom_log(img, s, t):
    """
    Filtro de media geométrica usando logaritmos (estable y eficiente).
    """
    img = img.astype(np.float32)
    filtrada = generic_filter(img, _media_geom_func_log, size=(s, t), mode='nearest')
    return np.clip(filtrada, 0, 255).astype(np.uint8)

def _media_geom_func_prod(pixels):
    # Evita valores 0 para evitar raíz de 0
    pixels = np.where(pixels == 0, 1e-5, pixels)
    # Calcula el producto de todos los valores de la vecindad
    producto = np.prod(pixels)
    # Eleva a la potencia 1/n, donde n es el número de elementos en la vecindad
    return producto ** (1.0 / len(pixels))

def filtro_media_geom_prod(img, s, t):
    """
    Filtro de media geométrica usando el producto directo (idéntico a la fórmula matemática).
    Puede causar underflow en ventanas grandes.
    """
    img = img.astype(np.float32)
    filtrada = generic_filter(img, _media_geom_func_prod, size=(s, t), mode='nearest')
    return np.clip(filtrada, 0, 255).astype(np.uint8)

def filtro_media_geom(img,s,t):
    img = img.astype(np.float32)
    img_sauv = np.zeros_like(img)
    pad_s = s // 2
    pad_t = t // 2
    padded = cv.copyMakeBorder(img, pad_s, pad_s, pad_t, pad_t, cv.BORDER_REPLICATE)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            producto = 1.0
            for m in range(-pad_s, pad_s + 1):
                for n in range(-pad_t, pad_t + 1):
                    valor = padded[i + pad_s + m, j + pad_t + n]
                    # Evita valores 0 (log(0) o raíz de 0 no definidos)
                    if valor == 0:
                        valor = 1e-5
                    producto *= valor
            img_sauv[i, j] = producto ** (1.0 / (s * t))

    return np.clip(img_sauv, 0, 255).astype(np.uint8)

def filtro_contraarmonica(img, Q, s, t):
    img = img.astype(np.float32)
    img_sauv = np.zeros_like(img)
    pad_s = s // 2
    pad_t = t // 2
    padded = cv.copyMakeBorder(img, pad_s, pad_s, pad_t, pad_t, cv.BORDER_REPLICATE)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            num = 0.0  # Numerador: suma de I^(Q+1)
            den = 0.0  # Denominador: suma de I^Q
            for m in range(-pad_s, pad_s + 1):
                for n in range(-pad_t, pad_t + 1):
                    valor = padded[i + pad_s + m, j + pad_t + n]
                    num += valor ** (Q + 1)
                    if valor != 0:
                        den += valor ** Q
                    else:
                        den += 1e-5  # Evitar división por cero
            if den != 0:
                img_sauv[i, j] = num / den
            else:
                img_sauv[i, j] = 0

    return np.clip(img_sauv, 0, 255).astype(np.uint8)

    