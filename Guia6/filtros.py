import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def filtro_rechazabanda_ideal(shape, coords_picos, W=10):
    """
    Genera la función de transferencia de un filtro ideal rechaza banda.
    shape: tupla (alto, ancho) del espectro.
    coords_picos: lista de coordenadas [(x1, y1), ...] de los picos a rechazar.
    W: ancho de la banda a rechazar (en píxeles).
    """
    H, W_ = shape
    filtro = np.ones((H, W_), dtype=np.float32)
    centro_u, centro_v = H // 2, W_ // 2

    for (col, fila) in coords_picos:
        du = fila - centro_u
        dv = col - centro_v

        D0 = np.sqrt(du**2 + dv**2)

        # Malla de coordenadas
        U, V = np.meshgrid(np.arange(W_), np.arange(H))
        D = np.sqrt((V - centro_u )**2 + (U - centro_v )**2)
        H_uv = np.ones_like(D)
        H_uv[(D >= D0 - W/2) & (D <= D0 + W/2)] = 0
        filtro *= H_uv

        # Simétrico respecto al centro
        D_sym = np.sqrt((V - centro_u )**2 + (U - centro_v )**2)
        H_uv_sym = np.ones_like(D)
        H_uv_sym[(D_sym >= D0 - W/2) & (D_sym <= D0 + W/2)] = 0
        filtro *= H_uv_sym

    return filtro

def filtro_rechazabanda_butterworth(shape, coords_picos, W, n=2):
    """
    Genera la función de transferencia de un filtro rechaza banda Butterworth.

    shape: tupla (alto, ancho) del espectro
    coords_picos: lista de coordenadas [(x1, y1), ...] de frecuencias a rechazar (en espectro centrado)
    W: ancho de la banda de rechazo
    n: orden del filtro
    """
    H, W_ = shape
    filtro = np.ones((H, W_), dtype=np.float32)
    centro_u, centro_v = H // 2, W_ // 2

    # Malla de coordenadas
    U, V = np.meshgrid(np.arange(W_), np.arange(H))
    D = np.sqrt((V - centro_u)**2 + (U - centro_v)**2)

    for (x, y) in coords_picos:
        du = y - centro_u
        dv = x - centro_v
        D0 = np.sqrt(du**2 + dv**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            term = (D * W) / (D**2 - D0**2 + 1e-8)
            H_uv = 1 / (1 + (term**(2 * n)))
            filtro *= H_uv

        # Parte simétrica (reflejo respecto al centro)
        D0_sym = D0
        with np.errstate(divide='ignore', invalid='ignore'):
            term_sym = (D * W) / (D**2 - D0_sym**2 + 1e-8)
            H_uv_sym = 1 / (1 + (term_sym**(2 * n)))
            filtro *= H_uv_sym

    return filtro

def filtro_notch_ideal(shape,coords_picos,M):
    H, W_ = shape
    filtro = np.ones((H, W_), dtype=np.float32)
    centro_u, centro_v = H // 2, W_ // 2

    # Malla de coordenadas
    U, V = np.meshgrid(np.arange(W_), np.arange(H))

    for (x, y) in coords_picos:
        du = y - centro_u
        dv = x - centro_v
        #Calcula el desplazamiento del pico respecto al centro.
        Dk = np.sqrt((V - centro_u - du)**2 + (U - centro_v - dv)**2) 
        Dk_sym = np.sqrt((V - centro_u + du)**2 + (U - centro_v + dv)**2)
        #Calcula la distancia de cada punto a la posición del pico y a su simétrico respecto al centro.
        filtro[(Dk <= M/2) | (Dk_sym <= M/2)] = 0
        #Pone a cero (bloquea) los puntos cercanos al pico y a su simétrico, formando el "notch".
    
    return filtro

def filtro_notch_butterworth(shape, coords_picos, D0, n=2):
    """
    Genera la función de transferencia de un filtro notch Butterworth.

    shape: tupla (alto, ancho) del espectro
    coords_picos: lista de coordenadas [(x1, y1), ...] de frecuencias a rechazar (en espectro centrado)
    D0: radio del notch (ancho de la banda de rechazo)
    n: orden del filtro
    """
    H, W_ = shape
    filtro = np.ones((H, W_), dtype=np.float32)  # Inicializa el filtro con unos (pasa todo)
    centro_u, centro_v = H // 2, W_ // 2         # Calcula el centro del espectro

    # Malla de coordenadas para todo el espectro
    U, V = np.meshgrid(np.arange(W_), np.arange(H))

    for (x, y) in coords_picos:
        du = y - centro_u                        # Desplazamiento vertical del pico respecto al centro
        dv = x - centro_v                        # Desplazamiento horizontal del pico respecto al centro

        # Calcula la distancia de cada punto al pico
        Dk = np.sqrt((V - centro_u - du)**2 + (U - centro_v - dv)**2)
        # Calcula la distancia de cada punto al simétrico del pico respecto al centro
        Dk_sym = np.sqrt((V - centro_u + du)**2 + (U - centro_v + dv)**2)

        # Calcula la función de transferencia Butterworth para el pico
        Hk = 1 / (1 + (D0 / (Dk + 1e-8))**(2 * n))
        # Calcula la función de transferencia Butterworth para el simétrico
        Hk_sym = 1 / (1 + (D0 / (Dk_sym + 1e-8))**(2 * n))

        # Multiplica ambos para formar el notch y lo aplica al filtro total
        filtro *= Hk * Hk_sym

    return filtro