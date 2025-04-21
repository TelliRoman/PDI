import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def rgb2hsi(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    i = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    s = np.zeros_like(i)
    denom = r + g + b
    mask = denom > 0
    s[mask] = 1 - (3 * min_rgb[mask] / denom[mask])
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1, 1))
    h = np.zeros_like(i)
    h[b <= g] = theta[b <= g]
    h[b > g] = 2 * np.pi - theta[b > g]
    h = h / (2 * np.pi)
    return h, s, i

# Cargamos imagen
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\patron.tif')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Convertimos a HSV
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)

# Mostramos la original
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen original")
plt.imshow(img_rgb)
plt.axis("off")

# Modificamos los canales
h[:, -29:] = 0  
s[:, :] = 255
v[:, :] = 255

# Unimos canales y convertimos a BGR para mostrar con matplotlib (que usa RGB)
img_modificada_hsv = cv.merge([h, s, v])
img_modificada_bgr = cv.cvtColor(img_modificada_hsv, cv.COLOR_HSV2BGR)
img_modificada_rgb = cv.cvtColor(img_modificada_bgr, cv.COLOR_BGR2RGB)

# Mostramos la imagen modificada
plt.subplot(1, 2, 2)
plt.title("Imagen modificada")
plt.imshow(img_modificada_rgb)
plt.axis("off")
plt.tight_layout()
plt.show()
