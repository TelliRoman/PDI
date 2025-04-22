import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
    h = (h * 180).astype(np.uint8)
    s = (s *255 ).astype(np.uint8)
    i = (i * 255).astype(np.uint8)

    return h, s, i



img=cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\chairs_oscura.jpg')
b,g,r=cv.split(img)
img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v=cv.split(img_hsv)
h1,s1,i1=rgb2hsi(r,g,b)

# Histograma de cada canal
hist_r = cv.calcHist([r], [0], None, [256], [0, 256])
hist_g = cv.calcHist([g], [0], None, [256], [0, 256])
hist_b = cv.calcHist([b], [0], None, [256], [0, 256])
hist_v= cv.calcHist([v], [0], None, [256], [0, 256])
hist_i1= cv.calcHist([i1], [0], None, [256], [0, 256])
#-----------------------------------------------------------------
r_eq = cv.equalizeHist(r)
g_eq = cv.equalizeHist(g)
b_eq = cv.equalizeHist(b)
v_eq = cv.equalizeHist(v)
i1_eq = cv.equalizeHist(i1)

img_eq = cv.merge((b_eq, g_eq, r_eq))
img_hsv_eq = cv.merge((h, s, v_eq))
img_hsi_eq = cv.merge((h, s, i1_eq))

#-----------------------------------------------------------------
#Muestro las imagenes originales y luego la imagen ecualizada
img_hsv_eq_bgr = cv.cvtColor(img_hsv_eq, cv.COLOR_HSV2BGR)
# Mostrar
plt.figure(figsize=(15, 5))

# Imagen RGB ecualizada
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img_eq, cv.COLOR_BGR2RGB))
plt.title('RGB Ecualizado')
plt.axis('off')

# Imagen HSV ecualizada
plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img_hsv_eq, cv.COLOR_HSV2RGB))
plt.title('HSV Ecualizado')
plt.axis('off')

# Imagen HSI ecualizada - mostramos directamente como si fuera RGB
plt.subplot(1, 3, 3)
plt.imshow(cv.merge((h1, s1, i1_eq)))  # Aunque no sea RGB, lo mostramos como 3 canales
plt.title('HSI Ecualizado (visual)')
plt.axis('off')

plt.tight_layout()
plt.show()

