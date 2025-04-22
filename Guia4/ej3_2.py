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



img=cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\camino.tif')
b,g,r=cv.split(img)
img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
h,s,v=cv.split(img_hsv)
h1,s1,i1=rgb2hsi(r,g,b)
mask=np.array([[-2,-1,-1],
               [-1,11,-1],
               [-1,-1,-2]])
b_pasa_alto=cv.filter2D(b, -1, mask)
g_pasa_alto=cv.filter2D(g, -1, mask)
r_pasa_alto=cv.filter2D(r, -1, mask)
img_pasa_alto=cv.merge((b_pasa_alto,g_pasa_alto,r_pasa_alto))
h_pasa_alto=cv.filter2D(h, -1, mask)
i1_pasa_alto=cv.filter2D(i1, -1, mask)
img_hsv_pasa_alto=cv.merge((h_pasa_alto,s,i1_pasa_alto))
img_hsv_pasa_alto_bgr = cv.cvtColor(img_hsv_pasa_alto, cv.COLOR_HSV2BGR)
img_hsi_pasa_alto=cv.merge((h,s,i1_pasa_alto))
img_hsi_pasa_alto_bgr = cv.cvtColor(img_hsi_pasa_alto, cv.COLOR_HSV2BGR)

plt.figure(figsize=(15, 5))

# Imagen Original
#plt.subplot(1, 3, 1)
#plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#plt.title('Original')
#plt.axis('off')

# Imagen RGB pasa alto
plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(img_pasa_alto, cv.COLOR_BGR2RGB))
plt.title('RGB pasa alto')
plt.axis('off')
#Imagen HSV pasa alto
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(img_hsv_pasa_alto_bgr, cv.COLOR_BGR2RGB))
plt.title('HSV pasa alto')
plt.axis('off')
#Imagen HSI pasa alto
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(img_hsi_pasa_alto_bgr, cv.COLOR_BGR2RGB))
plt.title('HSI pasa alto')
plt.axis('off')

plt.tight_layout()
plt.show()