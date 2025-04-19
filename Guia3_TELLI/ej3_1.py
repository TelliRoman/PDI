import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\camaleon.tif',cv.IMREAD_GRAYSCALE)

img_PB = cv.boxFilter(img , -1, (3,3),normalize=True)
img_MascaraDifusa = img - img_PB
img_MascaraDifusa = cv.normalize(img_MascaraDifusa, None, 0, 255, cv.NORM_MINMAX)
img_MascaraDifusa = img_MascaraDifusa.astype(np.uint8)

img_PB_Gauss = cv.GaussianBlur(img, (3, 3), 1)
img_MascaraDifusa_Gauss = img - img_PB_Gauss
img_MascaraDifusa_Gauss = cv.normalize(img_MascaraDifusa_Gauss, None, 0, 255, cv.NORM_MINMAX)
img_MascaraDifusa_Gauss = img_MascaraDifusa_Gauss.astype(np.uint8)

img_PB_Mediana = cv.medianBlur(img, 3)
img_MascaraDifusa_Mediana = img - img_PB_Mediana
img_MascaraDifusa_Mediana = cv.normalize(img_MascaraDifusa_Mediana, None, 0, 255, cv.NORM_MINMAX)
img_MascaraDifusa_Mediana = img_MascaraDifusa_Mediana.astype(np.uint8)

plt.figure(0)
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(img_PB, cmap='gray')
plt.title('Suavizada con Box 3x3')
plt.subplot(1,3,3)
plt.imshow(img_MascaraDifusa, cmap='gray')
plt.title('Mascara Difusa')

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(img_PB_Gauss, cmap='gray')
plt.title('Suavizada Gauss 3x3')
plt.subplot(1,3,3)
plt.imshow(img_MascaraDifusa_Gauss, cmap='gray')
plt.title('Mascara Difusa')

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.subplot(1,3,2)
plt.imshow(img_PB_Mediana, cmap='gray')
plt.title('Suavizada con Mediana 3x3')
plt.subplot(1,3,3)
plt.imshow(img_MascaraDifusa_Mediana, cmap='gray')
plt.title('Mascara Difusa')

plt.show()
'''
Toma la imagen original, le quita la parte suave (borrosa) y 
deja solo los detalles finos'''