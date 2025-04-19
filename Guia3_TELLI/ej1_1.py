import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\camaleon.tif',cv.IMREAD_GRAYSCALE)

mask1 = np.ones((3,3),np.float32)/9
mask2 = np.ones((3,3),np.float32)
mask2[0,0] = 0
mask2[2,0] = 0
mask2[2,2] = 0
mask2[0,2] = 0
mask2 = mask2/5
mask3 = np.ones((5,5),np.float32)/25

### Otras opciones de lo mismo ###
#img1 =cv.blur(img, (5,5), anchor=(-1,-1), borderType=cv.BORDER_DEFAULT)
#img2 = cv.boxFilter(img , -1, (3,3), anchor=(-1,-1), normalize=True, borderType=cv.BORDER_DEFAULT)

img1 = cv.filter2D(img,-1,mask1)
img2 = cv.filter2D(img,-1,mask2)
img3 = cv.filter2D(img,-1,mask3)

plt.figure(0)
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(2, 2, 2)
plt.imshow(img1, cmap='gray')
plt.title('Box 3x3')

plt.subplot(2, 2, 3)
plt.imshow(img3, cmap='gray')
plt.title('Box 5x5')

plt.subplot(2, 2, 4)
plt.imshow(img2, cmap='gray')
plt.title('Cruz 3x3')

plt.tight_layout()
plt.show()

'''
El filtro box 3x3 suaviza levemente.
El box 5x5 suaviza mucho más (más "borroso").
El filtro en cruz suaviza pero conserva más detalles que el box completo.'''
