import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def ruido_sp(img, probs, probp, valors, valorp):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            if r < probp:
                img[i, j] = valorp
            elif r < probs + probp:
                img[i, j] = valors
    return img

def ruido_exp(img, a):
    img_ruido = img.copy().astype(float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = np.random.rand()
            ruido = -np.log(r) / a
            img_ruido[i, j] += ruido
    img_ruido = np.clip(img_ruido, 0, 255).astype(np.uint8)
    return img_ruido

img = np.zeros((600, 600), dtype=np.uint8) 
img[:,0:200] = 60
img[:,200:400] = 120
img[:,400:] = 180
histima = cv.calcHist([img], [0], None, [256], [0, 256])

ruido_gauss = cv.randn( np.zeros((600, 600), dtype=np.uint8) , 0, 15)
img_ruido_gauss = img + ruido_gauss
histima_ruido_gauss = cv.calcHist([img_ruido_gauss], [0], None, [256], [0, 256])

ruido_uniform = cv.randu( np.zeros((600, 600), dtype=np.uint8) , 0, 10)
img_ruido_uniform = img + ruido_uniform
histima_ruido_uniform = cv.calcHist([img_ruido_uniform], [0], None, [256], [0, 256])

img_ruido_sp = ruido_sp(img.copy(), 0.1, 0.1, 235, 20)
histima_ruido_sp = cv.calcHist([img_ruido_sp], [0], None, [256], [0, 256])

img_ruido_exp = ruido_exp(img.copy(), 0.1)
histima_ruido_exp = cv.calcHist([img_ruido_exp], [0], None, [256], [0, 256])

plt.figure(1,figsize=(10, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Original')
plt.subplot(122)
plt.bar(range(256), histima.ravel(), color='gray')
plt.title('Histograma Original')

plt.figure(2,figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_ruido_gauss, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Ruido Gaussiano')
plt.subplot(122)
plt.bar(range(256), histima_ruido_gauss.ravel(), color='gray')
plt.title('Histograma Ruido Gaussiano')

plt.figure(3,figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_ruido_uniform, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Ruido Uniforme')
plt.subplot(122)
plt.bar(range(256), histima_ruido_uniform.ravel(), color='gray')
plt.title('Histograma Ruido uniforme')

plt.figure(4,figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_ruido_sp , cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Ruido SP')
plt.subplot(122)
plt.bar(range(256), histima_ruido_sp.ravel(), color='gray')
plt.title('Histograma Ruido SP')

plt.figure(5,figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_ruido_exp , cmap='gray', vmin=0, vmax=255)
plt.title('Imagen Ruido EXP')
plt.subplot(122)
plt.bar(range(256), histima_ruido_exp.ravel(), color='gray')
plt.title('Histograma Ruido EXP')

plt.show()