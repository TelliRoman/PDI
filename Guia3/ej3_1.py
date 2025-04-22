import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def diferencia_opcion_uno(img1,img2):
    h, w, c = img1.shape
    img2 = cv.resize(img2, (w, h))  # Redimensiona img2 si es necesario

    # Calcular la diferencia
    diferencia = img1.astype(np.float32) - img2.astype(np.float32)

    # Opción 1: Reescalado sumando 255 y dividiendo por 2
    img_norma_1 = (diferencia + 255) / 2
    img_norma_1 = np.clip(img_norma_1, 0, 255).astype(np.uint8)

    # Opción 2: Restar el mínimo y escalar a 255
    img_min = np.min(diferencia)
    
    img_max = np.max(diferencia)
    img_norma_2 = (diferencia - img_min) / (img_max - img_min) * 255
    img_norma_2 = np.clip(img_norma_2, 0, 255).astype(np.uint8)

    return img_norma_1, img_norma_2


img1= cv.imread(r"C:\Users\pablo\Desktop\PDI\PDI\Imagenes\camaleon.tif", cv.IMREAD_GRAYSCALE)
mask_promedio=np.ones((3,3), np.float32)/9
img2=cv.filter2D(img1, -1, mask_promedio)
diferencia = cv.absdiff(img1, img2)
mask_promedio1=np.ones((5,5), np.float32)/9
img3=cv.filter2D(img1, -1, mask_promedio1)
diferencia2 = cv.absdiff(img1, img3)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

ax[0,0].imshow(diferencia, cmap='gray')
ax[0,0].set_title("Diferencia 1")
ax[0,1].imshow(diferencia2, cmap='gray')
ax[0,1].set_title("Diferencia 2")
plt.show()


