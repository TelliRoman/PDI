import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def prewitt(img, tipoborde):
    if tipoborde not in ('Todos','Vertical','Horizontal','Diagonal izq','Diagonal der'):
        print('Tipo de borde no valido')
        return None
    mask = np.zeros([3,3],np.int8)
        
    if tipoborde == 'Vertical':
        mask[:,0] = -1
        mask[:,2] = 1
        imgfilt = cv.filter2D(img.copy(), -1, mask)
    elif tipoborde == 'Horizontal':
        mask[0,:] = -1
        mask[2,:] = 1
        imgfilt = cv.filter2D(img.copy(), -1, mask)
    elif tipoborde == 'Diagonal izq':
        mask[0,0] = -1
        mask[1,0] = -1
        mask[1,1] = -1
        mask[0,2] = 1
        mask[1,2] = 1
        mask[2,2] = 1
        imgfilt = cv.filter2D(img.copy(), -1, mask)
    elif tipoborde == 'Diagonal der':
        mask[0,2] = -1
        mask[1,2] = -1
        mask[2,2] = -1
        mask[0,0] = 1
        mask[1,0] = 1
        mask[2,0] = 1
        imgfilt = cv.filter2D(img.copy(), -1, mask)
    else:  # 'Todos'
        mask_v = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        mask_h = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        img_v = cv.filter2D(img.copy(), -1, mask_v)
        img_h = cv.filter2D(img.copy(), -1, mask_h)
        imgfilt = cv.magnitude(img_v.astype(np.float32), img_h.astype(np.float32))
        imgfilt = np.uint8(imgfilt)

    # Binarización usando un umbral automático (Otsu) Con el flag cv.THRESH_BINARY + cv.THRESH_OTSU, el umbral se calcula automáticamente usando el método de Otsu, que busca separar los píxeles en dos grupos (fondo y borde) de la mejor manera posible.
    _, img_bin = cv.threshold(imgfilt, 0, 255, 195)#cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img_bin

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\estanbul.tif',cv.IMREAD_GRAYSCALE)
bordes = prewitt(img,'Todos')
plt.figure(0)
plt.imshow(bordes, cmap='gray', vmin=0, vmax=255)
plt.show()