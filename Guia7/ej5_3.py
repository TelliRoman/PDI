import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

def media_geom_ventana(ventana):
    # Evita log(0) reemplazando ceros por un valor pequeño
    ventana = np.where(ventana == 0, 1e-5, ventana)
    return np.exp(np.mean(np.log(ventana)))

def filtro_media_geom(img, s, t):
    img = img.astype(np.float32)
    # Aplica la función media geométrica a cada ventana s x t
    filtrada = generic_filter(img, media_geom_ventana, size=(s, t), mode='nearest')
    return np.clip(filtrada, 0, 255).astype(np.uint8)

img1 = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\corrientes_ruidogris.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\iguazu_ruidogris.jpg', cv.IMREAD_GRAYSCALE)

img1_filtro = cv.medianBlur(img1, 3)
img1_filtro = filtro_media_geom(img1_filtro,3,3)
img2_filtro = cv.medianBlur(img2, 3)
img2_filtro = filtro_media_geom(img2_filtro,3,3)

plt.figure(1)
plt.subplot(121)
plt.imshow(img1,cmap='gray', vmin= 0 ,vmax= 255)
plt.subplot(122)
plt.imshow(img1_filtro,cmap='gray', vmin= 0 ,vmax= 255)
plt.figure(2)
plt.subplot(121)
plt.imshow(img2,cmap='gray', vmin= 0 ,vmax= 255)
plt.subplot(122)
plt.imshow(img2_filtro,cmap='gray', vmin= 0 ,vmax= 255)
plt.show()

def func_trackbar(x=None):
    umbralbajo1 = cv.getTrackbarPos('Bajo','Canny1')
    umbralalto1 = cv.getTrackbarPos('Alto','Canny1')
    
    umbralbajo2 = cv.getTrackbarPos('Bajo','Canny2')
    umbralalto2 = cv.getTrackbarPos('Alto','Canny2')
    global bordes1, bordes2
    bordes1 = cv.Canny(img1_filtro,umbralbajo1,umbralalto1,apertureSize=3,L2gradient=True)
    cv.imshow('Canny1', bordes1)
    bordes2 = cv.Canny(img2_filtro,umbralbajo2,umbralalto2,apertureSize=3,L2gradient=True)
    cv.imshow('Canny2', bordes2)

cv.namedWindow('Canny1')
cv.namedWindow('Canny2')

cv.createTrackbar('Bajo', 'Canny1', 0, 500, func_trackbar)
cv.createTrackbar('Alto', 'Canny1', 0, 500, func_trackbar)

cv.createTrackbar('Bajo', 'Canny2', 0, 500, func_trackbar)
cv.createTrackbar('Alto', 'Canny2', 0, 500, func_trackbar)

func_trackbar()
cv.waitKey(0)
cv.destroyAllWindows()

lineas1 = cv.HoughLines(bordes1, 1, np.pi/180, 100)
lineas2 = cv.HoughLines(bordes2, 1, np.pi/180, 100)

img1_filtro = cv.cvtColor(img1_filtro, cv.COLOR_GRAY2RGB)
img2_filtro = cv.cvtColor(img2_filtro, cv.COLOR_GRAY2RGB)

# Encuentra la línea más votada (mayor acumulador) para cada imagen
if lineas1 is not None:
    # En cv.HoughLines, cada línea es [rho, theta], pero no se retorna el valor de votos directamente.
    # Sin embargo, las líneas están ordenadas por votos descendente, así que la primera es la más votada.
    rho, theta = lineas1[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    #Convierte los parámetros polares (rho, theta) a coordenadas cartesianas.
    #(x0, y0) es un punto sobre la línea, a una distancia rho del origen, en la dirección theta.
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img1_filtro, (x1, y1), (x2, y2), (255, 0, 0), 2)

if lineas2 is not None:
    # En cv.HoughLines, cada línea es [rho, theta], pero no se retorna el valor de votos directamente.
    # Sin embargo, las líneas están ordenadas por votos descendente, así que la primera es la más votada.
    rho, theta = lineas2[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    #Convierte los parámetros polares (rho, theta) a coordenadas cartesianas.
    #(x0, y0) es un punto sobre la línea, a una distancia rho del origen, en la dirección theta.
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(img2_filtro, (x1, y1), (x2, y2), (255, 0, 0), 2)

plt.figure(1)
plt.imshow(img1_filtro)
plt.figure(2)
plt.imshow(img2_filtro)
plt.show()
