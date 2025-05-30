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

img = cv.imread(r'C:\Users\Roman\Documents\GitHub\PDI\Imagenes\mosquito.jpg',cv.IMREAD_GRAYSCALE)

gauss_20 = cv.randn(img.copy(), 0, 20)
gauss_20 = img + gauss_20
gauss_30 = cv.randn(img.copy(), 0, 30)
gauss_30 = img + gauss_30
gauss_10 = cv.randn(img.copy(), 0, 10)
gauss_10 = img + gauss_10
gauss_5 = cv.randn(img.copy(), 0, 5)
gauss_5 = img + gauss_5

impulsivo_0 = ruido_sp(img.copy(), 0, 0.05, 0, 0)      
impulsivo_255 = ruido_sp(img.copy(), 0.05, 0, 255, 0) 
impulsivo_128 = ruido_sp(img.copy(), 0.05, 0, 128, 0)  # gris medio
imagenes = [
    ("Original", img),
    ("Gauss 5", gauss_5),
    ("Gauss 10", gauss_10),
    ("Gauss 20", gauss_20),
    ("Gauss 30", gauss_30),
    ("Impulsivo 0", impulsivo_0),
    ("Impulsivo 255", impulsivo_255),
    ("Impulsivo 128", impulsivo_128)
]

def aplicar_detectores(imagen):
    # Sobel
    sobelx = cv.Sobel(imagen, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(imagen, cv.CV_64F, 0, 1, ksize=3)
    sobel = cv.magnitude(sobelx, sobely)
    sobel = np.abs(sobel)
    sobel = np.uint8(np.clip(np.abs(sobel), 0, 255))
    #_, resultado = cv.threshold(resultado, 30, 255, cv.THRESH_BINARY)
    _, sobel = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            
    # Laplaciano
    lap = cv.Laplacian(imagen, cv.CV_64F, ksize=3)
    lap_abs = np.abs(lap)
    lap_uint8 = np.uint8(np.clip(lap_abs, 0, 255))
    _, lap_bin = cv.threshold(lap_uint8, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Canny
    canny = cv.Canny(imagen, 100, 200)
    return sobel, lap_bin, canny

# Mostrar resultados
for nombre, im in imagenes:
    sobel, lap, canny = aplicar_detectores(im)
    plt.figure(figsize=(12, 3))
    plt.suptitle(f"Detectores sobre: {nombre}")
    plt.subplot(1, 4, 1)
    plt.imshow(im, cmap='gray')
    plt.title("Imagen")
    plt.axis('off')
    plt.subplot(1, 4, 2)
    plt.imshow(sobel, cmap='gray')
    plt.title("Sobel")
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(lap, cmap='gray')
    plt.title("Laplaciano")
    plt.axis('off')
    plt.subplot(1, 4, 4)
    plt.imshow(canny, cmap='gray')
    plt.title("Canny")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

'''
**Comparación:**  
- **Sobel**: sensible al ruido, especialmente al impulsivo.
- **Laplaciano**: muy sensible al ruido, resalta mucho el ruido impulsivo y gaussiano.
- **Canny**: más robusto, suele dar mejores resultados en presencia de ruido, pero puede perder bordes si el ruido es muy fuerte.

¿En qué zonas funciona mejor cada método y por qué?
Sobel:
Funciona bien en zonas con cambios de intensidad suaves y bordes bien definidos. Es bueno para detectar bordes orientados en X o Y,
 pero puede perder detalles en bordes diagonales o muy finos.
Laplaciano:
Detecta bordes en todas las direcciones y es muy sensible a cambios bruscos de 
intensidad. Resalta detalles finos, pero también responde mucho al ruido y a texturas pequeñas.
Canny:
Es el más robusto. Funciona mejor en zonas donde los bordes son claros y continuos,
ya que incluye supresión de no-máximos y umbralización doble. Puede perder bordes muy débiles, pero reduce la detección de falsos bordes.

Qué sucede con el ruido?
Sobel:
El ruido, especialmente el impulsivo, genera muchos falsos bordes. 
El método es sensible al ruido porque calcula derivadas locales.
Enfatiza los píxeles más cercanos al centro, consiguiendo una mejor
respuesta en presencia de ruido tipo gaussiano.
Laplaciano:
Es el más sensible al ruido de los tres. El ruido, tanto gaussiano como impulsivo, se amplifica mucho y aparecen muchos bordes falsos.
 Excesivamente sensible al ruido.
 Produce bordes dobles.
 No detecta dirección de los bordes.
 Se requiere determinar los cruces por cero.
 Su utilidad se limita a clasificar los puntos que pertenecen a la zona clara y a la zona
 oscura a cada lado del borde
Canny:
Es el más robusto frente al ruido, ya que incluye un paso de suavizado (Gaussiano)
antes de calcular los bordes. Sin embargo, si el ruido es muy fuerte, 
también puede fallar o perder bordes reales.

¿Con qué tipo de imágenes sacaría mejor provecho de los métodos?
Sobel:
Imágenes con bordes bien definidos y poco ruido, donde interesa detectar bordes en
direcciones específicas (horizontal o vertical).

Laplaciano:
Imágenes limpias, con detalles finos y sin mucho ruido. Útil para resaltar texturas
o detalles pequeños.

Canny:
Imágenes reales, con ruido moderado y bordes complejos. Es el método más general y
robusto para la mayoría de aplicaciones prácticas.

¿Qué tipo de preprocesamientos serían útiles?
Filtrado Gaussiano:
Suavizar la imagen antes de aplicar Sobel o Laplaciano ayuda a reducir el impacto
del ruido.

Mediana:
Muy útil para eliminar ruido impulsivo (salt-and-pepper) antes de cualquier
detector de bordes.

Normalización:
Mejorar el contraste de la imagen puede ayudar a que los bordes sean más 
detectables.
'''