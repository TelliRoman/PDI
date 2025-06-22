import cv2 as cv 
import numpy as np
# Leo y binarizo la imagen por ultimo la invierto
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\tarjeta.jpeg', cv.IMREAD_GRAYSCALE)
_, img_bin = cv.threshold(img, 210, 255, cv.THRESH_BINARY)
img= cv.bitwise_not(img_bin)

# Aplico erosion para quedarme con las letras grandes
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
img_limpia= cv.morphologyEx(img, cv.MORPH_ERODE, kernel)
#Aplico dilatacion para agrandar las letras grandes (pero tambien se agranda el ruido)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
img_limpia= cv.morphologyEx(img_limpia, cv.MORPH_DILATE, kernel)
#Vou y a usar esta imagen de mascara con la imagen original
img_limpia= cv.bitwise_not(img_limpia)
#Aplico mascara y me quedon las letras mas chicas (todavia con ruido)
imagen_solo_chicas = cv.bitwise_and(img, img_limpia)
# La imagen solo chicas, tiene las letra chicas y el contorno de las letras grandes, entonces aplico 
# una erosion para quedarme con por lo menos las letras chicas sin las grandes, aunque sea silueta
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,2))
kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (2,1))
imagen_solo_chicas = cv.morphologyEx(imagen_solo_chicas, cv.MORPH_ERODE, kernel)
imagen_solo_chicas = cv.morphologyEx(imagen_solo_chicas, cv.MORPH_ERODE, kernel2)

# Ahora con esta silueta de las letras chicas, las agrando muchisimo para se junte lo mas que pueda
# ya que esa silueta de letras chicas, es mi ruido en letras grandes, entonces la voy a usar de mascara
kernel = cv.getStructuringElement(cv.MORPH_RECT, (17,17))
imagen_solo_chicas=cv.morphologyEx(imagen_solo_chicas, cv.MORPH_DILATE, kernel)

#Iniverto para poder tener la mascara de letras chicas, que son siluetas pero nos sirvr
imagen_solo_chicas= cv.bitwise_not(imagen_solo_chicas)
img_solo_grandes =cv.bitwise_not(img_limpia)
img_solo_grandes = cv.bitwise_and(imagen_solo_chicas, img_solo_grandes)

# Ahora tengo las letras grandes, limpias asi sin nada
cv.imshow("Letras grandes", img_solo_grandes)
cv.waitKey(0)
cv.destroyAllWindows()

#Las letras grandes me sirven de mascara para quedarme con las chicas
#Aplico la mascara
img = cv.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Imagenes\tarjeta.jpeg', cv.IMREAD_GRAYSCALE)
_, img_bin = cv.threshold(img, 210, 255, cv.THRESH_BINARY)
img= cv.bitwise_not(img_bin)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
img_solo_grandes = cv.morphologyEx(img_solo_grandes, cv.MORPH_DILATE, kernel)

img_solo_grandes= cv.bitwise_not(img_solo_grandes)
img_solo_chicas = cv.bitwise_and(img, img_solo_grandes)

cv.imshow("Letras chicas", img_solo_chicas)
cv.waitKey(0)
cv.destroyAllWindows()






























