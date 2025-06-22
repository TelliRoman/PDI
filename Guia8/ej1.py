import cv2 as cv
import numpy as np

# Un kernel en forma de cruz 3x3 hecho a mano
kernela = np.array([[0, 0, 0],
                          [1, 0, 1],
                          [0, 0, 0]], dtype=np.uint8)
kernelb = np.array([[1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1]], dtype=np.uint8)

kernelc = np.array([[1, 0, 1],
                    [1, 0, 0],
                    [1, 0, 1]], dtype=np.uint8)

kerneld = np.array([[1, 0, 1],
                    [1, 0, 0],
                    [1, 1, 1]], dtype=np.uint8)

kernele = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)

kernelf=np.array([[1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0]], dtype=np.uint8)


img= cv.imread(r'C:\Users\pablo\OneDrive\Desktop\PDI\Imagenes\ferrari-c.jpg', cv.IMREAD_GRAYSCALE)
#binarizar la imagen
_, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# Aplicar las operaciones morfológicas
imga = cv.morphologyEx(img, cv.MORPH_ERODE, kernele)
#imga = cv.morphologyEx(imga, cv.MORPH_ERODE, kernele)
#imga = cv.morphologyEx(imga, cv.MORPH_ERODE, kernele)

# Redimensionar (50% del tamaño original)
img_resized = cv.resize(imga, (0, 0), fx=0.2, fy=0.2)

# Mostrar
cv.imshow("Imagen achicada", img_resized)
cv.waitKey(0)
cv.destroyAllWindows()

