import cv2 as cv
import numpy as np

def blister_analisis(img):
    errormax = 0.5 #Determina que cantidad de intensidad promedio menos a la foto original es aceptable para que este lleno
    umbral = 120 #Determina que es fondo y que es pastilla
    blister_com = cv.imread(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\blister_completo.jpg",cv.IMREAD_GRAYSCALE)
    intensidadprom = np.mean(blister_com)
    cv.imshow("Imagen", img)
    if (np.mean(img) >= intensidadprom - errormax):
        print("La imagen corresponde a un blister completo")
        cv.waitKey(0)
        cv.destroyAllWindows()
        return None
    else:
        vecpos = []
        for fila in range(0, 2):
            for col in range(0, 5):
                #cv.circle(img, (55+(col*50), 55+(fila*50)), 0, (0, 255, ), 2)
                #cv.imshow("Imagen", img)
                if img[55+(fila*50),55+(col*50)] < umbral:
                    vecpos.append((fila,col))
        print("La imagen corresponde a un blister incompleto con las siguientes pastillas faltantes;",vecpos)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return vecpos

blister_com = cv.imread(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\blister_completo.jpg",cv.IMREAD_GRAYSCALE)
blister_incom = cv.imread(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\blister_incompleto.jpg",cv.IMREAD_GRAYSCALE)
blister_analisis(blister_incom)
blister_analisis(blister_com)

