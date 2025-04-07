import cv2 as cv

# Función callback para capturar eventos del mouse
def mostrar_posicion(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        print(f"Mouse en fila={y}, columna={x}")
        print(f"Valor de intensidad: {img[y, x]}")

# Cargar imagen (puede ser cualquier imagen)
img = cv.imread(r"C:\Users\Roman\Documents\GitHub\PDI\Imagenes\blister_completo.jpg")

# Verificá que se cargó bien
if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()

# Mostrar imagen y asignar el callback
cv.namedWindow("Imagen")
cv.setMouseCallback("Imagen", mostrar_posicion)

while True:
    cv.imshow("Imagen", img)
    if cv.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cv.destroyAllWindows()
