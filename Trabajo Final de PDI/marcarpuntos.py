import cv2
import numpy as np
import mediapipe as mp

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Cargar tu imagen (reemplaza "tu_foto.jpg" con la ruta de tu imagen)

image = cv2.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Trabajo Final de PDI\rostro.jpg')
scale_percent = 50  # Reducir al 50%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
image = cv2.resize(image, (width, height))

if image is None:
    raise FileNotFoundError("¡No se pudo cargar la imagen!")

# Lista para guardar landmarks seleccionados
selected_landmarks = []

# Función para manejar clics del mouse
def click_event(event, x, y, flags, param):
    global selected_landmarks, image
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = image.shape[:2]
        min_dist = float('inf')
        closest_idx = -1
        for idx, lm in enumerate(face_landmarks.landmark):
            lm_x, lm_y = int(lm.x * w), int(lm.y * h)
            dist = np.sqrt((x - lm_x)**2 + (y - lm_y)**2)
            if dist < min_dist and dist < 20:  # Radio de 20px para selección
                min_dist = dist
                closest_idx = idx
        if closest_idx != -1:
            selected_landmarks.append(closest_idx)
            cv2.circle(image, (int(face_landmarks.landmark[closest_idx].x * w), 
                        int(face_landmarks.landmark[closest_idx].y * h)), 5, (0, 255, 0), -1)
            cv2.putText(image, str(closest_idx), (int(face_landmarks.landmark[closest_idx].x * w), 
                        int(face_landmarks.landmark[closest_idx].y * h) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"Landmark seleccionado: {closest_idx}")
            cv2.imshow("Seleccionar Landmarks", image)

# Procesar la imagen con MediaPipe
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if results.multi_face_landmarks:
    face_landmarks = results.multi_face_landmarks[0]
    
    # Dibujar todos los landmarks (puntos verdes)
    for idx, lm in enumerate(face_landmarks.landmark):
        x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
else:
    print("No se detectaron rostros en la imagen.")
    exit()

# Mostrar la imagen y configurar el evento de clic
cv2.namedWindow("Seleccionar Landmarks")
cv2.setMouseCallback("Seleccionar Landmarks", click_event)

print("Instrucciones:")
print("- Haz clic en un landmark para seleccionarlo (se marca en verde).")
print("- Presiona 'x' para mostrar los landmarks guardados en consola.")
print("- Presiona 'r' para resetear la selección.")
print("- Presiona 'q' para salir.")

while True:
    cv2.imshow("Seleccionar Landmarks", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        print("\nLandmarks seleccionados:", selected_landmarks)
    elif key == ord('r'):
        selected_landmarks = []
        image = cv2.imread(r'C:\Users\pablo\Desktop\PDI\PDI\Trabajo Final de PDI\rostro.jpg')
        # Redibujar landmarks base
        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        print("Selección reseteada.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()