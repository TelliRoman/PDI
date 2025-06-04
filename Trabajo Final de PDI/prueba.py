import cv2
import numpy as np
import time
import dlib
import mediapipe as mp

# --- CARGA DEL MODELO DE DLIB ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\pablo\Desktop\PDI\PDI\Trabajo Final de PDI\shape_predictor_68_face_landmarks.dat')

# MediaPipe
#mp_face_mesh = mp.solutions.face_mesh
#face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

def is_blurry(frame, threshold=100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var

def get_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    return np.mean(v_channel)

def sharpen_image(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    enhanced_frame = sharpen_image(frame.copy())
    blurry, blur_score = is_blurry(enhanced_frame)
    brightness = get_brightness(enhanced_frame)

    # -------- Detección con dlib --------
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(enhanced_frame, (x, y), 1, (255, 0, 0), -1)  # Azul para dlib

    # -------- Overlay de texto --------
    text_blur = "Borroso" if blurry else "Borroso no detectado"
    illumination_state = (
        "Oscura" if brightness < 60 else
        "Iluminación normal" if brightness < 200 else
        "Sobreexpuesta"
    )

    cv2.putText(enhanced_frame, f"{text_blur} | Var: {blur_score:.1f} | Lum: {brightness:.1f} ({illumination_state})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(enhanced_frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # -------- Mostrar ventana --------
    cv2.imshow("Detección dlib + MediaPipe", enhanced_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
