# Importar librerías necesarias
import cv2
import mediapipe as mp
import pickle
import time
import numpy as np
from normalizarKeypoints import *

# Inicializar MediaPipe Pose y el dibujador
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializa el modelo
pose = mp_pose.Pose()

# Inicializar constantes de texto
font = cv2.FONT_HERSHEY_SIMPLEX
fps_font_scale = 0.3
fps_thickness = 1
fps_color = (255, 255, 255)
fps_y = cv2.getTextSize("fps_text", font, fps_font_scale, fps_thickness)[1]

# Definir keypoints del torso
keypoints_torso = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_EYE,
    mp_pose.PoseLandmark.RIGHT_EYE,
    mp_pose.PoseLandmark.LEFT_EAR,
    mp_pose.PoseLandmark.RIGHT_EAR,
    mp_pose.PoseLandmark.MOUTH_LEFT,
    mp_pose.PoseLandmark.MOUTH_RIGHT,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]

# Cargar modelo entrenado
with open("deteccion-postura/modelo_postura.pkl", "rb") as f:
    model = pickle.load(f)

# Inicializar variables
prev_frame_time = 0

# Captura de imagen desde webcam
cap = cv2.VideoCapture(0)

while True:
    # Clasificar postura actual con el modelo entrenado
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calcular FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = current_time

    # Voltear horizontalmente para espejo
    frame = cv2.flip(frame, 1)

    # Convertir a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Mostrar FPS en esquina superior izquierda
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, fps_y+10), font, fps_font_scale, fps_color, fps_thickness, cv2.LINE_AA)

    # Si se detectan keypoints, calcular la postura
    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        # Extraer sólo los keypoints del torso: [x0, y0, x1, y1, ..., x8, y8]
        keypoints = []
        for idx in keypoints_torso:
            lm = landmarks[idx.value]
            keypoints.extend([lm.x, lm.y, lm.z])
        norm_keypoints = normalizarKeypoints(keypoints)

        # Clasificar postura
        prediction = model.predict(np.array([norm_keypoints]))[0]
        label = "Buena Postura" if prediction == "buena" else "Mala Postura"
        color = (0, 255, 0) if prediction == "buena" else (0, 0, 255)

        # Mostrar resultado centrado abajo
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        img_h, img_w = frame.shape[:2]
        x = (img_w - text_width) // 2
        y = img_h - 20  # 20 píxeles de margen inferior
        cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
        # (Opcional) Dibujar los keypoints
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Mostrar el frame con puntos
    cv2.imshow("Posture Keypoints", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
