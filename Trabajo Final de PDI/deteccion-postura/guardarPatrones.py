import cv2
import mediapipe as mp
import time
from normalizarKeypoints import *

# Inicializar MediaPipe Pose y el dibujador
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Inicializa el modelo
pose = mp_pose.Pose()

# Captura de video
cap = cv2.VideoCapture(0)

# Inicializar constantes
frames_a_guardar = 250  # Número de frames a guardar
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

# Inicializar tiempo
prev_frame_time = 0

# Inicializar variables para saber si ya se cargaron los datos de postura
datos_buena_postura = 0
datos_mala_postura = 0
guardando_buena_postura = False
guardando_mala_postura = False

while True:
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

    # Si se detectan keypoints, dibujarlos
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Si se presiona 'c', cambiar el estado para pasar a guardar datos de postura
    if cv2.waitKey(1) & 0xFF == ord('c'):
        if datos_buena_postura == 0: # Si no hay datos de buena postura
            guardando_buena_postura = True # Iniciar guardado de buena postura
        elif datos_mala_postura == 0 and not guardando_buena_postura: # Si se finalizó el guardado de buena postura y no hay datos de mala postura
            guardando_mala_postura = True # Iniciar guardado de mala postura

    # Guardar keypoints de buena postura
    if guardando_buena_postura and results.pose_landmarks:
        if datos_buena_postura < frames_a_guardar:
            keypoints = []
            for idx in keypoints_torso:
                lm = results.pose_landmarks.landmark[idx.value]
                keypoints.extend([lm.x, lm.y, lm.z])
            norm_keypoints = normalizarKeypoints(keypoints)    
            with open(f'deteccion-postura/datos_postura/buena_{datos_buena_postura}.csv', "w") as f:
                f.write(",".join(map(str, norm_keypoints)))
            datos_buena_postura += 1
        else:
            guardando_buena_postura = False

    # Guardar keypoints de mala postura
    if guardando_mala_postura and results.pose_landmarks:
        if datos_mala_postura < frames_a_guardar:
            keypoints = []
            for idx in keypoints_torso:
                lm = results.pose_landmarks.landmark[idx.value]
                keypoints.extend([lm.x, lm.y, lm.z])
            norm_keypoints = normalizarKeypoints(keypoints)
            with open(f"deteccion-postura/datos_postura/mala_{datos_mala_postura}.csv", "w") as f:
                f.write(",".join(map(str, norm_keypoints)))
            datos_mala_postura += 1
        else:
            guardando_mala_postura = False

    # Mostrar distintos mensajes según si se han cargado los datos de postura
    if datos_buena_postura == 0:
        text = "Presiona 'C' para guardar datos de buena postura"
        color = (255, 255, 255)  # Blanco
    elif datos_mala_postura == 0:
        text = "Presiona 'C' para guardar datos de mala postura"
        color = (255, 255, 255)  # Blanco
    else:
        text = "Datos de postura guardados correctamente"
        color = (255, 255, 255)  # Blanco

    if guardando_buena_postura:
        text = "Guardando datos de buena postura..."
        color = (255, 255, 255)  # Blanco

    if guardando_mala_postura:
        text = "Guardando datos de mala postura..."
        color = (255, 255, 255)  # Blanco

    # Mostrar el texto en la imagen centrado abajo
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Tamaño de imagen
    img_h, img_w = frame.shape[:2]

    # Calcular coordenadas para centrar abajo
    x = (img_w - text_width) // 2
    y = img_h - 20  # 20 píxeles de margen inferior
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Mostrar FPS en esquina superior izquierda
    fps_text = f"FPS: {int(fps)}"
    fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
    cv2.putText(frame, fps_text, (10, fps_y+10), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrar cantidad de buena postura en esquina superior izquierda
    fps_text = f"Buena postura: {int(datos_buena_postura)}"
    fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
    cv2.putText(frame, fps_text, (10, fps_y+25), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrar cantidad de buena postura en esquina superior izquierda
    fps_text = f"Mala postura: {int(datos_mala_postura)}"
    fps_y = cv2.getTextSize(text, font, font_scale, thickness)[1]
    cv2.putText(frame, fps_text, (10, fps_y+40), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


    # Mostrar el frame con puntos
    cv2.imshow("Posture Keypoints", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()