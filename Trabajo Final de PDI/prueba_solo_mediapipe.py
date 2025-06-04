import cv2
import mediapipe as mp
import numpy as np

# Inicialización de los módulos de MediaPipe para dibujo y detección holística
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Función para determinar si una imagen está borrosa usando la varianza del Laplaciano
def is_blurry(frame, threshold=100.0):
    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculamos la varianza del Laplaciano (detecta cambios de bordes)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Si la varianza es menor al umbral, se considera borrosa
    return lap_var < threshold, lap_var

# Función para medir la iluminación usando el canal V (brillo) del espacio HSV
def get_brightness(frame):
    # Convertimos a HSV para trabajar con el brillo
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # Canal V
    return np.mean(v_channel)  # Promedio del brillo

# Función para aplicar un desenfoque suave (máscara difusa)
def apply_soft_mask(frame, ksize=(5, 5)):
    # Aplicamos un filtro Gaussiano, útil para estabilizar ruido o movimientos bruscos
    return cv2.GaussianBlur(frame, ksize, 0)

# Inicializamos la cámara
cap = cv2.VideoCapture(0)

# Inicializamos MediaPipe Holistic (rostro + cuerpo + manos)
with mp_holistic.Holistic(
    static_image_mode=False,           # True si analizamos imágenes estáticas
    model_complexity=1,                # 0 = rápido pero menos preciso, 1 = balanceado, 2 = más preciso pero más lento
    smooth_landmarks=True,             # Suaviza las detecciones en video
    enable_segmentation=False,         # No segmentamos el cuerpo
    refine_face_landmarks=True,        # Detección refinada de rostro (68 puntos)
    min_detection_confidence=0.5,      # Confianza mínima para detectar
    min_tracking_confidence=0.5        # Confianza mínima para seguir puntos
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicamos desenfoque suave antes del análisis
        softened_frame = apply_soft_mask(frame)

        # Evaluamos borrosidad (Laplace)
        blurry, blur_score = is_blurry(softened_frame)

        # Medimos el brillo de la escena
        brightness = get_brightness(softened_frame)

        # Interpretamos el nivel de iluminación
        illumination_state = (
            "Oscura" if brightness < 60 else
            "Normal" if brightness < 200 else
            "Sobreexpuesta"
        )

        # Convertimos a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(softened_frame, cv2.COLOR_BGR2RGB)

        # Procesamos el frame con MediaPipe Holistic
        results = holistic.process(rgb_frame)

        # Copiamos el frame para dibujar sobre él
        annotated = softened_frame.copy()

        # --- DIBUJO DE LOS PUNTOS Y CONEXIONES ---

        # Dibuja los landmarks del rostro (68 puntos)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1)
            )

        # Dibuja los puntos del cuerpo (hombros, brazos, torso, etc.)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # Dibuja las manos si están visibles
        #if results.left_hand_landmarks:
         #   mp_drawing.draw_landmarks(
          #      annotated, results.left_hand_landmarks,
           #     mp_holistic.HAND_CONNECTIONS
            #)
        #if results.right_hand_landmarks:
         #   mp_drawing.draw_landmarks(
          #      annotated, results.right_hand_landmarks,
           #     mp_holistic.HAND_CONNECTIONS
            #)

        # --- TEXTO INFORMATIVO EN PANTALLA ---

        # Texto sobre borrosidad y valor de varianza
        cv2.putText(annotated, f"Borroso: {'SI' if blurry else 'NO'} | Var: {blur_score:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if blurry else (0, 255, 0), 2)

        # Texto sobre iluminación
        cv2.putText(annotated, f"Luz: {brightness:.1f} ({illumination_state})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Mostramos el resultado con todos los datos y dibujos
        cv2.imshow("MediaPipe + Calidad de Imagen", annotated)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberamos la cámara y cerramos ventanas
cap.release()
cv2.destroyAllWindows()
