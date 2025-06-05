import time
import numpy as np
import cv2
import mediapipe as mp

# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------

def distancia(p1, p2):
    """
    Calcula la distancia Euclídea entre dos puntos (x1,y1) y (x2,y2).
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def punto(idx, face_landmarks, w, h):
    """
    Convierte un índice de landmark (0..467) a coordenadas de píxel (x, y).
    'idx' es el índice del punto en face_landmarks,
    'face_landmarks' es la lista de 468 landmarks (cada uno con .x y .y normalizados),
    'w', 'h' son ancho y alto de la imagen en píxeles.
    """
    lm = face_landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def compute_distance_vector(face_landmarks, w, h, indices):
    """
    Dado un conjunto de landmarks faciales (face_landmarks) y una lista
    de índices 'indices', calcula todas las distancias normalizadas
    entre pares de puntos de esa lista. Devuelve un vector 1D con esas distancias.
    
    Normalización: dividimos cada distancia Euclídea en píxeles por la diagonal
    de la imagen (sqrt(w^2 + h^2)), para garantizar valores en [0,1].
    """
    # Convertimos a coordenadas de píxeles todos los puntos de interés:
    coords = [punto(idx, face_landmarks, w, h) for idx in indices]
    diag = np.sqrt(w**2 + h**2)  # factor de normalización

    # Calcular distancias entre cada par (i < j):
    dist_list = []
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            d = distancia(coords[i], coords[j])
            dist_list.append(d / diag)  # normalizamos
    
    return np.array(dist_list, dtype=np.float32)  # resultado en forma de vector

# ---------------------------------------------------------
# Configuración de índices de AUs
# ---------------------------------------------------------
# Definimos los índices de MediaPipe FaceMesh para cada AU.
au_landmarks = {
    "AU4": [70, 63, 53, 105, 52, 66, 65, 107, 55, 285, 336, 295, 296, 282, 334, 283, 293, 300],
    "AU6": [352, 346, 347, 280, 266, 330, 425, 118, 101, 142, 36, 50, 117, 123],
    "AU7": [381, 380, 477, 373, 390, 256, 252, 253, 254, 339, 255, 359, 446, 26, 154, 22, 153, 23, 145, 24, 144, 110, 163, 25, 7, 130, 341, 382],
    "AU9": [64, 235, 98, 327, 294, 274, 459, 457, 278, 360, 363, 281, 5, 51, 134, 131, 79, 239, 44, 289, 59],
    "AU10": [185, 74, 39, 37, 0, 267, 269, 270, 409, 272, 271, 268, 12, 38, 41, 40],
    "AU43": [382, 381, 380, 477, 373, 390, 249, 263, 466, 388, 387, 386, 385, 476, 384, 398, 173, 157, 158, 159, 160, 161, 7, 33, 163, 144, 145, 153, 154]
}

# Construimos un set de índices únicos:
selected_indices = sorted({idx for lst in au_landmarks.values() for idx in lst})

# ---------------------------------------------------------
# Construcción del vector de pesos
# ---------------------------------------------------------
# Queremos dar mayor peso a cualquier distancia que involucre AU4, AU9 o AU10.
weight_AUs = set(au_landmarks["AU4"] + au_landmarks["AU9"] + au_landmarks["AU10"])
# Factor de peso para esos AUs (por ejemplo, 2.0)
high_weight = 10000000

# Calculamos el vector de pesos, en el mismo orden que compute_distance_vector genera las distancias:
weight_vector = []
n_sel = len(selected_indices)
for i in range(n_sel):
    for j in range(i + 1, n_sel):
        idx_i = selected_indices[i]
        idx_j = selected_indices[j]
        # Si alguno de los dos índices (i o j) está en weight_AUs, asignamos high_weight; si no, peso=1.0
        if (idx_i in weight_AUs) or (idx_j in weight_AUs):
            weight_vector.append(high_weight)
        else:
            weight_vector.append(1.0)
weight_vector = np.array(weight_vector, dtype=np.float32)

# (Opcional) Colores para dibujar cada AU
au_colors = {
    "AU4": (255, 0, 0),
    "AU6": (0, 255, 0),
    "AU7": (0, 128, 255),
    "AU9": (0, 0, 255),
    "AU10": (255, 0, 255),
    "AU43": (255, 255, 0)
}

# ---------------------------------------------------------
# Otras funciones de preprocesamiento (como antes)
# ---------------------------------------------------------

def is_blurry(frame, threshold=100.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var

def get_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)

def normalize_v_channel(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    v = 255 * (v - v.min()) / (v.max() - v.min() + 1e-6)
    hsv[:, :, 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ---------------------------------------------------------
# Programa principal
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Variables para calibración
baseline_samples = []
baseline_vector = None
samples_collected = 0
calibration_time = 5  # segundos
start_time = time.time()
last_sample_time = start_time

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Preprocesamiento de calidad de imagen
        blurry, blur_score = is_blurry(frame)
        if blurry:
            frame = sharpen_image(frame)

        brightness = get_brightness(frame)
        illumination_state = (
            "Oscura" if brightness < 60 else
            "Normal" if brightness < 200 else
            "Sobreexpuesta"
        )
        if illumination_state == "Oscura":
            frame = adjust_gamma(frame, gamma=1.5)
        elif illumination_state == "Sobreexpuesta":
            frame = adjust_gamma(frame, gamma=0.7)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            if (v.max() - v.min()) < 60:
                frame = normalize_v_channel(frame)

        # 2) Detectar landmarks con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        annotated = frame.copy()

        current_time = time.time()

        if results.face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.face_landmarks.landmark

            # FASE DE CALIBRACIÓN (primeros 5 segundos)
            if baseline_vector is None:
                # Tomar muestra una vez por segundo
                if (current_time - last_sample_time) >= 1.0 and samples_collected < calibration_time:
                    dist_vec = compute_distance_vector(landmarks, w, h, selected_indices)
                    baseline_samples.append(dist_vec)
                    samples_collected += 1
                    last_sample_time = current_time
                    print(f"Calibración: muestra {samples_collected}/5 registrada.")

                if samples_collected == calibration_time:
                    baseline_vector = np.mean(np.stack(baseline_samples, axis=0), axis=0)
                    print("Calibración completada. Baseline guardado.")
            else:
                # MODO EN TIEMPO REAL: calculamos desviación ponderada
                current_vec = compute_distance_vector(landmarks, w, h, selected_indices)
                diff = np.abs(current_vec - baseline_vector)

                # Calculamos weighted_diff y weighted_mean
                weighted_diff = diff * weight_vector
                pain_score = float(np.sum(weighted_diff) / np.sum(weight_vector))

                # Mostrar el puntaje de "dolor"
                cv2.putText(
                    annotated,
                    f"Dolor (peso AU4/AU9/AU10): {pain_score:.3f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # 3) Dibujo de puntos de AUs (visualización)
            for au, indices in au_landmarks.items():
                color = au_colors[au]
                for idx in indices:
                    x, y = punto(idx, landmarks, w, h)
                    cv2.circle(annotated, (x, y), 2, color, -1)
                fx, fy = punto(indices[0], landmarks, w, h)
                cv2.putText(annotated, au, (fx, fy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            cv2.putText(
                annotated,
                "No se detecta rostro",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # 4) Indicadores de calidad de imagen
        cv2.putText(
            annotated,
            f"Borroso: {'SI' if blurry else 'NO'} | Var: {blur_score:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if blurry else (0, 255, 0),
            2
        )
        cv2.putText(
            annotated,
            f"Luz: {brightness:.1f} ({illumination_state})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # 5) Mostramos la imagen
        cv2.imshow("Dolor Ponderado (AU4/AU9/AU10)", annotated)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
