import numpy as np
import mediapipe as mp

# Indices de hombros en la lista de 9 keypoints
idx_hombro_izq = 7
idx_hombro_der = 8

def normalizarKeypoints(keypoints_xy):
    X_k = np.array(keypoints_xy).reshape(-1, 2)
    P_med = (X_k[idx_hombro_izq] + X_k[idx_hombro_der])/2
    X_rel_k = X_k - P_med
    D = np.linalg.norm(X_k[idx_hombro_izq] - X_k[idx_hombro_der])
    X_norm_k = X_rel_k / D
    return X_norm_k.flatten()