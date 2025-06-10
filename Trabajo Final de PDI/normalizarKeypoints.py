import numpy as np
import mediapipe as mp

# Indices de hombros en la lista de 9 keypoints
idx_hombro_izq = 7
idx_hombro_der = 8

def normalizarKeypoints(keypoints_xyz, idx_A=idx_hombro_izq, idx_B=idx_hombro_der):
    X_k = np.array(keypoints_xyz).reshape(-1, 3)
    P_med = (X_k[idx_A] + X_k[idx_B]) / 2
    X_rel_k = X_k - P_med
    D = np.linalg.norm(X_k[idx_A] - X_k[idx_B])
    X_norm_k = X_rel_k / D
    return X_norm_k.flatten()