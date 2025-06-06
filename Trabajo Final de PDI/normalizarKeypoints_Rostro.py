import numpy as np
import mediapipe as mp


def normalizarKeypoints(keypoints_xyz):
    X_k = np.array(keypoints_xyz).reshape(-1, 3)
    idx_A=X_k.shape[0]-2
    idx_B=X_k.shape[0]-1
    P_med = (X_k[idx_A] + X_k[idx_B]) / 2
    X_rel_k = X_k - P_med
    D = np.linalg.norm(X_k[idx_A] - X_k[idx_B])
    X_norm_k = X_rel_k / D
    return X_norm_k.flatten()