import cv2
import numpy as np

def read_camera(path):
    return cv2.imread(path)

def read_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1,4)