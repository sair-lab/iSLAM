import cv2
import numpy as np


def gt_vel_stop_detector(vels, vel_th):
    return np.nonzero(np.linalg.norm(vels, axis=1) <= vel_th)[0]