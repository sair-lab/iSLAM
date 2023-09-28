import numpy as np

def func(x, a, b, c):
    return a * np.exp(b * x) + c

def kitti_imu_func(rx, tx):
    ry = [0.0009536675582686507*x + 1e-06 if x < 0.005 else func(x, 0.00254876, 0.00375321, -0.00254304) for x in rx]
    # ty = [func(x, 0.00270267, 1.64663426, 0.05185503) for x in tx]
    ty = [0.05142675009977031 if x < 1 else func(x, 0.03021272, 0.90654935, -0.02337283) for x in tx]
    return np.array(ry), np.array(ty)

def kitti_vo_func(rx, tx):
    ry = [func(x, 0.00130778, 0.29282534, -0.00130624) for x in rx]
    ty = [0.024659894599410842*x + 0.001 if x < 0.5 else func(x, 3.68277025e-04, 2.12383050e+00, 1.22649253e-02) for x in tx]
    return np.array(ry), np.array(ty)
