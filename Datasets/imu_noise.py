import numpy as np

# vel_mean = np.array([0., 0., 0.],dtype=np.float32)
# vel_dev = np.array([1., 1., 1.],dtype=np.float32)
# accel_mean = np.array([0., 0., 0.],dtype=np.float32)
# accel_dev = np.array([1., 1., 1.],dtype=np.float32)
# gyro_mean = np.array([0., 0., 0.],dtype=np.float32)
# gyro_dev = np.array([1., 1., 1.],dtype=np.float32)
# angles_mean = np.array([0., 0., 0.],dtype=np.float32)
# angles_dev = np.array([1., 1., 1.],dtype=np.float32)

# # statistics from Jiasen
# vel_mean = np.array([-0.0111,0.0151,-0.0021],dtype=np.float32)
# vel_dev = np.array([1.329,1.441,0.764],dtype=np.float32)
# accel_mean = np.array([-0.8964,0.0151,-8.6236],dtype=np.float32)
# accel_dev = np.array([4.408,3.220,1.858],dtype=np.float32)
# gyro_mean = np.array([-1.932e-05,-7.916e-04,2.469e-03],dtype=np.float32)
# gyro_dev = np.array([0.1316,0.1368,0.1851],dtype=np.float32)
# angles_mean = np.array([-0.0547,0.0564,0.0160],dtype=np.float32)
# angles_dev = np.array([0.3591,0.3419,1.6780],dtype=np.float32)
# angles_6dof_mean = np.array([ 0.10997726,  0.05451011, -0.05865304, -0.05628474,  0.10971725, -0.05418512],dtype=np.float32)
# angles_6dof_dev = np.array([0.63793228, 0.68623939, 0.32188721, 0.68500043, 0.63749058, 0.32594217],dtype=np.float32)

# my new statistics
# vel_mean = np.array([-0.00774187, -0.00534718, -0.00123131],dtype=np.float32)
# vel_dev = np.array([1.3457327 , 1.3775525 , 0.37740588],dtype=np.float32)
# accel_mean = np.array([-1.0022876 ,  0.05453655, -8.642994],dtype=np.float32)
# accel_dev = np.array([4.345426 , 3.136036 , 1.7666128],dtype=np.float32)
# gyro_mean = np.array([-0.00010223, -0.00108385,  0.00119425],dtype=np.float32)
# gyro_dev = np.array([0.12500995, 0.13041076, 0.17545371],dtype=np.float32)
# angles_mean = np.array([-0.05790313,  0.06029133,  0.02655479],dtype=np.float32)
# angles_dev = np.array([0.3595147 , 0.33688033, 1.6698864],dtype=np.float32)
# angles_6dof_mean = np.array([ 0.10997726,  0.05451011, -0.05865304, -0.05628474,  0.10971725, -0.05418512],dtype=np.float32)
# angles_6dof_dev = np.array([0.63793228, 0.68623939, 0.32188721, 0.68500043, 0.63749058, 0.32594217],dtype=np.float32)

vel_mean = np.array([0., 0., 0.],dtype=np.float32)
vel_dev = np.array([1., 1., 1.],dtype=np.float32)
accel_mean = np.array([0.0, 0., -8.],dtype=np.float32)
accel_dev = np.array([1., 1., 1.],dtype=np.float32)
gyro_mean = np.array([0., 0., 0.],dtype=np.float32)
gyro_dev = np.array([0.13,0.13,0.13],dtype=np.float32)
angles_mean = np.array([0., 0., 0.],dtype=np.float32)
angles_dev = np.array([0.4,0.4,1.7],dtype=np.float32)
angles_6dof_mean = np.array([0., 0., 0., 0., 0., 0.],dtype=np.float32)
angles_6dof_dev = np.array([0.7, 0.7, 0.3, 0.7, 0.7, 0.3],dtype=np.float32)
accel_nograv_mean = np.array([0., 0., 0.],dtype=np.float32)
accel_nograv_dev = np.array([1., 1., 1.],dtype=np.float32)

accel_realsense_official = {
    'b'         : np.array([0.6859999999999999, 0.6859999999999999, 0.6859999999999999]),
    'b_drift'   : np.array([0.000294, 0.000294, 0.000294]),
    'b_corr'    : np.array([100., 100., 100.]),
    # deviation = [0.00868665, 0.00976136, 0.01078207]
    'vrw'       : np.array([0.020788939366884498, 0.020788939366884498, 0.020788939366884498]) * 60 / 60 * np.sqrt(1./200),
    # https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi055-ds000.pdf
    # g is the 9.8 m / s^2
    # temperature drift: +-1 mg/1K
    # bias: +-70 mg (bias < 0.6859999999999999 m/s^2)
    # noise density: 150 ug/sqrt(Hz) ( deviation = 0.020788939366884498 / 2 )
    # bias drift: 30ug (drift < 0.000294 m/s^2)
}

gyro_realsense_official = {
    'b'         : np.array([0.017453292519943295, 0.017453292519943295, 0.017453292519943295]),
    'b_drift'   : np.array([9.69627362219072e-06, 9.69627362219072e-06, 9.69627362219072e-06,]),
    'b_corr'    : np.array([100., 100., 100.,]),
    # deviation = [0.00222886, 0.001893, 0.00154683]
    'arw'       : np.array([0.003455575618567618, 0.003455575618567618, 0.003455575618567618]) * 60 / 60 * np.sqrt(1./200),
    # https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi055-ds000.pdf
    # deg is pi/180
    # temperature drift: +- 0.015 deg/s/K
    # bias: +- 1 deg/s (bias < 0.017453292519943295 rad / s)
    # noise density: 0.014 deg / s /sqrt(Hz) ( deviation = 0.003455575618567618 / 2 )
    # output noise: 0.1 deg / s ( deviation = 0.0017453292519943296 )
    # bias drift: 2 deg / hour ( drift < 9.69627362219072e-06 rad / sec )
}

class IMU:
    def __init__(self, 
            accel_err, 
            gyro_err, 
            fs = 100, 
            accel_rand_b = np.random.randn(3), 
            gyro_rand_b = np.random.randn(3) 
        ):
        """
        This class is to record the information of an imu. 
        ```python
        IMU(accel_err = accel_mid, gyro_err = gyro_mid, fs = 100)
        ```
        """
        self.accel_err = accel_err.copy()
        self.gyro_err  = gyro_err.copy()
        self.accel_err["b"] = accel_rand_b* accel_err["b"]
        self.gyro_err["b"] = gyro_rand_b* gyro_err["b"]
        self.fs = fs

    def get_accel_err(self):
        """
        
        """
        return self.accel_err

    def get_gyro_err(self):
        """
        
        """
        return self.gyro_err

    def get_fs(self):
        """

        """
        return self.fs

def get_bias_drift(bias_corr, bias_drift, n, fs):
    """

    """
    m = bias_corr.shape[0]
    SingleDrift = np.random.randn(n,m)
    a = 1. - 1./fs/bias_corr
    b = 1./fs*bias_drift
    a_pw = np.power(np.expand_dims(a,0),np.expand_dims(np.arange(n),1))
    temp = (b * SingleDrift / a_pw)
    temp2 = np.cumsum(temp,axis = 0)
    temp3 = np.zeros_like(temp2)
    temp3[1:] = temp2[:-1]
    ans = temp3 * a_pw
    return ans

def accel_noise(imu_info, n = 100, Vibration = None):
    """
    ```python
    accel_noise(imu_info = IMU(), n = 100)
    ```
    Notice
    ------
    This is to generate the Noise. \n
    You can select some noise randomly after using this function.  
    """
    accel_err = imu_info.get_accel_err()
    bias = np.array(accel_err["b"])
    bias_drift = np.array(accel_err["b_drift"])
    bias_corr = np.array(accel_err["b_corr"])
    vrw = np.array(accel_err["vrw"])
    fs = imu_info.fs
    dt = np.float64(1)/fs

    m = bias.shape[0]

    if (Vibration is None):
        Vibration = np.zeros((n,m))
    
    Gauss = np.random.randn(n,m)
    Noise_bias = np.repeat( np.expand_dims(bias, 0) ,n ,0)
    Noise_vrw = vrw / np.sqrt(dt) * Gauss
    Noise_bias_drift = get_bias_drift(bias_corr, bias_drift, n , fs)

    return [
        Noise_bias,
        Noise_bias_drift, 
        Noise_vrw, 
        Vibration,
    ]

def gyro_noise(imu_info, n = 100):
    """
    Notice
    ------
    see `noise.accel_noise`
    """
    gyro_err = imu_info.get_gyro_err()
    bias = gyro_err["b"]
    bias_drift = gyro_err["b_drift"]
    bias_corr = gyro_err["b_corr"]
    arw  = gyro_err["arw"]
    fs = imu_info.get_fs()
    dt = np.float64(1.)/fs

    m = bias.shape[0]

    Gauss = np.random.randn(n,m)
    Noise_bias = np.repeat( np.expand_dims(bias, 0) ,n ,0)
    Noise_arw = arw / np.sqrt(dt) * Gauss
    Noise_bias_drift = get_bias_drift(bias_corr, bias_drift, n, fs)
    
    return [
        Noise_bias,
        Noise_bias_drift,
        Noise_arw,
    ]

def add_realsense_noise(accel, gyro, scale, fs=100):
    accel_rand_b = np.random.randn(3)
    gyro_rand_b = np.random.randn(3)
    # print(accel_rand_b, gyro_rand_b)
    imu_info = IMU(
        accel_err = accel_realsense_official, gyro_err = gyro_realsense_official, fs=fs,
        accel_rand_b = accel_rand_b, gyro_rand_b = gyro_rand_b
    )
    accelnoise = accel_noise(imu_info = imu_info, n = accel.shape[0])
    gyronoise = gyro_noise(imu_info = imu_info, n = gyro.shape[0])
    accel  = accel + ((accelnoise[0] + accelnoise[1] + accelnoise[2] + accelnoise[3])*scale).astype(np.float32)
    # print('acc bias noise',accelnoise[0][0])
    gyro = gyro + ((gyronoise[0] + gyronoise[1] + gyronoise[2])*scale).astype(np.float32)
    return accel, gyro

if __name__ == '__main__':
    gyro = np.zeros((1000, 3))
    accel = np.zeros((1000, 3))
    accel_noise, gyro_noise = add_realsense_noise(accel, gyro, scale=1.0)
    import ipdb;ipdb.set_trace()