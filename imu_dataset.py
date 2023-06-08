import torch
import pykitti
import numpy as np
import pypose as pp
from datetime import datetime
import torch.utils.data as Data
from os import path


class KITTI_IMU(Data.Dataset):
    def __init__(self, root, dataname, drive, duration=10, step_size=1, mode='train'):
        super().__init__()
        self.duration = duration
        self.data = pykitti.raw(root, dataname, drive)
        self.seq_len = len(self.data.timestamps) - 1
        assert mode in ['evaluate', 'train', 'test'], "{} mode is not supported.".format(mode)

        self.dt = torch.tensor([datetime.timestamp(self.data.timestamps[i+1]) - datetime.timestamp(self.data.timestamps[i]) for i in range(self.seq_len)])
        self.gyro = torch.tensor([[self.data.oxts[i].packet.wx, self.data.oxts[i].packet.wy, self.data.oxts[i].packet.wz] for i in range(self.seq_len)])
        self.acc = torch.tensor([[self.data.oxts[i].packet.ax, self.data.oxts[i].packet.ay, self.data.oxts[i].packet.az] for i in range(self.seq_len)])
        self.gt_rot = pp.euler2SO3(torch.tensor([[self.data.oxts[i].packet.roll, self.data.oxts[i].packet.pitch, self.data.oxts[i].packet.yaw] for i in range(self.seq_len)]))
        self.gt_vel = self.gt_rot @ torch.tensor([[self.data.oxts[i].packet.vf, self.data.oxts[i].packet.vl, self.data.oxts[i].packet.vu] for i in range(self.seq_len)])
        self.gt_pos = torch.tensor(np.array([self.data.oxts[i].T_w_imu[0:3, 3] for i in range(self.seq_len)]))

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(0, end_frame - start_frame - self.duration, step_size)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1 : end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            'init_rot': self.gt_rot[frame_id : end_frame_id], # TODO: the init rotation might be used in gravity compensation 
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}


class TartanAir_IMU(Data.Dataset):
    def __init__(self, posefile, imudir, img_fps=10.0, imu_mul=10, duration=10, sample_step=1, mode='train'):
        super().__init__()
        self.duration = duration
        assert mode in ['evaluate', 'train', 'test', 'all'], "{} mode is not supported.".format(mode)

        #################### from TrajFolderDataset ############################################################

        poselist = np.loadtxt(posefile).astype(np.float32)
        assert(poselist.shape[1] == 7) # position + quaternion
        start_frame = 0
        end_frame = len(poselist)
        self.poses = poselist[start_frame:end_frame:sample_step]

        # acceleration in the body frame
        accels = np.load(path.join(imudir, "accel_left.npy")).astype(np.float32)
        # angular rate in the body frame
        gyros = np.load(path.join(imudir, "gyro_left.npy")).astype(np.float32)
        # velocity in the body frame
        vels = np.load(path.join(imudir,"vel_body.npy")).astype(np.float32)
        # velocity in the world frame
        vels_world = np.load(path.join(imudir,"vel_left.npy")).astype(np.float32)
        # # accel w/o gravity in body frame
        # accels_nograv = np.load(path.join(imudir, "accel_nograv_body.npy")).astype(np.float32)
        self.accels, self.gyros, self.vels, self.vels_world, self.accels_nograv = [], [], [], [], []
        for frame_idx in range(start_frame, end_frame-1, sample_step):
            self.accels.append(accels[frame_idx*imu_mul:(frame_idx+sample_step)*imu_mul])
            self.gyros.append(gyros[frame_idx*imu_mul:(frame_idx+sample_step)*imu_mul])
            self.vels.append(vels[frame_idx*imu_mul:(frame_idx+sample_step)*imu_mul])
            self.vels_world.append(vels_world[frame_idx*imu_mul:(frame_idx+sample_step)*imu_mul])
            # self.accels_nograv.append(accels_nograv[frame_idx*imu_mul:(frame_idx+1)*imu_mul])
        self.accels = np.stack(self.accels, axis=0)
        self.gyros = np.stack(self.gyros, axis=0)
        self.vels = np.stack(self.vels, axis=0)
        self.vels_world = np.stack(self.vels_world, axis=0)
        # self.accels_nograv = np.stack(self.accels_nograv, axis=0)
        assert(self.accels.shape == self.gyros.shape == self.vels.shape == self.vels_world.shape)
        print('Load {} of {} IMU frames in {}'.format(self.accels.shape[:2], len(accels), imudir))

        dt = 1.0/img_fps * sample_step / imu_mul
        self.imu_dts = np.full(self.accels.shape[:2], dt, dtype=np.float32)
        if self.poses is not None:
            init_pos = self.poses[0, :3]
            init_rot = self.poses[0, 3:]
            init_vel = self.vels_world[0, 0, :]
        else:
            init_pos = np.zeros(3, dtype=np.float32)
            init_rot = np.array([0, 0, 0, 1], dtype=np.float32)
            init_vel = np.zeros(3, dtype=np.float32)
        self.imu_init = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}
        # self.gravity = -9.81007
        self.gravity = -9.8

        ########################################################################################################

        self.dt = torch.tensor(self.imu_dts[:, 0]).squeeze()
        self.gyro = torch.tensor(self.gyros[:, 0, :]).squeeze()
        self.acc = torch.tensor(self.accels[:, 0, :]).squeeze()
        self.gt_rot = pp.SO3(self.poses[:, 3:])
        self.gt_vel = torch.tensor(self.vels_world[:, 0, :]).squeeze()
        self.gt_pos = torch.tensor(self.poses[:, :3])

        self.duration = duration
        self.seq_len = len(self.poses) - 1

        start_frame = 0
        end_frame = self.seq_len
        if mode == 'train':
            end_frame = np.floor(self.seq_len * 0.5).astype(int)
        elif mode == 'test':
            start_frame = np.floor(self.seq_len * 0.5).astype(int)

        self.index_map = [i for i in range(0, end_frame - start_frame - self.duration, sample_step)]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id = self.index_map[i]
        end_frame_id = frame_id + self.duration
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1 : end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            'init_rot': self.gt_rot[frame_id : end_frame_id], # TODO: the init rotation might be used in gravity compensation 
            'init_vel': self.gt_vel[frame_id][None, ...],
        }

    def get_init_value(self):
        return {'pos': self.gt_pos[:1],
                'rot': self.gt_rot[:1],
                'vel': self.gt_vel[:1]}

    def get_all_data(self):
        frame_id = self.index_map[0]
        end_frame_id = self.seq_len
        return {
            'dt': self.dt[frame_id: end_frame_id],
            'acc': self.acc[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gyro': self.gyro[frame_id: end_frame_id],
            'gt_pos': self.gt_pos[frame_id+1 : end_frame_id+1],
            'gt_rot': self.gt_rot[frame_id+1 : end_frame_id+1],
            'gt_vel': self.gt_vel[frame_id+1 : end_frame_id+1],
            'init_pos': self.gt_pos[frame_id][None, ...],
            'init_rot': self.gt_rot[frame_id : end_frame_id], # TODO: the init rotation might be used in gravity compensation 
            'init_vel': self.gt_vel[frame_id][None, ...],
        }


def imu_collate(data):
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])

    gt_pos = torch.stack([d['gt_pos'] for d in data])
    gt_rot = torch.stack([d['gt_rot'] for d in data])
    gt_vel = torch.stack([d['gt_vel'] for d in data])

    init_pos = torch.stack([d['init_pos'] for d in data])
    init_rot = torch.stack([d['init_rot'] for d in data])
    init_vel = torch.stack([d['init_vel'] for d in data])

    dt = torch.stack([d['dt'] for d in data]).unsqueeze(-1)
    
    return {
        'dt': dt,
        'acc': acc,
        'gyro': gyro,

        'gt_pos': gt_pos,
        'gt_vel': gt_vel,
        'gt_rot': gt_rot,

        'init_pos': init_pos,
        'init_vel': init_vel,
        'init_rot': init_rot,
    }


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to", obj)
