import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pypose as pp


class IMUCorrector_CNN_GRU(nn.Module):
    def __init__(self, in_channel=6, out_channel=64, hidden_size=128, kernel_size=10, num_layers=2):
        super(IMUCorrector_CNN_GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=10)
        self.gelu = nn.GELU()
        self.gru = nn.GRU(out_channel, hidden_size, num_layers, batch_first=True)
        self.encoder = nn.Sequential(self.conv1, nn.GELU(), self.gru)
        self.cov_decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 6),
            nn.GELU()
        )
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 6),
            nn.GELU()
        )

        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=True)

    def forward(self, data, eval=True):
        self.train() if not eval else self.eval()
        _ = torch.set_grad_enabled(not eval)

        acc = data['acc']
        gyro = data['gyro']

        if len(acc.shape) == 2:
            acc = acc.unsqueeze(0)
            gyro = gyro.unsqueeze(0)

        x = torch.cat([acc, gyro], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.gelu(x)
        x = x.permute(0, 2, 1)

        encoder_output, hidden = self.gru(x)
        pose_output = self.pose_decoder(encoder_output)
        cov_output = self.cov_decoder(encoder_output)

        pose_output = torch.repeat_interleave(pose_output, 10, dim=1)
        cov_output = torch.repeat_interleave(cov_output, 10, dim=1)

        corrected_acc = pose_output[..., 0:3] + acc
        corrected_gyro = pose_output[..., 3:6] + gyro

        acc_cov = torch.exp(cov_output[..., 0:3])
        gyro_cov = torch.exp(cov_output[..., 3:6])

        corrected_acc = corrected_acc.squeeze(0)
        corrected_gyro = corrected_gyro.squeeze(0)
        acc_cov = acc_cov.squeeze(0)
        gyro_cov = gyro_cov.squeeze(0)
        # print(corrected_acc.shape, corrected_gyro.shape, acc_cov.shape, gyro_cov.shape)

        if eval:
            return corrected_acc, corrected_gyro, acc_cov, gyro_cov
        else:
            return self.imu(
                init_state=data['init_state'], dt=data['dt'].unsqueeze(-1),
                gyro=corrected_gyro, acc=corrected_acc,
                acc_cov=acc_cov, gyro_cov=gyro_cov
            )


if __name__ == '__main__':
    import sys
    sys.path.insert(0, sys.path[0]+'/..')

    from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
    from Datasets.TrajFolderDataset import TrajFolderDataset

    data_root = '/projects/academic/cwx/kitti_raw/2011_09_30/2011_09_30_drive_0033_sync'
    data_type = 'kitti'
    start_frame = 0
    end_frame = -1
    trainroot = './train_results/test_imudenoise'
    vo_model_name = './models/stereo_cvt_tartanvo_1914.pkl'
    batch_size = 8

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([
        CropCenter((448, 640), fix_ratio=True), 
        DownscaleFlow(), 
        Normalize(mean=mean, std=std, keep_old=True), 
        ToTensor(),
        SqueezeBatchDim()
    ])
    dataset = TrajFolderDataset(
        datadir=data_root, datatype=data_type, transform=transform,
        start_frame=start_frame, end_frame=end_frame
    )

    imu_denoise_net = IMUCorrector_CNN_GRU().to(torch.double)

    pretrain = torch.load('./models/imu_denoise.pkl')
    imu_denoise_net.load_state_dict(pretrain)

    data = {}
    data['acc'] = torch.tensor(dataset.accels[:80])
    data['gyro'] = torch.tensor(dataset.gyros[:80])
    data['dt'] = torch.tensor(dataset.imu_dts[:80])
    init_state = {}
    for k in dataset.imu_init.keys():
        if k == 'rot':
            init_state[k] = pp.SO3(dataset.imu_init[k])
        else:
            init_state[k] = torch.tensor(dataset.imu_init[k])
    data['init_state'] = init_state

    states = imu_denoise_net.forward(data, eval=False)

    import matplotlib.pyplot as plt
    gt_pos = dataset.poses[:, :3]
    pos = state['pos'].numpy()
    plt.plot(gt_pos[:, 0], gt_pos[:, 1])
    plt.plot(pos[:, 0], pos[:, 1])
    plt.savefig('./temp/imu_denoise_traj.png')
