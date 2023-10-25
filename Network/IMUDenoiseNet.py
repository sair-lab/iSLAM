import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pypose as pp


class IMUCorrector_CNN_GRU_WO_COV(nn.Module):
    def __init__(self, in_channel=6, out_channel=64, hidden_size=128, kernel_size=10, num_layers=1):
        super(IMUCorrector_CNN_GRU_WO_COV, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=10)
        self.gelu = nn.GELU()
        self.gru = nn.GRU(out_channel, hidden_size, num_layers, batch_first=True)
        self.encoder = nn.Sequential(self.conv1, nn.GELU(), self.gru)

        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 6),
            nn.GELU()
        )

    def forward(self, data, eval=True):
        self.train() if not eval else self.eval()
        with torch.set_grad_enabled(not eval):

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

            partten = torch.ones(pose_output.shape[1], dtype=int) * 10
            partten[-1] = acc.shape[1] - 10*pose_output.shape[1] + 10
            partten = partten.to(acc.device)
            pose_output = torch.repeat_interleave(pose_output, partten, dim=1)

            corrected_acc = pose_output[..., 0:3] + acc
            corrected_gyro = pose_output[..., 3:6] + gyro


            corrected_acc = corrected_acc.squeeze(0)
            corrected_gyro = corrected_gyro.squeeze(0)

            # print(corrected_acc.shape, corrected_gyro.shape, acc_cov.shape, gyro_cov.shape)

            return corrected_acc, corrected_gyro, None, None
                
class IMUCorrector_CNN_GRU(nn.Module):
    def __init__(self, in_channel=6, out_channel=64, hidden_size=128, kernel_size=10, num_layers=1):
        super(IMUCorrector_CNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=10)
        self.gelu = nn.GELU()
        self.gru = nn.GRU(out_channel, hidden_size, num_layers, batch_first=True)
        self.encoder = nn.Sequential(self.conv1, nn.GELU(), self.gru)
        self.pose_decoder = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 6),
            nn.GELU()
        )
        self.imu = pp.module.IMUPreintegrator(reset=True, prop_cov=True)
    def forward(self, data, eval=True):
        self.train() if not eval else self.eval()
        with torch.set_grad_enabled(not eval):
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
            partten = torch.ones(pose_output.shape[1], dtype=int) * 10
            partten[-1] = acc.shape[1] - 10*pose_output.shape[1] + 10
            partten = partten.to(acc.device)
            pose_output = torch.repeat_interleave(pose_output, partten, dim=1)
            corrected_acc = pose_output[..., 0:3] + acc
            corrected_gyro = pose_output[..., 3:6] + gyro
            corrected_acc = corrected_acc.squeeze(0)
            corrected_gyro = corrected_gyro.squeeze(0)
            # print(corrected_acc.shape, corrected_gyro.shape, acc_cov.shape, gyro_cov.shape)
            if eval:
                return corrected_acc, corrected_gyro, None, None
            else:
                return self.imu(
                    init_state=data['init_state'], dt=data['dt'].unsqueeze(-1),
                    gyro=corrected_gyro, acc=corrected_acc,
                )
            