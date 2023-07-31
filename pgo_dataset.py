import os
import torch
import numpy as np
import pypose as pp


class PVGO_Dataset:
    def __init__(self, poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                    device='cuda:0', init_with_imu_rot=True, init_with_imu_vel=True):

        N = poses_np.shape[0]
        M = len(links)
        self.dtype = torch.get_default_dtype()

        self.ids = torch.arange(0, N, dtype=torch.int64).view(-1, 1)
        self.edges = torch.tensor(links, dtype=torch.int64).to(device)
        self.poses = pp.SE3(motions.detach()).to(self.dtype).to(device)
        self.poses_withgrad = pp.SE3(motions).to(self.dtype).to(device)
        self.infos = torch.stack([torch.eye(6)]*M).to(self.dtype).to(device)    # No use yet

        self.imu_drots = pp.SO3(imu_drots_np).to(self.dtype).to(device)
        self.imu_dtrans = torch.from_numpy(imu_dtrans_np).to(self.dtype).to(device)
        self.imu_dvels = torch.from_numpy(imu_dvels_np).to(self.dtype).to(device)
        self.dts = torch.tensor(dts).view(-1, 1).to(self.dtype).to(device)
        
        if init_with_imu_rot:
            rot = pp.SO3(poses_np[0, 3:])
            rots = [rot]
            for drot in imu_drots_np:
                rot = rot @ pp.SO3(drot)
                rots.append(rot)
            rots = torch.stack(rots)
            trans = torch.from_numpy(poses_np[:, :3])
            assert N == rots.size(0) == trans.size(0)
            self.nodes = pp.SE3(torch.cat([trans, rots.tensor()], dim=1)).to(self.dtype).to(device)
        else:
            self.nodes = pp.SE3(poses_np).to(self.dtype).to(device)

        if init_with_imu_vel:
            vels_np = np.cumsum(np.concatenate([imu_init['vel'].reshape(1, -1), imu_dvels_np], axis=0), axis=0)
            self.vels = torch.from_numpy(vels_np).to(self.dtype).to(device)
        else:
            vels_ = torch.diff(self.nodes.translation(), dim=0) / self.dts
            self.vel0 = torch.from_numpy(imu_init['vel']).to(self.dtype).to(device)
            self.vels = torch.cat((self.vel0.view(1, -1), vels_), dim=0)
            
        assert N == self.ids.size(0) == self.nodes.size(0) == self.vels.size(0)
        assert M == self.edges.size(0) == self.poses.size(0)
        assert N-1 == self.imu_drots.size(0) == self.imu_dtrans.size(0) == self.imu_dvels.size(0) == self.dts.size(0)
