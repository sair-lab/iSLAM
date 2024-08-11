import torch
import numpy as np
import pypose as pp

import pandas as pd
from pyntcloud import PyntCloud


class Mapper:
    def __init__(self, rgb2imu_pose):
        self.frame_points = []
        self.rgb2imu_pose = rgb2imu_pose

    def add_frame(self, image, depth, pose, fx, fy, cx, cy, mask=None, sample_rate=0.1):
        height, width = depth.shape

        u_lin = torch.linspace(0, width-1, width)
        v_lin = torch.linspace(0, height-1, height)
        u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
        uv1 = torch.stack([u, v, torch.ones_like(u)])

        if isinstance(pose, pp.LieTensor):
            T = pose @ self.rgb2imu_pose
        else:
            T = pp.SE3(pose) @ self.rgb2imu_pose
        K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3)
        K_inv = torch.linalg.inv(K)

        if mask is None:
            mask = torch.ones(u.shape, dtype=torch.bool)
        if sample_rate < 1:
            mask = torch.logical_and(mask, torch.rand(u.shape) <= sample_rate)
        mask_3c = torch.stack([mask, mask, mask])

        z = depth[mask].view(-1)
        uv1 = uv1[mask_3c].view(3, -1).t()
        colors = image[mask_3c].view(3, -1).t()

        points_local = z.unsqueeze(-1) * (K_inv.unsqueeze(0) @ uv1.unsqueeze(-1)).squeeze()
        points_world = T.unsqueeze(0) @ points_local.view(-1, 3)
        points = torch.cat([points_world, colors.view(-1, 3)], dim=1)

        self.frame_points.append(points)

    def generate_cloud(self):
        return torch.cat(self.frame_points, dim=0).numpy()

    def write_ply(self, fname):
        points = self.generate_cloud()

        pos = points[:, :3]
        col = points[:, 3:]

        cloud = PyntCloud(pd.DataFrame(
            data=np.hstack((pos, col)),
            columns=["x", "y", "z", "blue", "green", "red"]
        ))
        cloud.to_file(fname)
