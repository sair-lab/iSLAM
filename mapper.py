import torch
import numpy as np
import pypose as pp

import matplotlib.pyplot as plt

import pandas as pd
from pyntcloud import PyntCloud


class Mapper:
    def __init__(self):
        self.frame_points = []

    def add_frame(self, image, depth, pose, fx, fy, cx, cy, mask=None, sample_rate=0.1):
        height, width = depth.shape

        # build UV map
        u_lin = torch.linspace(0, width-1, width)
        v_lin = torch.linspace(0, height-1, height)
        u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
        uv = torch.stack([u, v])
        uv1 = torch.stack([u, v, torch.ones_like(u)])

        if isinstance(pose, pp.LieTensor):
            T = pose
        else:
            T = pp.SE3(pose)
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

        # print('mapper z', torch.min(z), torch.max(z))

        points_local = z.unsqueeze(-1) * (K_inv.unsqueeze(0) @ uv1.unsqueeze(-1)).squeeze()

        # print('mapper lp', torch.max(torch.linalg.norm(points_local, dim=1)))

        points_world = T.unsqueeze(0) @ points_local.view(-1, 3)

        # print('mapper wp', torch.max(torch.linalg.norm(points_world, dim=1)))

        points = torch.cat([points_world, colors.view(-1, 3)], dim=1)
        print(points.shape)

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

    def save_data(self, fname):
        points = self.generate_cloud()
        np.savetxt(fname, points)


def normalize_3d(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    lim_len = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) * 0.5
    xlim = (xlim[0]+xlim[1])/2 - lim_len/2, (xlim[0]+xlim[1])/2 + lim_len/2
    ylim = (ylim[0]+ylim[1])/2 - lim_len/2, (ylim[0]+ylim[1])/2 + lim_len/2
    zlim = (zlim[0]+zlim[1])/2 - lim_len/2, (zlim[0]+zlim[1])/2 + lim_len/2
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


if __name__ == '__main__':
    image = torch.rand((3, 10, 15))
    depth = torch.ones((10, 15))
    pose = pp.SE3([-0.3478,  0.1789,  0.5823, -0.6996, -0.0011,  0.1420,  0.7003])
    print('pose', pose)
    fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0
    mask = torch.zeros((10, 15), dtype=torch.bool)
    for i in range(10):
        for j in range(15):
            mask[i, j] = (i+j)%2

    mapper = Mapper()
    mapper.add_frame(image, depth, pose, fx, fy, cx, cy, mask)

    mapper.write_ply('cloud.ply')

    cloud = mapper.generate_cloud()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.scatter3D(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 3:])
    plt.savefig('cloud.png')