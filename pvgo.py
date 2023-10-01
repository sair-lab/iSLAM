import time
import numpy as np

import torch
from torch import nn

from dataset_statistics import kitti_imu_func, kitti_vo_func

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseVelGraph(nn.Module):
    def __init__(self, nodes, vels):
        super().__init__()

        assert nodes.size(0) == vels.size(0)
        self.nodes = pp.Parameter(nodes.clone())
        self.vels = torch.nn.Parameter(vels.clone())


    def forward(self, edges, poses, imu_drots, imu_dtrans, imu_dvels, dts):
        nodes = self.nodes
        vels = self.vels
        
        # E = edges.size(0)
        # M = nodes.size(0) - 1
        # assert E == poses.size(0)
        # assert M == imu_drots.size(0) == imu_dtrans.size(0) == imu_dvels.size(0)
        
        # VO constraint
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        pgerr = error.Log().tensor()

        # delta velocity constraint
        adjvelerr = imu_dvels - torch.diff(vels, dim=0)

        # imu rotation constraint
        node1 = nodes.rotation()[ :-1]
        node2 = nodes.rotation()[1:  ]
        error = imu_drots.Inv() @ node1.Inv() @ node2
        imuroterr = error.Log().tensor()

        # translation-velocity cross constraint
        transvelerr = torch.diff(nodes.translation(), dim=0) - (vels[:-1] * dts + imu_dtrans)

        return pgerr, adjvelerr, imuroterr, transvelerr
        # return torch.cat([
        #     1 * pgerr.view(-1),
        #     0.1 * adjvelerr.view(-1),
        #     10 * imuroterr.view(-1),
        #     0.1 * transvelerr.view(-1)
        # ])


    def vo_loss(self, edges, poses):
        nodes = self.nodes

        node1 = nodes[edges[:, 0]].detach()
        node2 = nodes[edges[:, 1]].detach()
        error = poses.Inv() @ node1.Inv() @ node2
        error = error.Log().tensor()

        trans_loss = torch.sum(error[:, :3]**2, dim=1)
        rot_loss = torch.sum(error[:, 3:]**2, dim=1)

        return trans_loss, rot_loss


    def vo_loss_unroll(self, edges, poses):
        nodes = self.nodes

        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        error = error.Log().tensor()

        trans_loss = torch.sum(error[:, :3]**2, dim=1)
        rot_loss = torch.sum(error[:, 3:]**2, dim=1)

        return trans_loss, rot_loss

    
    def align_to(self, target, idx=0):
        # align nodes[idx] to target
        source = self.nodes[idx].detach()
        vels = target.rotation() @ source.rotation().Inv() @ self.vels
        nodes = target @ source.Inv() @ self.nodes
        return nodes, vels


def run_pvgo(init_nodes, init_vels, vo_motions, links, dts, imu_drots, imu_dtrans, imu_dvels, 
                device='cuda:0', radius=1e4, loss_weight=(1,1,1,1)):

    # information matrix
    vo_motions_cpu = vo_motions.detach().cpu()
    vo_rot_norms = torch.norm(vo_motions_cpu.rotation().Log(), dim=1).numpy()
    vo_trans_norms = torch.norm(vo_motions_cpu.translation(), dim=1).numpy()
    imu_rot_norms = torch.norm(imu_drots.Log(), dim=1).numpy()
    imu_trans_norms = torch.norm(imu_dvels * dts.unsqueeze(-1), dim=1).numpy()

    vo_rot_covs, vo_trans_covs = kitti_vo_func(vo_rot_norms, vo_trans_norms)
    imu_rot_covs, imu_trans_covs = kitti_imu_func(imu_rot_norms, imu_trans_norms)
    imu_vel_covs = imu_trans_covs / dts.numpy()**2

    # cov
    # vo_rot_infos = (1 / vo_rot_covs)
    # vo_trans_infos = (1 / vo_trans_covs)
    # imu_rot_infos = (1 / imu_rot_covs)
    # imu_vel_infos = (1 / imu_vel_covs)
    # transvel_infos = np.ones(len(init_nodes)-1)
    # transvel_infos = imu_vel_infos

    # scov
    # vo_rot_infos = (1 / vo_rot_covs) / np.mean(1 / vo_rot_covs) * loss_weight[0]**2
    # vo_trans_infos = (1 / vo_trans_covs) / np.mean(1 / vo_trans_covs) * loss_weight[0]**2
    # imu_rot_infos = (1 / imu_rot_covs) / np.mean(1 / imu_rot_covs) * loss_weight[2]**2
    # imu_vel_infos = (1 / imu_vel_covs) / np.mean(1 / imu_vel_covs) * loss_weight[1]**2
    # transvel_infos = np.ones(len(init_nodes)-1) * loss_weight[3]**2

    # seye
    vo_rot_infos = np.ones_like(vo_rot_covs) * loss_weight[0]**2
    vo_trans_infos = np.ones_like(vo_trans_covs) * loss_weight[0]**2
    imu_rot_infos = np.ones_like(imu_rot_covs) * loss_weight[2]**2
    imu_vel_infos = np.ones_like(imu_vel_covs) * loss_weight[1]**2
    transvel_infos = np.ones(len(init_nodes)-1) * loss_weight[3]**2

    vo_info_mats       = [torch.diag(torch.tensor([vo_trans_infos[i]]*3 + [vo_rot_infos[i]]*3))
                          for i in range(len(vo_trans_infos))]
    imu_rot_info_mats  = [torch.diag(torch.tensor([imu_rot_infos[i]]*3))
                          for i in range(len(imu_rot_infos))]
    imu_vel_info_mats  = [torch.diag(torch.tensor([imu_vel_infos[i]]*3))
                          for i in range(len(imu_vel_infos))]
    transvel_info_mats = [torch.diag(torch.tensor([transvel_infos[i]*3]))
                          for i in range(len(transvel_infos))]
        
    # init inputs
    edges = links.to(device)
    poses = vo_motions.detach().to(device)
    imu_drots = imu_drots.to(device)
    imu_dtrans = imu_dtrans.to(device)
    imu_dvels = imu_dvels.to(device)
    dts = dts.unsqueeze(-1).to(device)
    vo_info_mats = torch.stack(vo_info_mats).to(torch.float32).to(device)
    imu_rot_info_mats = torch.stack(imu_rot_info_mats).to(torch.float32).to(device)
    imu_vel_info_mats = torch.stack(imu_vel_info_mats).to(torch.float32).to(device)
    transvel_info_mats = torch.stack(transvel_info_mats).to(torch.float32).to(device)

    # build graph and optimizer
    graph = PoseVelGraph(init_nodes, init_vels).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    # start_time = time.time()

    # optimization loop
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts), 
                              weight=(vo_info_mats, imu_vel_info_mats, imu_rot_info_mats, transvel_info_mats))
        # loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts))
        scheduler.step(loss)

    # end_time = time.time()
    # print('pgo time:', end_time - start_time)

    # get loss for backpropagate
    trans_loss, rot_loss = graph.vo_loss(edges, vo_motions)

    # for test
    # trans_loss, rot_loss = graph.vo_loss_unroll(edges, data.poses_withgrad)

    # align nodes to the original first pose
    nodes, vels = graph.align_to(init_nodes[0].to(device))
    nodes = nodes.detach().cpu()
    vels = vels.detach().cpu()

    covs = {'vo_rot':vo_rot_infos, 'imu_rot':imu_rot_infos,
            'vo_trans':vo_trans_infos, 'imu_vel':imu_vel_infos,
            'transvel':transvel_infos}

    return trans_loss, rot_loss, nodes, vels, covs
