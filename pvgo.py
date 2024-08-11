import time
import numpy as np

import torch
from torch import nn

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseVelGraph(nn.Module):
    def __init__(self, nodes, vels, reproj=None):
        super().__init__()

        assert nodes.size(0) == vels.size(0)
        self.nodes = pp.Parameter(nodes.clone())
        self.vels = torch.nn.Parameter(vels.clone())

        self.reproj = reproj


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

        if self.reproj is not None:
            node1 = nodes[ :-1]
            node2 = nodes[1:  ]
            motion = node1.Inv() @ node2
            motion[0] = 0.1
            reprojerr = self.reproj(motion)
            if len(reprojerr.shape) == 3:
                reprojerr = reprojerr.view(-1, self.reproj.N*2)
            return pgerr, adjvelerr, imuroterr, transvelerr, reprojerr
        
        else:
            return pgerr, adjvelerr, imuroterr, transvelerr


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
    

    def imu_loss(self, imu_drots, imu_dvels):
        nodes = self.nodes
        vels = self.vels

        # delta velocity constraint
        adjvelerr = imu_dvels - torch.diff(vels, dim=0)

        # imu rotation constraint
        node1 = nodes.rotation()[ :-1]
        node2 = nodes.rotation()[1:  ]
        error = imu_drots.Inv() @ node1.Inv() @ node2
        imuroterr = error.Log().tensor()

        trans_loss = torch.sum(adjvelerr**2, dim=1)
        rot_loss = torch.sum(imuroterr**2, dim=1)

        return trans_loss, rot_loss

    
    def align_to(self, target, idx=0):
        # align nodes[idx] to target
        source = self.nodes[idx].detach()
        vels = target.rotation() @ source.rotation().Inv() @ self.vels
        nodes = target @ source.Inv() @ self.nodes
        return nodes, vels


def run_pvgo(init_nodes, init_vels, vo_motions, links, dts, imu_drots, imu_dtrans, imu_dvels, 
                device='cuda:0', radius=1e4, loss_weight=(1,1,1,1), reproj=None, target='vo'):
    
    vo_rot_infos = np.ones(len(links)) * loss_weight[0]**2
    vo_trans_infos = np.ones(len(links)) * loss_weight[0]**2
    imu_rot_infos = np.ones(len(init_nodes)-1) * loss_weight[2]**2
    imu_vel_infos = np.ones(len(init_nodes)-1) * loss_weight[1]**2
    transvel_infos = np.ones(len(init_nodes)-1) * loss_weight[3]**2
    if reproj is not None:
        reproj_infos = np.ones(len(init_nodes)-1) * (loss_weight[4]/reproj.N)**2

    vo_info_mats       = [torch.diag(torch.tensor([vo_trans_infos[i]]*3 + [vo_rot_infos[i]]*3))
                          for i in range(len(vo_trans_infos))]
    imu_rot_info_mats  = [torch.diag(torch.tensor([imu_rot_infos[i]]*3))
                          for i in range(len(imu_rot_infos))]
    imu_vel_info_mats  = [torch.diag(torch.tensor([imu_vel_infos[i]]*3))
                          for i in range(len(imu_vel_infos))]
    transvel_info_mats = [torch.diag(torch.tensor([transvel_infos[i]]*3))
                          for i in range(len(transvel_infos))]
    if reproj is not None:
        reproj_info_mats = [torch.diag(torch.tensor([reproj_infos[i]]*(reproj.N*2)))
                          for i in range(len(reproj_infos))]
    
    # init inputs
    edges = links.to(device)
    poses = vo_motions.detach().to(device)

    imu_drots_grad = imu_drots.to(device)
    imu_dvels_grad = imu_dvels.to(device)

    imu_drots = imu_drots.detach().to(device)
    imu_dtrans = imu_dtrans.detach().to(device)
    imu_dvels = imu_dvels.detach().to(device)
    dts = dts.unsqueeze(-1).to(device)

    vo_info_mats = torch.stack(vo_info_mats).to(torch.float32).to(device)
    imu_rot_info_mats = torch.stack(imu_rot_info_mats).to(torch.float32).to(device)
    imu_vel_info_mats = torch.stack(imu_vel_info_mats).to(torch.float32).to(device)
    transvel_info_mats = torch.stack(transvel_info_mats).to(torch.float32).to(device)

    weights = [vo_info_mats, imu_vel_info_mats, imu_rot_info_mats, transvel_info_mats]
    if reproj is not None:
        reproj_info_mats = torch.stack(reproj_info_mats).to(torch.float32).to(device)
        weights.append(reproj_info_mats)

    # build graph and optimizer
    graph = PoseVelGraph(init_nodes.detach(), init_vels.detach(), reproj).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    # start_time = time.time()

    # optimization loop
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts), weight=weights)
        # loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts))
        scheduler.step(loss)

    # end_time = time.time()
    # print('pgo time:', end_time - start_time)

    # get loss for backpropagate
    if target == 'vo':
        trans_loss, rot_loss = graph.vo_loss(edges, vo_motions)
    elif target == 'imu':
        trans_loss, rot_loss = graph.imu_loss(imu_drots_grad, imu_dvels_grad)
    else:
        trans_loss = rot_loss = None, None

    # for test
    # trans_loss, rot_loss = graph.vo_loss_unroll(edges, data.poses_withgrad)

    # align nodes to the original first pose
    nodes, vels = graph.align_to(init_nodes[0].to(device))
    nodes = nodes.detach().cpu()
    vels = vels.detach().cpu()

    covs = {'vo_rot':vo_rot_infos, 'imu_rot':imu_rot_infos,
            'vo_trans':vo_trans_infos, 'imu_vel':imu_vel_infos,
            'transvel':transvel_infos}
    if reproj is not None:
        covs['reproj'] = reproj_infos

    return trans_loss, rot_loss, nodes, vels, covs
