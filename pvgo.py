import time

import torch
from torch import nn
import torch.utils.data as Data

import numpy as np
import matplotlib.pyplot as plt

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

from pgo_dataset import PVGO_Dataset

 
class PoseVelGraph(nn.Module):
    def __init__(self, nodes, vels, device, loss_weight):
        super().__init__()
        self.device = device

        assert nodes.size(0) == vels.size(0)
        self.nodes = pp.Parameter(nodes)
        self.vels = torch.nn.Parameter(vels)

        assert len(loss_weight) == 4
        # loss weight hyper para
        self.l1 = loss_weight[0]
        self.l2 = loss_weight[1]
        self.l3 = loss_weight[2]
        self.l4 = loss_weight[3]


    def forward(self, edges, poses, imu_drots, imu_dtrans, imu_dvels, dts):
        nodes = self.nodes
        vels = self.vels
        
        E = edges.size(0)
        M = nodes.size(0) - 1
        assert E == poses.size(0)
        assert M == imu_drots.size(0) == imu_dtrans.size(0) == imu_dvels.size(0)
        
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

        return torch.cat((  self.l1 * pgerr.view(-1), 
                            self.l2 * adjvelerr.view(-1), 
                            self.l3 * imuroterr.view(-1), 
                            self.l4 * transvelerr.view(-1)  ), dim=0)


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


def run_pvgo(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                device='cuda:0', init_with_imu_rot=True, init_with_imu_vel=True, radius=1e4, loss_weight=(1,1,1,1)):

    # init inputs
    data = PVGO_Dataset(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                        device, init_with_imu_rot, init_with_imu_vel)
    nodes, vels = data.nodes, data.vels
    edges, poses = data.edges, data.poses
    imu_drots, imu_dtrans, imu_dvels = data.imu_drots, data.imu_dtrans, data.imu_dvels
    dts = data.dts
    node0 = nodes[0].clone()

    # build graph and optimizer
    graph = PoseVelGraph(nodes, vels, device, loss_weight).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    # start_time = time.time()

    # optimization loop
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts))
        scheduler.step(loss)

    # end_time = time.time()
    # print('pgo time:', end_time - start_time)

    # get loss for backpropagate
    trans_loss, rot_loss = graph.vo_loss(edges, data.poses_withgrad)

    # for test
    # trans_loss, rot_loss = graph.vo_loss_unroll(edges, data.poses_withgrad)

    # align nodes to the original first pose
    nodes, vels = graph.align_to(node0)
    nodes = nodes.detach().cpu()
    vels = vels.detach().cpu()
    
    # calc motions
    edges = edges.cpu()
    motions = nodes[edges[:, 0]].Inv() @ nodes[edges[:, 1]]

    return trans_loss, rot_loss, nodes.numpy(), vels.numpy(), motions.numpy()
