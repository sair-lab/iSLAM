import os
import torch
import warnings
import argparse
import numpy as np
import pypose as pp
from torch import nn
from pgo_dataset import G2OPGO, VOPGO, PVGO_Dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseVelGraph(nn.Module):
    def __init__(self, nodes, vels, device, loss_weight, stop_frames=[]):
        super().__init__()
        assert nodes.size(0) == vels.size(0)
        self.nodes = pp.Parameter(nodes)
        self.vels = torch.nn.Parameter(vels)
        # self.para_nodes = pp.Parameter(nodes[1:])
        # self.para_vels = torch.nn.Parameter(vels[1:])
        # self.node0 = nodes[0].cpu()
        # self.vel0 = vels[0].cpu()
        # self.nodes = torch.cat((self.node0.view(1, -1), self.para_nodes), dim=0)
        # self.vels = torch.cat((self.vel0.view(1, -1), self.para_vels), dim=0)
        self.device = device

        assert len(loss_weight) == 4
        # loss weight hyper para
        self.l1 = loss_weight[0]
        self.l2 = loss_weight[1]
        self.l3 = loss_weight[2]
        self.l4 = loss_weight[3]

        self.stop_frames = stop_frames


    # def nodes(self):
    #     return torch.cat((self.node0.to(self.device).view(1, -1), self.para_nodes), dim=0)
    
    # def vels(self):
    #     return torch.cat((self.vel0.to(self.device).view(1, -1), self.para_vels), dim=0)


    def forward(self, edges, poses, imu_drots, imu_dtrans, imu_dvels, dts):
        nodes = self.nodes
        vels = self.vels
        
        # E = edges.size(0)
        # M = nodes.size(0) - 1
        # assert E == poses.size(0)
        # assert M == imu_drots.size(0) == imu_dtrans.size(0) == imu_dvels.size(0)
        
        # pose graph constraint
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        pgerr = error.Log().tensor()

        # adj vel constraint
        # error = imu_dvels - torch.diff(vels, dim=0)
        # adjvelerr = torch.cat((error, torch.zeros_like(error)), dim=1)
        adjvelerr = imu_dvels - torch.diff(vels, dim=0)

        # imu rot constraint
        node1 = nodes.rotation()[ :-1]
        node2 = nodes.rotation()[1:  ]
        error = imu_drots.Inv() @ node1.Inv() @ node2
        # error = error.Log().tensor()
        # imuroterr = torch.cat((torch.zeros_like(error), error), dim=1)
        imuroterr = error.Log().tensor()

        # trans vel constraint
        # error = torch.diff(nodes.translation(), dim=0) - (vels[:-1] * dts + imu_dtrans)
        # transvelerr = torch.cat((error / dts, torch.zeros_like(error)), dim=1)
        transvelerr = torch.diff(nodes.translation(), dim=0) - (vels[:-1] * dts + imu_dtrans)

        # test_run
        # return torch.cat((  self.l1 * pgerr, 
        #                     self.l2 * adjvelerr, 
        #                     self.l3 * imuroterr, 
        #                     self.l4 * transvelerr  ), dim=0)
        return torch.cat((  self.l1 * pgerr.view(-1), 
                            self.l2 * adjvelerr.view(-1), 
                            self.l3 * imuroterr.view(-1), 
                            self.l4 * transvelerr.view(-1)  ), dim=0)


    def vo_loss(self, edges, poses):
        nodes = self.nodes
        vels = self.vels

        node1 = nodes[edges[:, 0]].detach()
        node2 = nodes[edges[:, 1]].detach()
        error = poses.Inv() @ node1.Inv() @ node2
        error = error.Log().tensor()
        trans_loss = torch.sum(error[:, :3]**2, dim=1)
        rot_loss = torch.sum(error[:, 3:]**2, dim=1)

        return trans_loss, rot_loss

    
    def align_to(self, target, idx=0):
        source = self.nodes[idx].detach()
        vels = target.rotation() @ source.rotation().Inv() @ self.vels
        nodes = target @ source.Inv() @ self.nodes
        return nodes, vels



def run_pvgo(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                device='cuda:0', init_with_imu_rot=True, init_with_imu_vel=True,
                radius=1e4, loss_weight=(1,1,1,1), stop_frames=[]):

    # print(imu_init)

    data = PVGO_Dataset(poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                        device, init_with_imu_rot, init_with_imu_vel)
    nodes, vels = data.nodes, data.vels
    edges, poses = data.edges, data.poses
    imu_drots, imu_dtrans, imu_dvels = data.imu_drots, data.imu_dtrans, data.imu_dvels
    dts = data.dts
    node0 = nodes[0].clone()

    graph = PoseVelGraph(nodes, vels, device, loss_weight, stop_frames).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=False)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

    ### the 1st implementation: for customization and easy to extend
    while scheduler.continual:
        # TODO: weights
        loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts))
        scheduler.step(loss)

    ### The 2nd implementation: equivalent to the 1st one, but more compact
    # scheduler.optimize(input=(edges, poses), weight=infos)

    trans_loss, rot_loss = graph.vo_loss(edges, data.poses_withgrad)

    nodes, vels = graph.align_to(node0)
    nodes = nodes.detach().cpu()
    vels = vels.detach().cpu()
    # print('test in run pvgo:', node0, nodes[0])
    
    edges = edges.cpu()
    motions = nodes[edges[:, 0]].Inv() @ nodes[edges[:, 1]]

    return trans_loss, rot_loss, nodes.numpy(), vels.numpy(), motions.numpy()
