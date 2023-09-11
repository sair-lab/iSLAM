import time

import torch
from torch import nn

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau

 
class PoseGraph(nn.Module):
    def __init__(self, nodes):
        super().__init__()

        self.nodes = pp.Parameter(nodes.clone())


    def forward(self, edges, poses):
        nodes = self.nodes
        
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        pgerr = error.Log().tensor()

        return pgerr

    
    def align_to(self, target, idx=0):
        # align nodes[idx] to target
        source = self.nodes[idx].detach()
        nodes = target @ source.Inv() @ self.nodes
        return nodes


def run_pgo(init_nodes, motions, links, device='cuda:0', radius=1e4):

    # init inputs
    edges = links.to(device)
    poses = motions.detach().to(device)

    # build graph and optimizer
    graph = PoseGraph(init_nodes).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    # start_time = time.time()

    # optimization loop
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses))
        scheduler.step(loss)

    # end_time = time.time()
    # print('pgo time:', end_time - start_time)

    # align nodes to the original first pose
    nodes = graph.align_to(init_nodes[0].to(device))
    nodes = nodes.detach().cpu()
    
    return nodes
