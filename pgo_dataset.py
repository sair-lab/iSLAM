import os,torch
import numpy as np
import pypose as pp
import torch.utils.data as Data


class G2OPGO(Data.Dataset):
    def __init__(self, root, dataname, device='cpu'):
        super().__init__()

        def info2mat(info):
            mat = np.zeros((6,6))
            ix = 0
            for i in range(mat.shape[0]):
                mat[i,i:] = info[ix:ix+(6-i)]
                mat[i:,i] = info[ix:ix+(6-i)]
                ix += (6-i)
            return mat
        self.dtype = torch.get_default_dtype()
        filename = os.path.join(root, dataname)
        ids, nodes, edges, poses, infos = [], [], [], [], []
        with open(filename) as f:
            for line in f:
                line = line.split()
                if line[0] == 'VERTEX_SE3:QUAT':
                    ids.append(torch.tensor(int(line[1]), dtype=torch.int64))
                    nodes.append(pp.SE3(np.array(line[2:], dtype=np.float64)))
                elif line[0] == 'EDGE_SE3:QUAT':
                    edges.append(torch.tensor(np.array(line[1:3], dtype=np.int64)))
                    poses.append(pp.SE3(np.array(line[3:10], dtype=np.float64)))
                    infos.append(torch.tensor(info2mat(np.array(line[10:], dtype=np.float64))))

        self.ids = torch.stack(ids)
        self.nodes = torch.stack(nodes).to(self.dtype).to(device)
        self.edges = torch.stack(edges).to(device) # have to be LongTensor
        self.poses = torch.stack(poses).to(self.dtype).to(device)
        self.infos = torch.stack(infos).to(self.dtype).to(device)
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)


class VOPGO(Data.Dataset):
    def __init__(self, poses_np, motions, links, infomats=None, device='cpu'):
        super().__init__()

        N = poses_np.shape[0]
        M = len(links)
        self.dtype = torch.get_default_dtype()

        self.ids = torch.arange(0, N, dtype=torch.int64).view(-1, 1)
        self.nodes = pp.SE3(poses_np).to(self.dtype).to(device)
        self.edges = torch.tensor(links, dtype=torch.int64).to(device)
        self.poses = pp.SE3(motions.detach()).to(self.dtype).to(device)
        self.poses_withgrad = pp.SE3(motions).to(self.dtype).to(device)
        if infomats is not None:
            raise NotImplementedError
        else:
            self.infos = torch.stack([torch.eye(6)]*M).to(self.dtype).to(device)
            
        assert self.ids.size(0) == self.nodes.size(0) \
               and self.edges.size(0) == self.poses.size(0) == self.infos.size(0)

    def init_value(self):
        return self.nodes.clone()

    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)


class PVGO_Dataset(Data.Dataset):
    def __init__(self, poses_np, motions, links, imu_drots_np, imu_dtrans_np, imu_dvels_np, imu_init, dts, 
                    device='cpu', init_with_imu_rot=True, init_with_imu_vel=True):
        
        super().__init__()

        N = poses_np.shape[0]
        M = len(links)
        self.dtype = torch.get_default_dtype()

        self.ids = torch.arange(0, N, dtype=torch.int64).view(-1, 1)
        self.edges = torch.tensor(links, dtype=torch.int64).to(device)
        self.poses = pp.SE3(motions.detach()).to(self.dtype).to(device)
        self.poses_withgrad = pp.SE3(motions).to(self.dtype).to(device)

        # No use
        self.infos = torch.stack([torch.eye(6)]*M).to(self.dtype).to(device)

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
            nodes_ = pp.SE3(torch.cat([trans, rots.tensor()], dim=1)).to(self.dtype).to(device)
        else:
            nodes_ = pp.SE3(poses_np).to(self.dtype).to(device)
        # self.nodes = self.align_nodes(imu_init['rot'], imu_init['pos'], 0, nodes_)
        # print('test in pvgo dataset:', imu_init, self.nodes[0])
        self.nodes = nodes_

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
    
    def __getitem__(self, i):
        return self.edges[i], self.poses[i], self.infos[i]

    def __len__(self):
        return self.edges.size(0)

    def align_nodes(self, rot, trans, idx, nodes):
        tar = pp.SE3(np.concatenate((trans, rot))).to(nodes.dtype).to(nodes.device)
        return tar @ nodes[idx].Inv() @ nodes

