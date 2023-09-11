import torch
from torch.utils.data import Dataset, DataLoader

import pypose as pp
import numpy as np

from Datasets.TrajFolderDataset import TrajFolderDataset
from Datasets.transformation import cvtSE3_pypose, pose2motion_pypose, motion2pose_pypose

from pgo import run_pgo


def gt_pose_loop_detector(poses, min_loop, trans_th, rot_th):
    N = len(poses)
    poses = cvtSE3_pypose(poses)

    links = []
    for i in range(N-min_loop):
        motions = poses[i].Inv() @ poses[i+min_loop:]
        trans_mag = torch.norm(motions.translation(), dim=1)
        rot_mag = torch.norm(motions.rotation().Log(), dim=1)
        matches = torch.nonzero(torch.logical_and(trans_mag <= trans_th, rot_mag <= rot_th))
        matches += i + min_loop
        links.extend([[i, m] for m in matches])

    print(f'nlinks={len(links)}, rot_th={rot_th}, trans_th={trans_th}')

    return links


class LoopClosure:
    def __init__(self, dataset, batch_size, loop_edges_file=None):
        motions = cvtSE3_pypose(dataset.motions)
        rot_mag = torch.norm(motions.rotation().Log(), dim=1)
        trans_mag = torch.norm(motions.translation(), dim=1)

        if loop_edges_file is not None:
            self.loop_edges = np.loadtxt(loop_edges_file, dtype=int)
            self.loop_edges = torch.tensor(self.loop_edges)
        else:
            min_loop = batch_size * 10
            rot_th = torch.max(rot_mag)
            trans_th = torch.max(trans_mag) * 2.5
            self.loop_edges = gt_pose_loop_detector(dataset.poses, min_loop, trans_th, rot_th)
            self.loop_edges = torch.tensor(self.loop_edges)
        
        self.loop_dataset = TrajFolderDataset(
            datadir=dataset.datadir, datatype=dataset.datatype, transform=dataset.transform,
            start_frame=dataset.start_frame, end_frame=dataset.end_frame, loader=dataset.loader,
            links=self.loop_edges.tolist()
        )
        self.batch_size = batch_size

    def forward_vo_on_loops(self, tartanvo):
        loop_dataloader = DataLoader(self.loop_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        loop_motions_list = []
        for sample in loop_dataloader:
            loop_motions = tartanvo(sample)
            loop_motions_list.extend(loop_motions.detach().cpu())

        return pp.SE3(torch.stack(loop_motions_list))
    
    def perform(self, tartanvo, poses):
        keyframes = set([i for i in range(0, len(poses), self.batch_size)])
        keyframes = keyframes.union(set(self.loop_edges.view(-1).tolist()))
        keyframes = list(keyframes)
        keyframes.sort()
        keyframes = torch.tensor(keyframes)

        loop_edges_in_keyframes = torch.searchsorted(keyframes, self.loop_edges)

        links = [[i, i+1] for i in range(len(keyframes)-1)]
        links = torch.cat([torch.tensor(links), loop_edges_in_keyframes], dim=0)

        keyframe_poses = poses[keyframes]
        path_motions = pose2motion_pypose(keyframe_poses)
        loop_motions = self.forward_vo_on_loops(tartanvo)
        motions = torch.cat([path_motions, loop_motions], dim=0)

        loopclosure_poses = run_pgo(keyframe_poses, motions, links, 'cuda')

        return loopclosure_poses, keyframes, self.loop_edges, loop_motions.detach().cpu()


if __name__ == '__main__':
    from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
    from TartanVO import TartanVO

    data_root = '/home/data2/kitti_raw/2011_09_30/2011_09_30_drive_0033_sync'
    data_type = 'kitti'
    start_frame = 0
    end_frame = -1
    trainroot = './train_results/test_loopclosure'
    vo_model_name = './models/stereo_cvt_tartanvo_1914.pkl'
    batch_size = 8

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([
        CropCenter((448, 640), fix_ratio=True), 
        DownscaleFlow(), 
        Normalize(mean=mean, std=std, keep_old=True), 
        ToTensor(),
        SqueezeBatchDim()
    ])
    dataset = TrajFolderDataset(
        datadir=data_root, datatype=data_type, transform=transform,
        start_frame=start_frame, end_frame=end_frame
    )

    tartanvo = TartanVO(
        vo_model_name=vo_model_name, 
        correct_scale=False, fix_parts=('flow', 'stereo'), T_body_cam=dataset.rgb2imu_pose
    )

    loop_closure = LoopClosure(dataset, batch_size)

    motions = pose2motion_pypose(dataset.poses)
    error_rate = (torch.rand_like(motions.Log()) - 0.5) * 2
    motions = (motions.Log() * (1 + error_rate)).Exp()
    poses = motion2pose_pypose(motions, dataset.poses[0])
    loopclosure_poses, keyframes, loop_edges, loop_motions = loop_closure.perform(tartanvo, poses)
    
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/vo_pose.txt', poses.numpy())
    np.savetxt(trainroot+'/lc_pose.txt', loopclosure_poses.numpy())
    np.savetxt(trainroot+'/keyframes.txt', keyframes.numpy(), fmt='%d')
    np.savetxt(trainroot+'/loop_edges.txt', loop_edges.numpy(), fmt='%d')
    np.savetxt(trainroot+'/loop_motions.txt', loop_motions.numpy())
