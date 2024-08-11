from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
from Datasets.transformation import motion2pose_pypose, pose2motion_pypose
from Datasets.TrajFolderDataset import TrajFolderDataset

from TartanVO import TartanVO

from pvgo import run_pvgo
from imu_integrator import IMUModule

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pypose as pp
import numpy as np
import cv2

import os
from os import makedirs
from os.path import isdir, isfile

from timer import Timer
import time

from arguments import get_args


def init_epoch():
    global current_idx, init_state, dataiter
    current_idx = 0
    init_state = dataset.imu_init
    dataiter = iter(dataloader)

    init_pose = np.concatenate((init_state['pos'], init_state['rot']))

    # init lists for recording trajectories
    global vo_motions_list, vo_poses_list, pgo_motions_list, pgo_poses_list, pgo_vels_list
    vo_motions_list = []
    vo_poses_list = [init_pose]
    pgo_motions_list = []
    pgo_poses_list = [init_pose]
    pgo_vels_list = [init_state['vel']]

    global imu_poses_list, imu_motions_list, imu_vels_list, imu_dtrans_list, imu_drots_list, imu_dvels_list
    imu_poses_list = [init_pose]
    imu_motions_list = []
    imu_vels_list = [init_state['vel']]
    imu_dtrans_list = []
    imu_drots_list = []
    imu_dvels_list = []


def snapshot(final=False):
    if not isdir('{}/{}'.format(trainroot, epoch)):
        makedirs('{}/{}'.format(trainroot, epoch))

    np.savetxt('{}/{}/vo_pose.txt'.format(trainroot, epoch), np.stack(vo_poses_list))
    np.savetxt('{}/{}/vo_motion.txt'.format(trainroot, epoch), np.stack(vo_motions_list))
    np.savetxt('{}/{}/pgo_pose.txt'.format(trainroot, epoch), np.stack(pgo_poses_list))
    np.savetxt('{}/{}/pgo_motion.txt'.format(trainroot, epoch), np.stack(pgo_motions_list))
    np.savetxt('{}/{}/pgo_vel.txt'.format(trainroot, epoch), np.stack(pgo_vels_list))
    np.savetxt('{}/{}/imu_pose.txt'.format(trainroot, epoch), np.stack(imu_poses_list))
    np.savetxt('{}/{}/imu_motion.txt'.format(trainroot, epoch), np.stack(imu_motions_list))


if __name__ == '__main__':

    timer = Timer()

    torch.set_float32_matmul_precision('high')

    args = get_args()
    print('\n==============================================')
    print(args)
    print('==============================================\n')

    trainroot = args.result_dir

    ############################## init dataset ######################################################################   
    timer.tic('dataset')
    print('Loading dataset:', args.data_root)
    
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
        datadir=args.data_root, datatype=args.data_type, transform=transform,
        start_frame=args.start_frame, end_frame=args.end_frame
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.worker_num, 
                            shuffle=False, drop_last=False)
    
    timer.toc('dataset')

    ############################## init VO model ######################################################################
    print('Loading VO model:', args.vo_model_name, args.pose_model_name)

    tartanvo = TartanVO(
        vo_model_name=args.vo_model_name, pose_model_name=args.pose_model_name, 
        correct_scale=args.use_gt_scale, fix_parts=args.fix_model_parts, use_kitti_coord=(dataset.datatype!='tartanair')
    )

    ############################## init IMU module ######################################################################
    print('Loading IMU model:', args.imu_denoise_model_name)

    imu_module = IMUModule(
        dataset.accels, dataset.gyros, dataset.imu_dts,
        dataset.accel_bias, dataset.gyro_bias,
        dataset.imu_init, dataset.gravity, dataset.rgb2imu_sync, 
        device='cuda', denoise_model_name=args.imu_denoise_model_name,
        denoise_accel=True, denoise_gyro=(dataset.datatype!='kitti')
    )

    ############################## logs before running ######################################################################
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/timestamp.txt', dataset.rgb_ts, fmt='%.3f')

    ############################## init variables ######################################################################
    epoch = 1
    init_epoch()
    
    ############################## forward VO and IMU Nets ######################################################################
    while True:
        timer.tic('step')

        # try to load data batch
        try:
            timer.tic('load')
            sample = next(dataiter)
            timer.toc('load')
        except StopIteration:
            break

        print(f'Processing {current_idx}/{len(dataset)} ...')

        ############################## forward VO ######################################################################
        timer.tic('vo')

        vo_result = tartanvo(sample)
        motions = vo_result['motion']
        T_IL = dataset.rgb2imu_pose.to(motions.device)
        motions = T_IL @ motions @ T_IL.Inv()

        timer.toc('vo')

        T0_vo = vo_poses_list[-1]
        poses_vo = motion2pose_pypose(motions[:args.batch_size], T0_vo)
        poses_vo_np = poses_vo.detach().cpu().numpy()
        vo_poses_list.extend(poses_vo_np[1:])
        vo_motions_list.extend(motions.detach().cpu().numpy())

        ############################## IMU preintegration ######################################################################
        timer.tic('imu')

        st = current_idx
        end = min(current_idx + args.batch_size, len(dataset))

        imu_trans, imu_rots, imu_covs, imu_vels = imu_module.integrate(
            st, end, init_state, motion_mode=False
        )
        imu_poses = pp.SE3(torch.cat((imu_trans, imu_rots.tensor()), axis=1))
        imu_motions = pose2motion_pypose(imu_poses)
        imu_poses_list.extend(imu_poses[1:].detach().cpu().numpy())
        imu_motions_list.extend(imu_motions.detach().cpu().numpy())
        imu_vels_list.extend(imu_vels[1:].detach().cpu().numpy())

        imu_dtrans, imu_drots, imu_dcovs, imu_dvels = imu_module.integrate(
            st, end, init_state, motion_mode=True
        )
        imu_dtrans_list.extend(imu_dtrans.detach().cpu().numpy())
        imu_drots_list.extend(imu_drots.detach().cpu().numpy())
        imu_dvels_list.extend(imu_dvels.detach().cpu().numpy())

        timer.toc('imu')

        ############################## local PVGO ######################################################################
        timer.tic('pgo')

        dts = sample['dt']
        links = sample['link'] - current_idx

        _, _, pgo_poses, pgo_vels, covs = run_pvgo(
            imu_poses, imu_vels,
            motions, links, dts,
            imu_drots, imu_dtrans, imu_dvels,
            device='cuda', radius=1e4,
            loss_weight=args.loss_weight,
            target=''
        )
        pgo_motions = pose2motion_pypose(pgo_poses)

        pgo_poses_np = pgo_poses.detach().cpu().numpy()
        pgo_vels_np = pgo_vels.detach().cpu().numpy()

        pgo_motions_list.extend(pgo_motions.detach().cpu().numpy())
        pgo_poses_list.extend(pgo_poses_np[1:])
        pgo_vels_list.extend(pgo_vels_np[1:])

        timer.toc('pgo')

        ############################## after batch ######################################################################
        current_idx += args.batch_size

        # set init state as the last frame in this batch
        init_state = {'rot':pgo_poses_np[-1][3:], 'pos':pgo_poses_np[-1][:3], 'vel':pgo_vels_np[-1]}
        # last_vel = (poses_vo_np[-1][:3] - poses_vo_np[-2][:3]) / sample['dt'][-1].item()
        # init_state = {'rot':poses_vo_np[-1][3:], 'pos':poses_vo_np[-1][:3], 'vel':last_vel, 'pose_vo':poses_vo_np[-1]}
        # normalize quaternion
        init_state['rot'] /= np.linalg.norm(init_state['rot'])

        timer.toc('step')

        print('[time] step: {:.3f}, load: {:.3f}, vo: {:.3f}, pgo: {:.3f}'.format(
            timer.last('step'), timer.last('load'), timer.last('vo'), timer.last('pgo'), 
        ))
        
    ############################## run PVGO ######################################################################
    timer.tic('global')

    print('Running global optimization ...')

    links = torch.tensor(dataset.links)
    dts = torch.tensor(dataset.rgb_dts, dtype=torch.float32)
    imu_poses = pp.SE3(np.stack(imu_poses_list))
    imu_vels = torch.tensor(np.stack(imu_vels_list), dtype=torch.float32)
    # vo_poses = pp.SE3(np.stack(vo_poses_list))
    # vo_vels = torch.cat([torch.tensor(dataset.vels[0:1], dtype=torch.float32), torch.diff(vo_poses.translation(), dim=0) * dts.unsqueeze(-1)], dim=0)
    # pgo_poses = pp.SE3(np.stack(pgo_poses_list))
    # pgo_vels = torch.tensor(np.stack(pgo_vels_list), dtype=torch.float32)
    motions = pp.SE3(np.stack(vo_motions_list))
    imu_drots = pp.SO3(np.stack(imu_drots_list))
    imu_dtrans = torch.tensor(np.stack(imu_dtrans_list), dtype=torch.float32)
    imu_dvels = torch.tensor(np.stack(imu_dvels_list), dtype=torch.float32)

    _, _, pgo_poses, pgo_vels, covs = run_pvgo(
        imu_poses, imu_vels,
        motions, links, dts,
        imu_drots, imu_dtrans, imu_dvels,
        device='cpu', radius=1e4,
        loss_weight=args.loss_weight,
        target=''
    )
    pgo_motions = pose2motion_pypose(pgo_poses)

    pgo_motions_list = pgo_motions.numpy()
    pgo_poses_list = pgo_poses.numpy()
    pgo_vels_list = pgo_vels.numpy()

    timer.toc('global')

    print('[time] global: {:.3f}'.format(timer.last('global')))
    
    ############################## log and snapshot ######################################################################
    timer.tic('snapshot')
    snapshot()
    timer.toc('snapshot')
