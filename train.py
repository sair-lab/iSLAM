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

    global imu_poses_list, imu_motions_list, vo_rev_poses_list, vo_rcam_poses_list
    imu_poses_list = [init_pose]
    imu_motions_list = []
    vo_rev_poses_list = [init_pose]
    vo_rcam_poses_list = [init_pose]


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

    start_time = time.time()
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
                            shuffle=False, drop_last=True)
    
    timer.toc('dataset')

    ############################## init VO model ######################################################################
    pose_model_name = args.pose_model_name
    if args.start_epoch > 1:
        for i in range(args.start_epoch-1, 0, -1):
            last_model_name = '{}/{}/vonet.pkl'.format(args.save_model_dir, i)
            if isfile(last_model_name):
                pose_model_name = last_model_name
                break

    print('Loading VO model:', args.vo_model_name, pose_model_name)

    tartanvo = TartanVO(
        vo_model_name=args.vo_model_name, pose_model_name=pose_model_name, 
        correct_scale=args.use_gt_scale, fix_parts=args.fix_model_parts, use_kitti_coord=(dataset.datatype!='tartanair')
    )
    if args.vo_optimizer == 'adam':
        vo_optimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)
    elif args.vo_optimizer == 'rmsprop':
        vo_optimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)
    elif args.vo_optimizer == 'sgd':
        vo_optimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)

    ############################## init IMU module ######################################################################
    imu_denoise_model_name = args.imu_denoise_model_name
    if args.start_epoch > 1:
        for i in range(args.start_epoch-1, 0, -1):
            last_model_name = '{}/{}/imudenoise.pkl'.format(args.save_model_dir, i)
            if isfile(last_model_name):
                imu_denoise_model_name = last_model_name
                break

    print('Loading IMU model:', imu_denoise_model_name)

    imu_module = IMUModule(
        dataset.accels, dataset.gyros, dataset.imu_dts,
        dataset.accel_bias, dataset.gyro_bias,
        dataset.imu_init, dataset.gravity, dataset.rgb2imu_sync, 
        device='cuda', denoise_model_name=imu_denoise_model_name,
        denoise_accel=True, denoise_gyro=(dataset.datatype!='kitti')
    )

    if imu_module.use_denoise_model:
        imu_optimizer = optim.Adam(imu_module.denoiser.parameters(), lr=3e-5)

    ############################## logs before running ######################################################################
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/timestamp.txt', dataset.rgb_ts, fmt='%.3f')

    ############################## init before loop ######################################################################
    train_target = [''] + ['vo', 'imu'] * 100
    prev_vo_motions = None

    epoch = args.start_epoch
    epoch_step = len(dataset) // args.batch_size
    step_cnt = (args.start_epoch - 1) * epoch_step
    total_step = epoch_step * args.train_epoch

    init_epoch()
    
    ############################## main training loop ######################################################################
    while epoch <= args.train_epoch:    # this while loops per batch (step)
        timer.tic('step')

        # try to load data batch
        try:
            timer.tic('load')
            sample = next(dataiter)
            timer.toc('load')

        # procedure when an epoch finishes
        except StopIteration:
            # optimize after each epoch (go through the whole trajectory)
            if train_target[epoch] == 'vo':
                vo_optimizer.step()
                vo_optimizer.zero_grad()
            elif train_target[epoch] == 'imu':
                imu_optimizer.step()
                imu_optimizer.zero_grad()

            if args.save_model_dir is not None and len(args.save_model_dir) > 0:
                if not isdir('{}/{}'.format(args.save_model_dir, epoch)):
                    makedirs('{}/{}'.format(args.save_model_dir, epoch))
                if train_target[epoch] == 'vo':
                    save_model_name = '{}/{}/vonet.pkl'.format(args.save_model_dir, epoch)
                    torch.save(tartanvo.vonet.state_dict(), save_model_name)
                elif train_target[epoch] == 'imu':
                    save_model_name = '{}/{}/imudenoise.pkl'.format(args.save_model_dir, epoch)
                    torch.save(imu_module.denoiser.state_dict(), save_model_name)

            snapshot(final=True)

            prev_vo_motions = pp.SE3(np.stack(vo_motions_list)).cuda()

            epoch += 1
            init_epoch()
            
            continue

        step_cnt += 1
        print('\nStart train step {} at epoch {} ...'.format(step_cnt, epoch))
        print('Train target:', train_target[epoch])

        ############################## forward VO ######################################################################
        timer.tic('vo')

        try:
            assert train_target[epoch] != 'vo'
            motions = prev_vo_motions[current_idx:current_idx+args.batch_size]
            
        except:
            vo_result = tartanvo(sample)
            motions = vo_result['motion']
            T_IL = dataset.rgb2imu_pose.to(motions.device)
            motions = T_IL @ motions @ T_IL.Inv()

        timer.toc('vo')
 
        T0 = pgo_poses_list[-1]
        poses = motion2pose_pypose(motions[:args.batch_size], T0)
        motions_np = motions.detach().cpu().numpy()
        poses_np = poses.detach().cpu().numpy()

        T0_vo = vo_poses_list[-1]
        poses_vo = motion2pose_pypose(motions[:args.batch_size], T0_vo)
        poses_vo_np = poses_vo.detach().cpu().numpy()
        vo_motions_list.extend(motions_np)
        vo_poses_list.extend(poses_vo_np[1:])

        ############################## IMU preintegration ######################################################################
        timer.tic('imu')

        st = current_idx
        end = current_idx + args.batch_size

        imu_trans, imu_rots, imu_covs, imu_vels = imu_module.integrate(
            st, end, init_state, motion_mode=False
        )
        imu_poses = pp.SE3(torch.cat((imu_trans, imu_rots.tensor()), axis=1))
        imu_motions = pose2motion_pypose(imu_poses)
        imu_poses_list.extend(imu_poses[1:].numpy())
        imu_motions_list.extend(imu_motions.numpy())

        imu_dtrans, imu_drots, imu_dcovs, imu_dvels = imu_module.integrate(
            st, end, init_state, motion_mode=True
        )

        timer.toc('imu')
        
        ############################## run PVGO ######################################################################
        timer.tic('pgo')

        dts = sample['dt']
        links = base_links = sample['link'] - current_idx

        trans_loss, rot_loss, pgo_poses, pgo_vels, covs = run_pvgo(
            imu_poses, imu_vels,
            motions, links, dts,
            imu_drots, imu_dtrans, imu_dvels,
            device='cuda', radius=1e4,
            loss_weight=args.loss_weight,
            target=train_target[epoch]
        )
        pgo_motions = pose2motion_pypose(pgo_poses)

        pgo_motions = pgo_motions.numpy()
        pgo_poses = pgo_poses.numpy()
        pgo_vels = pgo_vels.numpy()

        pgo_motions_list.extend(pgo_motions)
        pgo_poses_list.extend(pgo_poses[1:])
        pgo_vels_list.extend(pgo_vels[1:])

        timer.toc('pgo')
        
        ############################## backpropagation ######################################################################
        timer.tic('opt')

        # backpropagate VO
        loss_bp = torch.cat((args.rot_w * rot_loss, args.trans_w * trans_loss))
        # only backpropagate, no optimize
        if loss_bp.requires_grad:
            loss_bp.backward(torch.ones_like(loss_bp))

        timer.toc('opt')

        ############################## log and snapshot ######################################################################
        timer.tic('snapshot')

        if step_cnt < 10 or step_cnt % args.snapshot_interval == 0:
            snapshot()

        timer.toc('snapshot')

        current_idx += args.batch_size
        # set init state as the last frame in this batch
        init_state = {'rot':pgo_poses[-1][3:], 'pos':pgo_poses[-1][:3], 'vel':pgo_vels[-1], 'pose_vo':poses_vo[-1]}
        # normalize quaternion
        init_state['rot'] /= np.linalg.norm(init_state['rot'])

        timer.toc('step')

        print('[time] step: {:.3f}, load: {:.3f}, vo: {:.3f}, pgo: {:.3f}, opt: {:.3f}'.format(
            timer.last('step'), timer.last('load'), timer.last('vo'), timer.last('pgo'), timer.last('opt'), 
        ))

        print('Epoch progress: {:.2%}, time left {:.2f}min'.format((step_cnt-epoch_step*(epoch-1))/epoch_step, (epoch_step*epoch-step_cnt)*timer.avg('step')/60))
        print('Train progress: {:.2%}, time left {:.2f}min'.format(step_cnt/total_step, (total_step-step_cnt)*timer.avg('step')/60))

    end_time = time.time()
    print('\nTotal time consume:', end_time-start_time)
