from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
from Datasets.transformation import tartan2kitti_pypose, motion2pose_pypose, cvtSE3_pypose
from Datasets.TrajFolderDataset import TrajFolderDatasetPVGO
from evaluator.evaluate_rpe import calc_motion_error, calc_rot_error
from TartanVO import TartanVO

from pvgo import run_pvgo
from imu_integrator import run_imu_preintegrator

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pypose as pp
import numpy as np
import cv2

import os
import argparse
from os import makedirs
from os.path import isdir
from timer import Timer
import time
import wandb
os.environ["WANDB_SILENT"] = "true"


def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--flow-model-name', default='',
                        help='name of pretrained flow model (default: "")')
    parser.add_argument('--pose-model-name', default='',
                        help='name of pretrained pose model (default: "")')
    parser.add_argument('--stereo-model-name', default='',
                        help='name of pretrained stereo model (default: "")')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained vo model. if provided, this will override the other seperated models (default: "")')
    parser.add_argument('--data-root', default='',
                        help='data root dir (default: "")')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='start frame (default: 0)')
    parser.add_argument('--end-frame', type=int, default=-1,
                        help='end frame (default: -1)')
    parser.add_argument('--train-step', type=int, default=1000000,
                    help='number of interactions in total (default: 1000000)')
    parser.add_argument('--train-epoch', type=int, default=1000,
                    help='number of epoches in total (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--print-interval', type=int, default=1,
                        help='the interval for printing the loss (default: 1)')
    parser.add_argument('--snapshot-interval', type=int, default=1000,
                        help='the interval for snapshot results (default: 1000)')
    parser.add_argument('--project-name', default='',
                        help='name of the peoject (default: "")')
    parser.add_argument('--train-name', default='',
                        help='name of the training (default: "")')
    parser.add_argument('--result-dir', default='',
                        help='root directory of results (default: "")')
    parser.add_argument('--save-model-dir', default='',
                        help='root directory for saving models (default: "")')
    parser.add_argument('--use-imu', action='store_true', default=True,
                        help='use imu (default: "True")')
    parser.add_argument('--use-pvgo', action='store_true', default=False,
                        help='use pose-velocity graph optimization (default: "False")')
    parser.add_argument('--loss-weight', default='(1,1,1,1)',
                        help='weight of the loss terms (default: \'(1,1,1,1)\')')
    parser.add_argument('--mode', default='train-all', choices=['test', 'train-all'],
                        help='running mode: test, train-all (default: train-all)')
    parser.add_argument('--vo-optimizer', default='adam', choices=['adam', 'rmsprop', 'sgd'],
                        help='VO optimizer: adam, rmsprop, sgd (default: adam)')
    parser.add_argument('--use-loop-closure', action='store_true', default=False,
                        help='use loop closure or not (default: False)')
    parser.add_argument('--use-stop-constraint', action='store_true', default=False,
                        help='use stop constraint or not (default: False)')
    parser.add_argument('--use-stereo', type=int, default=0,
                        help='use stereo (default: 0)')
    parser.add_argument('--device', default='cuda',
                        help='device (default: "cuda")')
    parser.add_argument('--data-type', default='tartanair', choices=['tartanair', 'kitti', 'euroc'],
                        help='data type: tartanair, kitti, euroc (default: "tartanair")')
    parser.add_argument('--fix-model-parts', default=[], nargs='+',
                        help='fix some parts of the model (default: [])')
    parser.add_argument('--not-write-log', action='store_true', default=False,
                        help='not write log to wandb (default: "False")')
    parser.add_argument('--use-skip-frame', action='store_true', default=False,
                        help='use 1-skip frames (default: "False")')
    parser.add_argument('--rot-w', type=float, default=1,
                        help='loss rot part weight (default: 1)')
    parser.add_argument('--trans-w', type=float, default=1,
                        help='loss trans part weight (default: 1)')
    parser.add_argument('--delay-optm', action='store_true', default=False,
                        help='optimize once per traj (default: "False")')
    parser.add_argument('--train-portion', type=float, default=1,
                        help='portion to bp loss (default: "False")')
    parser.add_argument('--use-gt-scale', action='store_true', default=False,
                        help='use gt scale to correct trans scale (default: "False")')

    args = parser.parse_args()
    args.loss_weight = eval(args.loss_weight)   # string to tuple

    return args


if __name__ == '__main__':
    start_time = time.time()

    timer = Timer()
    args = get_args()
    print('\n========================================')
    print(f'args.use_gt_scale = {args.use_gt_scale}')
    print('========================================\n')

    device_id = 0
    if args.device.startswith('cuda:'):
        torch.cuda.set_device(args.device)
        device_id = int(args.device[5:])
    
    if not args.not_write_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project_name,
            name=args.train_name,
            # track hyperparameters and run metadata
            config={
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'optimizer': args.vo_optimizer,
                'data_dir': args.data_root,
                'start_frame': args.start_frame,
                'end_frame': args.end_frame,
                'loss_weight': args.loss_weight
            }
        )
        
    if args.use_stereo == 1:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
                                DownscaleFlow(), 
                                Normalize(mean=mean, std=std, keep_old=True), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])
    else:
        transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
                                DownscaleFlow(), 
                                Normalize(), 
                                ToTensor(),
                                SqueezeBatchDim()
                            ])

    dataset = TrajFolderDatasetPVGO(datadir=args.data_root, datatype=args.data_type, transform=transform,
                                    start_frame=args.start_frame, end_frame=args.end_frame,
                                    batch_size=(args.batch_size if args.use_skip_frame else None))
    if args.use_skip_frame:
        vo_batch_size = args.batch_size*2 - 1
    else:
        vo_batch_size = args.batch_size
    dataloader = DataLoader(dataset, batch_size=vo_batch_size, shuffle=False, drop_last=False)
    dataiter = iter(dataloader)

    trainroot = args.result_dir
    print('Train root:', trainroot)

    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/link.txt', np.array(dataset.links), fmt='%d')
    np.savetxt(trainroot+'/stop_frame.txt', np.array(dataset.stop_frames), fmt='%d')

    if args.use_imu and not dataset.has_imu:
        print("No IMU data! Turn use_imu to False.")
        args.use_imu = False

    if args.use_imu:
        timer.tic('imu')

        imu_motion_mode = False

        imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
            dataset.accels, dataset.gyros, dataset.imu_dts, 
            init=dataset.imu_init, gravity=dataset.gravity, 
            device=args.device, motion_mode=imu_motion_mode,
            rgb2imu_sync=dataset.rgb2imu_sync)

        imu_poses = np.concatenate((imu_trans, imu_rots), axis=1)
        np.savetxt(trainroot+'/imu_pose.txt', imu_poses)
        np.savetxt(trainroot+'/imu_vel.txt', imu_vels)

        imu_motion_mode = True

        imu_trans, imu_rots, imu_covs, imu_vels = run_imu_preintegrator(
            dataset.accels, dataset.gyros, dataset.imu_dts, 
            init=dataset.imu_init, gravity=dataset.gravity, 
            device=args.device, motion_mode=imu_motion_mode,
            rgb2imu_sync=dataset.rgb2imu_sync)

        imu_motions = np.concatenate((imu_trans, imu_rots), axis=1)
        np.savetxt(trainroot+'/imu_motion.txt', imu_motions)
        np.savetxt(trainroot+'/imu_dvel.txt', imu_vels)
        
        timer.toc('imu')
        print('imu preintegration time:', timer.tot('imu'))

        np.savetxt(trainroot+'/imu_accel.txt', dataset.accels.reshape(-1, 3))
        np.savetxt(trainroot+'/imu_gyro.txt', dataset.gyros.reshape(-1, 3))
        np.savetxt(trainroot+'/gt_vel.txt', dataset.vels.reshape(-1, 3))
        np.savetxt(trainroot+'/imu_dt.txt', dataset.imu_dts.reshape(-1, 1))
        # exit(0)

    tartanvo = TartanVO(vo_model_name=args.vo_model_name, flow_model_name=args.flow_model_name, pose_model_name=args.pose_model_name,
                        device_id=device_id, use_stereo=args.use_stereo, correct_scale=args.use_gt_scale, 
                        fix_parts=args.fix_model_parts, use_DDP=False)
    if args.vo_optimizer == 'adam':
        posenetOptimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'rmsprop':
        posenetOptimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)
    elif args.vo_optimizer == 'sgd':
        posenetOptimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr = args.lr)

    epoch = 1
    train_step_cnt = 0
    rot_th = 1.982635854754768
    trans_th = 0.013683050676729371

    current_idx = 0
    init_state = dataset.imu_init
    init_state['pose_vo'] = np.concatenate((init_state['pos'], init_state['rot']))

    vo_motions_list = []
    vo_poses_list = [np.concatenate((init_state['pos'], init_state['rot']))]
    pgo_motions_list = []
    pgo_poses_list = [np.concatenate((init_state['pos'], init_state['rot']))]
    pgo_vels_list = [init_state['vel']]
    opt_mask = [[], []]

    vo_batches_per_train_step = 1
    
    # for train_step_cnt in range(args.train_step):
    while epoch <= args.train_epoch:
        timer.tic('step')
        
        train_step_cnt += 1
        print('Start train step {} at epoch {} ...'.format(train_step_cnt, epoch))
        
        vo_new_frames_cnt = 0
        vo_new_motions = []
        vo_new_poses_np = [np.concatenate((init_state['pos'], init_state['rot']))]
        vo_new_links = []
        gt_new_motions = []

        for vo_batch_cnt in range(vo_batches_per_train_step):
            try:
                sample = next(dataiter)
            except StopIteration:
                finish_flag = True
                break

            print('\tStart vo step {} ...'.format(vo_batch_cnt))
            
            use_joint_flow = args.use_skip_frame
            if use_joint_flow:
                links = sample['link'] - current_idx
                img0 = sample['img0']
                img1 = sample['img1']
                flow = []
                for i in range(links.shape[0]):
                    if links[i, 0] + 1 == links[i, 1]:
                        f = tartanvo.pred_flow(img0[i], img1[i])
                        flow.append(f)
                    else:
                        flow_to_join = [flow[k] for k in range(links[i, 0], links[i, 1])]
                        f = tartanvo.join_flow(flow_to_join)
                        flow.append(f)
                flow = torch.stack(flow)
                sample['flow'] = flow

            timer.tic('vo')
                
            is_train = args.mode.startswith('train')
            res = tartanvo.run_batch(sample, is_train)
            motions = res['pose']
            flows = res['flow']

            timer.toc('vo')

            timer.tic('cvt')

            # print(dataset.rgb2imu_pose.matrix())
            # print(dataset.rgb2imu_pose.Inv().matrix())

            motions = tartan2kitti_pypose(motions)
            T_ic = dataset.rgb2imu_pose.to(args.device)
            motions = T_ic @ motions @ T_ic.Inv()

            T0 = vo_new_poses_np[-1]
            poses = motion2pose_pypose(motions[:args.batch_size], T0)

            motions_np = motions.detach().cpu().numpy()
            poses_np = poses.detach().cpu().numpy()

            T0_vo = vo_poses_list[-1]
            poses_vo = motion2pose_pypose(motions[:args.batch_size], T0_vo)
            poses_vo = poses_vo.detach().cpu().numpy()
            vo_motions_list.extend(motions_np)
            vo_poses_list.extend(poses_vo[1:])

            vo_new_frames_cnt += args.batch_size
            vo_new_motions.extend(motions)
            vo_new_poses_np.extend(poses_np[1:])
            vo_new_links.extend(sample['link'].numpy() - current_idx)
            gt_new_motions.extend(sample['motion'])

            timer.toc('cvt')

        if vo_new_frames_cnt == 0:
            if args.delay_optm:
                posenetOptimizer.step()
                posenetOptimizer.zero_grad()

            if args.save_model_dir is not None and len(args.save_model_dir) > 0:
                if not isdir('{}/{}'.format(args.save_model_dir, epoch)):
                    makedirs('{}/{}'.format(args.save_model_dir, epoch))
                save_model_name = '{}/{}/vonet.pkl'.format(args.save_model_dir, epoch)
                torch.save(tartanvo.vonet.state_dict(), save_model_name)

            epoch += 1
            dataiter = iter(dataloader)

            R_changes, t_changes, R_norms, t_norms = calc_motion_error(np.stack(vo_motions_list), np.stack(pgo_motions_list), allow_rescale=False)
            percent = 0.95
            partition_idx = int(percent * len(vo_motions_list))
            rot_th = np.partition(R_norms, partition_idx)[partition_idx]
            trans_th = np.partition(t_changes, partition_idx)[partition_idx]

            current_idx = 0
            init_state = dataset.imu_init
            init_state['pose_vo'] = np.concatenate((init_state['pos'], init_state['rot']))

            vo_motions_list = []
            vo_poses_list = [np.concatenate((init_state['pos'], init_state['rot']))]
            pgo_motions_list = []
            pgo_poses_list = [np.concatenate((init_state['pos'], init_state['rot']))]
            pgo_vels_list = [init_state['vel']]
            opt_mask = [[], []]
            
            continue

        vo_new_motions = torch.stack(vo_new_motions)
        vo_new_motions_np = vo_new_motions.detach().cpu().numpy()
        vo_new_poses_np = np.stack(vo_new_poses_np)
        vo_new_links = np.stack(vo_new_links)
        gt_new_motions = torch.stack(gt_new_motions)
        
        timer.tic('pgo')

        if args.use_imu and args.use_pvgo:
            st = current_idx
            end = current_idx + vo_new_frames_cnt
            current_imu_rots = imu_rots[st:end]
            current_imu_trans = imu_trans[st:end]
            current_imu_vels = imu_vels[st:end]
            current_dts = dataset.rgb_dts[st:end]
            trans_loss, rot_loss, pgo_poses, pgo_vels, pgo_motions = run_pvgo(
                vo_new_poses_np, vo_new_motions, vo_new_links, 
                current_imu_rots, current_imu_trans, current_imu_vels, init_state, current_dts, 
                device=args.device, loss_weight=args.loss_weight, stop_frames=dataset.stop_frames,
                init_with_imu_rot=True, init_with_imu_vel=True)

        pgo_motions_list.extend(pgo_motions)
        pgo_poses_list.extend(pgo_poses[1:])
        pgo_vels_list.extend(pgo_vels[1:])

        timer.toc('pgo')
        
        timer.tic('opt')

        if args.mode.startswith('train') and epoch > 0:
            # use masks
            R_changes, t_changes, R_norms, t_norms = calc_motion_error(vo_new_motions_np, pgo_motions, allow_rescale=False)
            # rot_mask = R_norms >= rot_th
            # trans_mask = t_changes >= trans_th
            if args.train_portion >= 1:
                rot_mask = np.ones(R_norms.shape[0]).astype(bool)
                trans_mask = np.ones(t_norms.shape[0]).astype(bool)
            else:
                rot_mask = np.zeros(R_norms.shape[0]).astype(bool)
                trans_mask = np.zeros(t_norms.shape[0]).astype(bool)
                itv = int(1 / args.train_portion)
                for i in range(R_norms.shape[0]):
                    if (current_idx + i) % itv == 0:
                        rot_mask[i] = True
                        trans_mask[i] = True

            if np.any(rot_mask) or np.any(trans_mask):
                if np.any(rot_mask) and np.any(trans_mask):
                    loss_bp = torch.cat((args.rot_w * rot_loss[rot_mask], args.trans_w * trans_loss[trans_mask]))
                elif np.any(rot_mask):
                    loss_bp = rot_loss[rot_mask]
                else:
                    loss_bp = trans_loss[trans_mask]

                if not args.delay_optm:
                    posenetOptimizer.zero_grad()
                loss_bp.backward(torch.ones_like(loss_bp))
                if not args.delay_optm:
                    posenetOptimizer.step()

            opt_mask[0].extend(rot_mask)
            opt_mask[1].extend(trans_mask)

        timer.toc('opt')

        timer.toc('step')

        timer.tic('print')

        if train_step_cnt % args.print_interval == 0:
            st = current_idx
            end = current_idx + vo_new_frames_cnt
            poses_gt = dataset.poses[st:end+1]

            motions_gt = gt_new_motions
            if args.data_type == 'tartanair':
                motions_gt = tartan2kitti_pypose(motions_gt).numpy()
            else:
                motions_gt = cvtSE3_pypose(motions_gt).numpy()
            
            vo_R_errs, vo_t_errs, R_norms, t_norms = calc_motion_error(motions_gt, vo_new_motions_np, allow_rescale=False)
            print('Pred: R:%.5f t:%.5f' % (np.mean(vo_R_errs), np.mean(vo_t_errs)))

            # imu_R_errs, _ = calc_rot_error(motions_gt[:args.batch_size, 3:], current_imu_rots)
            # print('IMU : R:%.5f' % (np.mean(imu_R_errs)))
            
            pgo_R_errs, pgo_t_errs, _, _ = calc_motion_error(motions_gt, pgo_motions, allow_rescale=False)
            print('PVGO: R:%.5f t:%.5f' % (np.mean(pgo_R_errs), np.mean(pgo_t_errs)))

            print('Norm: R:%.5f t:%.5f' % (np.mean(R_norms), np.mean(t_norms)))

            pose_R_errs, pose_t_errs, _, _ = calc_motion_error(poses_gt, np.array(vo_poses_list[st:end+1]), allow_rescale=False)
            print('VO P: R:%.5f t:%.5f' % (np.mean(pose_R_errs), np.mean(pose_t_errs)))
            pose_R_errs, pose_t_errs, _, _ = calc_motion_error(poses_gt, pgo_poses, allow_rescale=False)
            print('PG P: R:%.5f t:%.5f' % (np.mean(pose_R_errs), np.mean(pose_t_errs)))

            # if not args.not_write_log:
            #     for i in range(args.batch_size):
            #         wandb.log({
            #             'vo mrot err': vo_R_errs[i],
            #             'vo mtrans err': vo_t_errs[i],
            #             'imu mrot err': imu_R_errs[i],
            #             'pgo mrot err': pgo_R_errs[i],
            #             'pgo mtrans err': pgo_t_errs[i],
            #             'gt mrot norm': R_norms[i],
            #             'gt mtrans norm': t_norms[i],
            #             'pose rot err': pose_R_errs[i],
            #             'pose trans err': pose_t_errs[i],
            #             'vo-pgo mrot': vo_R_errs[i] - pgo_R_errs[i],
            #             'vo-pgo mtrans': vo_t_errs[i] - pgo_t_errs[i],
            #         }, step = current_idx + i)
                    
            #         if args.use_skip_frame and i < args.batch_size-1:
            #             j = i + args.batch_size
            #             wandb.log({
            #                 'vo skip mrot err': vo_R_errs[j],
            #                 'vo skip mtrans err': vo_t_errs[j],
            #                 'pgo skip mrot err': pgo_R_errs[j],
            #                 'pgo skip mtrans err': pgo_t_errs[j],
            #                 'gt skip mrot norm': R_norms[j],
            #                 'gt skip mtrans norm': t_norms[j],
            #                 'vo-pgo skip mrot': vo_R_errs[j] - pgo_R_errs[j],
            #                 'vo-pgo skip mtrans': vo_t_errs[j] - pgo_t_errs[j],
            #             }, step = current_idx + i)

        timer.toc('print')

        timer.tic('snapshot')

        # if train_step_cnt % args.snapshot_interval == 0:
        #     traindir = trainroot+'/'+str(train_step_cnt)
        #     if not isdir(traindir):
        #         makedirs(traindir)
            
        #     np.savetxt(traindir+'/pose.txt', poses_np)
        #     np.savetxt(traindir+'/motion.txt', motions_np)

        #     np.savetxt(traindir+'/pgo_pose.txt', pgo_poses)
        #     np.savetxt(traindir+'/pgo_motion.txt', pgo_motions)
        #     np.savetxt(traindir+'/pgo_vel.txt', pgo_vels)

        #     np.savetxt(traindir+'/gt_pose.txt', poses_gt)
        #     np.savetxt(traindir+'/gt_motion.txt', motions_gt)

        if not isdir('{}/{}'.format(trainroot, epoch)):
            makedirs('{}/{}'.format(trainroot, epoch))
        np.savetxt('{}/{}/vo_pose.txt'.format(trainroot, epoch), np.stack(vo_poses_list))
        np.savetxt('{}/{}/vo_motion.txt'.format(trainroot, epoch), np.stack(vo_motions_list))
        np.savetxt('{}/{}/pgo_pose.txt'.format(trainroot, epoch), np.stack(pgo_poses_list))
        np.savetxt('{}/{}/pgo_motion.txt'.format(trainroot, epoch), np.stack(pgo_motions_list))
        np.savetxt('{}/{}/pgo_vel.txt'.format(trainroot, epoch), np.stack(pgo_vels_list))
        np.savetxt('{}/{}/opt_mask.txt'.format(trainroot, epoch), np.array(opt_mask))

        timer.toc('snapshot')

        print('[time] step: {}, vo: {}, pgo: {}, opt: {}, print: {}, snapshot: {}'.format(
            timer.last('step'), timer.last('vo'), timer.last('pgo'), timer.last('opt'), 
            timer.last('print'), timer.last('snapshot')))

        current_idx += vo_new_frames_cnt
        init_state = {'rot':pgo_poses[-1][3:], 'pos':pgo_poses[-1][:3], 'vel':pgo_vels[-1], 'pose_vo':poses_vo[-1]}
        init_state['rot'] /= np.linalg.norm(init_state['rot'])

    if not args.not_write_log:
        wandb.finish(quiet=True)

    end_time = time.time()
    print('\nTotal time consume:', end_time-start_time)
