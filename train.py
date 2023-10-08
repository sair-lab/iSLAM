from Datasets.transformation import tartan2kitti_pypose, motion2pose_pypose, pose2motion_pypose, cvtSE3_pypose
from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
from dense_ba import DenseReprojectionLoss, SparseReprojectionLoss, FAST_point_detector
from Datasets.utils import kitti_raw2odometry, euroc_raw2short
from Datasets.TrajFolderDataset import TrajFolderDataset
from Evaluator.evaluate_rpe import calc_motion_error
from TartanVO import TartanVO
from mapper import Mapper

from pvgo import run_pvgo
from imu_integrator import IMUModule
from loop_closure import LoopClosure

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

    global keyframes, covs_dict_list
    keyframes = None
    covs_dict_list = {}

    global mapper
    mapper = Mapper()


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
    if args.vo_reverse_edge:
        np.savetxt('{}/{}/vo_rev_pose.txt'.format(trainroot, epoch), np.stack(vo_rev_poses_list))
    if args.vo_right_cam:
        np.savetxt('{}/{}/vo_rcam_pose.txt'.format(trainroot, epoch), np.stack(vo_rcam_poses_list))

    if not args.use_gt_scale and args.enable_mapping:
        mapper.save_data('{}/{}/cloud.txt'.format(trainroot, epoch))
        mapper.write_ply('{}/{}/cloud.ply'.format(trainroot, epoch))

    for k, v in covs_dict_list.items():
        np.savetxt('{}/{}/info_{}.txt'.format(trainroot, epoch, k), np.stack(v))

    if final:
        if keyframes is not None:
            np.savetxt('{}/{}/loop_pose.txt'.format(trainroot, epoch), loopclosure_poses.numpy())
            np.savetxt('{}/{}/keyframes.txt'.format(trainroot, epoch), keyframes.numpy(), fmt='%d')
            np.savetxt('{}/{}/loop_edge.txt'.format(trainroot, epoch), loop_edges.numpy(), fmt='%d')
            np.savetxt('{}/{}/loop_motion.txt'.format(trainroot, epoch), loop_motions.numpy())


def reverse_sample(sample, left_right=False):
    res = sample.copy()
    if not left_right:
        res['img0'], res['img1'] = res['img1'], res['img0']
        res['img0_r'], res['img1_r'] = res['img1_r'], res['img0_r']
        res['img0_norm'], res['img1_norm'] = res['img1_norm'], res['img0_norm']
        res['img0_r_norm'], res['img1_r_norm'] = res['img1_r_norm'], res['img0_r_norm']
        res['link'] = res['link'][:, (1,0)]
        res['motion'] = pp.SE3(res['motion']).Inv().tensor()
    else:
        res['img0'], res['img0_r'] = res['img0_r'], res['img0']
        res['img1'], res['img1_r'] = res['img1_r'], res['img1']
        res['img0_norm'], res['img0_r_norm'] = res['img0_r_norm'], res['img0_norm']
        res['img1_norm'], res['img1_r_norm'] = res['img1_r_norm'], res['img1_norm']
        res['extrinsic'] = pp.SE3(res['extrinsic']).Inv().tensor()
    return res


if __name__ == '__main__':

    start_time = time.time()
    timer = Timer()

    torch.set_float32_matmul_precision('high')

    print('\ndevice:', torch.cuda.get_device_name(), '\tcount:', torch.cuda.device_count())
    args = get_args()
    print('\n==============================================')
    print(args)
    print('==============================================\n')

    trainroot = args.result_dir

    ############################## init dataset ######################################################################   
    timer.tic('dataset')
    print('Loading dataset ...')
    
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
    print('Load dataset time:', timer.tot('dataset'))

    ############################## init VO model ######################################################################
    if args.start_epoch == 1:
        tartanvo = TartanVO(
            vo_model_name=args.vo_model_name, 
            correct_scale=args.use_gt_scale, fix_parts=args.fix_model_parts
        )
    else:
        last_pose_model_name = '{}/{}/vonet.pkl'.format(args.save_model_dir, args.start_epoch - 1)
        tartanvo = TartanVO(
            vo_model_name=args.vo_model_name, pose_model_name=last_pose_model_name, 
            correct_scale=args.use_gt_scale, fix_parts=args.fix_model_parts
        )
    if args.vo_optimizer == 'adam':
        vo_optimizer = optim.Adam(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)
    elif args.vo_optimizer == 'rmsprop':
        vo_optimizer = optim.RMSprop(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)
    elif args.vo_optimizer == 'sgd':
        vo_optimizer = optim.SGD(tartanvo.vonet.flowPoseNet.parameters(), lr=args.lr)

    ############################## init IMU module ######################################################################
    imu_module = IMUModule(
        dataset.accels, dataset.gyros, dataset.imu_dts,
        dataset.imu_init, dataset.gravity, dataset.rgb2imu_sync, 
        device='cuda', denoise_model_name=args.imu_denoise_model_name,
        denoise_accel=True, denoise_gyro=(dataset.datatype!='kitti')
    )

    ############################## init loop closure ######################################################################
    folder = 'loop_edges1-5'
    if dataset.datatype == 'kitti':
        idx = kitti_raw2odometry(dataset.datadir)
        loop_edges_file = f'./{folder}/result_kitti{idx}/loop_ransac.txt'
        loop_motions_file = f'./{folder}/result_kitti{idx}/loop_ransac_motion.txt'
    elif dataset.datatype == 'euroc':
        idx = euroc_raw2short(dataset.datadir)
        loop_edges_file = f'./{folder}/result_euroc-{idx}/loop_ransac.txt'
        loop_motions_file = f'./{folder}/result_euroc-{idx}/loop_ransac_motion.txt'
    if isfile(loop_edges_file):
        loop_closure = LoopClosure(dataset, args.batch_size, loop_edges_file, loop_motions_file)
    else:
        loop_closure = None

    # for debug, output edge figures
    if loop_closure is not None:
        for l in loop_closure.loop_edges:
            s = dataset.get_pair(l[0], l[1])
            img0 = s['img0'].permute(1, 2, 0).numpy()
            img1 = s['img1'].permute(1, 2, 0).numpy()
            img0 = (img0 * 255).astype(np.uint8)
            img1 = (img1 * 255).astype(np.uint8)
            cv2.imwrite(trainroot+'/'+str(l[0].item())+'.png', img0)
            cv2.imwrite(trainroot+'/'+str(l[1].item())+'.png', img1)

    ############################## logs before running ######################################################################
    with open(trainroot+'/args.txt', 'w') as f:
        f.write(str(args))
    np.savetxt(trainroot+'/gt_pose.txt', dataset.poses)
    np.savetxt(trainroot+'/timestamp.txt', dataset.rgb_ts, fmt='%.3f')

    ############################## init before loop ######################################################################
    epoch = args.start_epoch
    epoch_step = len(dataset) // args.batch_size
    step_cnt = (args.start_epoch - 1) * epoch_step
    total_step = epoch_step * args.train_epoch
    init_epoch()
    
    ############################## main training loop ######################################################################
    while epoch <= args.train_epoch:    # this while loops per batch (step)
        timer.tic('step')

        try:
            step_cnt += 1
            print('\nStart train step {} at epoch {} ...'.format(step_cnt, epoch))

            timer.tic('load')
            # load data batch
            sample = next(dataiter)
            timer.toc('load')

        # procedure when an epoch finishes
        except StopIteration:
            # optimize after each epoch (go through the whole trajectory)
            vo_optimizer.step()
            vo_optimizer.zero_grad()

            if args.save_model_dir is not None and len(args.save_model_dir) > 0:
                if not isdir('{}/{}'.format(args.save_model_dir, epoch)):
                    makedirs('{}/{}'.format(args.save_model_dir, epoch))
                save_model_name = '{}/{}/vonet.pkl'.format(args.save_model_dir, epoch)
                torch.save(tartanvo.vonet.state_dict(), save_model_name)

            # loop closure
            if loop_closure is not None:
                loopclosure_poses, keyframes, loop_edges, loop_motions = \
                    loop_closure.perform(pp.SE3(np.stack(pgo_poses_list)), tartanvo)

            snapshot(final=True)

            epoch += 1
            init_epoch()
            
            continue

        ############################## forward VO ######################################################################
        timer.tic('vo')

        vo_result = tartanvo(sample)
        motions = vo_result['motion']
        T_IL = dataset.rgb2imu_pose.to(motions.device)
        motions = T_IL @ motions @ T_IL.Inv()

        if args.vo_reverse_edge:
            sample_rev = reverse_sample(sample, False)
            motions_rev = tartanvo(sample_rev)
            motions_rev = T_IL @ motions_rev @ T_IL.Inv()

        if args.vo_right_cam:
            sample_rcam = reverse_sample(sample, True)
            T_IR = T_IL @ dataset.right2left_pose.to(T_IL.device)
            motions_in_rcam = T_IR.Inv() @ motions.detach() @ T_IR
            scales_in_rcam = torch.norm(motions_in_rcam.translation(), dim=1)
            motions_rcam = tartanvo(sample_rcam, given_scale=scales_in_rcam)
            motions_rcam = T_IR @ motions_rcam @ T_IR.Inv()

        timer.toc('vo')

        # batch_motion_se3 = pp.cumprod(motions, dim=0)[-1].Log()
        # print(batch_motion_se3.size())
        # jac = {}
        # for i in range(6):
        #     batch_motion_se3[i].backward(retain_graph=True)
        #     for name, para in tartanvo.vonet.flowPoseNet.named_parameters():
        #         if name in jac:
        #             jac[name].append(para.grad.clone())
        #         else:
        #             jac[name] = [para.grad.clone()]
        #         para.grad.zero_()
        # mem = 0
        # for name in jac.keys():
        #     jac[name] = torch.stack(jac[name])
        #     print(name, jac[name].size(), jac[name].element_size(), jac[name].nelement())
        #     mem += jac[name].element_size() * jac[name].nelement()
        # print('mem in GB', mem / 1e9)
        # quit()
 
        T0 = pgo_poses_list[-1]
        poses = motion2pose_pypose(motions[:args.batch_size], T0)
        motions_np = motions.detach().cpu().numpy()
        poses_np = poses.detach().cpu().numpy()

        T0_vo = vo_poses_list[-1]
        poses_vo = motion2pose_pypose(motions[:args.batch_size], T0_vo)
        poses_vo_np = poses_vo.detach().cpu().numpy()
        vo_motions_list.extend(motions_np)
        vo_poses_list.extend(poses_vo_np[1:])

        if args.vo_reverse_edge:
            T0_vo_rev = vo_rev_poses_list[-1]
            poses_vo_rev = motion2pose_pypose(motions_rev[:args.batch_size].Inv(), T0_vo_rev)
            vo_rev_poses_list.extend(poses_vo_rev.detach().cpu().numpy()[1:])

        if args.vo_right_cam:
            T0_vo_rcam = vo_rcam_poses_list[-1]
            poses_vo_rcam = motion2pose_pypose(motions_rcam[:args.batch_size], T0_vo_rcam)
            vo_rcam_poses_list.extend(poses_vo_rcam.detach().cpu().numpy()[1:])

        ############################## IMU preintegration ######################################################################
        timer.tic('imu')

        st = current_idx
        end = current_idx + args.batch_size

        imu_trans, imu_rots, imu_covs, imu_vels = imu_module.integrate(st, end, init_state, motion_mode=False)
        imu_poses = pp.SE3(torch.cat((imu_trans, imu_rots.tensor()), axis=1))
        imu_motions = pose2motion_pypose(imu_poses)
        imu_poses_list.extend(imu_poses[1:].numpy())
        imu_motions_list.extend(imu_motions.numpy())

        imu_dtrans, imu_drots, imu_dcovs, imu_dvels = imu_module.integrate(st, end, init_state, motion_mode=True)

        timer.toc('imu')
        
        ############################## run PVGO ######################################################################
        timer.tic('pgo')

        dts = sample['dt']
        links = base_links = sample['link'] - current_idx

        if args.vo_reverse_edge:
            motions = torch.cat([motions, motions_rev], dim=0)
            links = torch.cat([links, base_links[:, (1,0)]], dim=0)
            # print(links)
            # print(motions.shape)
        
        if args.vo_right_cam:
            motions = torch.cat([motions, motions_rcam], dim=0)
            links = torch.cat([links, base_links], dim=0)
            # print(links)
            # print(motions.shape)

        if keyframes is not None:
            a = torch.searchsorted(keyframes, current_idx, right=False)
            b = torch.searchsorted(keyframes, current_idx+args.batch_size, right=True)
            if b - a > 1:
                loop_motions = pose2motion_pypose(loopclosure_poses[a:b]).tensor()
                loop_links = torch.tensor([[keyframes[i], keyframes[i+1]] for i in range(a, b-1)]) - current_idx
                motions = torch.cat([motions, loop_motions.to(motions.device)], dim=0)
                links = torch.cat([links, loop_links], dim=0)
            # print(links)
            # print(motions.shape)

        if len(args.loss_weight) == 5:
            height, width = vo_result['depth'].shape[-2:]
            point2d = FAST_point_detector(sample['img0'], height, width, N=10)
            fx, fy, cx, cy = vo_result['intrinsic']
            reproj = SparseReprojectionLoss(
                point2d, vo_result['depth'], vo_result['flow'], 
                fx, fy, cx, cy, dataset.rgb2imu_pose, motions.device
            )
        else:
            reproj = None

        trans_loss, rot_loss, pgo_poses, pgo_vels, covs = run_pvgo(
            imu_poses, imu_vels,
            motions, links, dts,
            imu_drots, imu_dtrans, imu_dvels,
            device='cuda', radius=1e4,
            loss_weight=args.loss_weight,
            reproj=reproj
        )
        pgo_motions = pose2motion_pypose(pgo_poses)

        pgo_motions = pgo_motions.numpy()
        pgo_poses = pgo_poses.numpy()
        pgo_vels = pgo_vels.numpy()

        pgo_motions_list.extend(pgo_motions)
        pgo_poses_list.extend(pgo_poses[1:])
        pgo_vels_list.extend(pgo_vels[1:])

        for k in covs.keys():
            if k not in covs_dict_list:
                covs_dict_list[k] = []
            covs_dict_list[k].extend(covs[k])

        timer.toc('pgo')
        
        ############################## backpropagation ######################################################################
        timer.tic('opt')

        # generate mask according to the portion of training
        if args.train_portion >= 1:
            rot_mask = np.ones(motions.shape[0]).astype(bool)
            trans_mask = np.ones(motions.shape[0]).astype(bool)
        else:
            rot_mask = np.zeros(motions.shape[0]).astype(bool)
            trans_mask = np.zeros(motions.shape[0]).astype(bool)
            itv = int(1 / args.train_portion)
            for i in range(R_norms.shape[0]):
                if (current_idx + i) % itv == 0:
                    rot_mask[i] = True
                    trans_mask[i] = True

        if np.any(rot_mask) or np.any(trans_mask):
            loss_bp = torch.cat((args.rot_w * rot_loss[rot_mask], args.trans_w * trans_loss[trans_mask]))
            # only backpropagate, no optimize
            loss_bp.backward(torch.ones_like(loss_bp))

        timer.toc('opt')

        ############################## mapping ######################################################################
        if not args.use_gt_scale and args.enable_mapping:
            depths = vo_result['depth']
            masks = vo_result['mask']

            for i in range(0, args.batch_size, 2):
                img = sample['img0'][i].permute(1, 2, 0).numpy()
                img = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_LINEAR)
                img = torch.from_numpy(img).permute(2, 0, 1)

                intrinsic_calib = sample['intrinsic_calib']
                fx, fy, cx, cy = intrinsic_calib[i] / 4

                mapper.add_frame(img, depths[i], pp.SE3(pgo_poses[i]) @ dataset.rgb2imu_pose, fx, fy, cx, cy, masks[i])

        ############################## log and snapshot ######################################################################
        timer.tic('print')

        if step_cnt % args.print_interval == 0:
            st = current_idx
            end = current_idx + args.batch_size
            poses_gt = dataset.poses[st:end+1]
            motions_gt = dataset.motions[st:end]
            motions_gt = cvtSE3_pypose(motions_gt).numpy()
            
            vo_R_errs, vo_t_errs, R_norms, t_norms = calc_motion_error(motions_gt, motions_np, allow_rescale=False)
            print('Pred: R:%.5f t:%.5f' % (np.mean(vo_R_errs), np.mean(vo_t_errs)))
            
            pgo_R_errs, pgo_t_errs, _, _ = calc_motion_error(motions_gt, pgo_motions, allow_rescale=False)
            print('PVGO: R:%.5f t:%.5f' % (np.mean(pgo_R_errs), np.mean(pgo_t_errs)))

            print('Norm: R:%.5f t:%.5f' % (np.mean(R_norms), np.mean(t_norms)))

            pose_R_errs, pose_t_errs, _, _ = calc_motion_error(poses_gt, np.array(vo_poses_list[st:end+1]), allow_rescale=False)
            print('VO P: R:%.5f t:%.5f' % (np.mean(pose_R_errs), np.mean(pose_t_errs)))

            pose_R_errs, pose_t_errs, _, _ = calc_motion_error(poses_gt, pgo_poses, allow_rescale=False)
            print('PG P: R:%.5f t:%.5f' % (np.mean(pose_R_errs), np.mean(pose_t_errs)))

        timer.toc('print')

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

        # for test
        # if step_cnt >= 5:
        #     break

    end_time = time.time()
    print('\nTotal time consume:', end_time-start_time)
