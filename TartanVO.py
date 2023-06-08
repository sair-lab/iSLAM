import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import cv2
import time
import random
import numpy as np
import pypose as pp

from Network.VONet import VONet, MultiCamVONet
from Network.StereoVONet import StereoVONet

from dense_ba import scale_from_disp_flow
from Datasets.transformation import tartan2kitti_pypose
from Datasets.utils import save_images, warp_images

np.set_printoptions(precision=4, suppress=True, threshold=10000)


class TartanVO:
    def __init__(self, vo_model_name=None, pose_model_name=None, flow_model_name=None, stereo_model_name=None,
                    use_imu=False, use_stereo=0, device_id=0, correct_scale=True, fix_parts=(), use_DDP=True,
                    extrinsic_encoder_layers=2, trans_head_layers=3, normalize_extrinsic=False):
        
        # import ipdb;ipdb.set_trace()
        self.device_id = device_id
        
        if use_stereo==1:
            self.vonet = VONet(fix_parts=fix_parts)
            # # wenshan's version below
            # stereonorm = 0.02 # the norm factor for the stereonet
            # self.vonet = StereoVONet(network=1, intrinsic=True, flowNormFactor=1.0, stereoNormFactor=stereonorm, poseDepthNormFactor=0.25, 
            #                             down_scale=True, config=1, fixflow=True, fixstereo=True, autoDistTarget=0.)
        elif use_stereo==2.1 or use_stereo==2.2:
            self.vonet = MultiCamVONet(flowNormFactor=1.0, use_stereo=use_stereo, fix_parts=fix_parts,
                                        extrinsic_encoder_layers=extrinsic_encoder_layers, trans_head_layers=trans_head_layers)

        # load the whole model
        if vo_model_name is not None and vo_model_name != "":
            print('load vo network...')
            self.load_model(self.vonet, vo_model_name)
        # can override part of the model
        if flow_model_name is not None and flow_model_name != "":
            print('load pwc network...')
            # data = torch.load('models/' + flow_model_name)
            # self.vonet.flowNet.load_state_dict(data)
            self.load_model(self.vonet.flowNet, flow_model_name)
        if pose_model_name is not None and pose_model_name != "":
            print('load pose network...')
            self.load_model(self.vonet.flowPoseNet, pose_model_name)
        if use_stereo==1 and stereo_model_name is not None and stereo_model_name != "":
            print('load stereo network...')
            self.load_model(self.vonet.stereoNet, stereo_model_name)
            
        self.pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32).cuda(self.device_id) # the output scale factor
        self.flow_norm = 20 # scale factor for flow

        self.use_imu = use_imu
        self.use_stereo = use_stereo
        self.correct_scale = correct_scale
        self.normalize_extrinsic = normalize_extrinsic
        
        self.vonet.flowNet = self.vonet.flowNet.cuda(self.device_id)
        if use_stereo==1:
            self.vonet.stereoNet = self.vonet.stereoNet.cuda(self.device_id)
        self.vonet.flowPoseNet = self.vonet.flowPoseNet.cuda(self.device_id)
        if use_DDP:
            self.vonet.flowPoseNet = DistributedDataParallel(self.vonet.flowPoseNet, device_ids=[self.device_id])


    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname, map_location='cuda:%d'%self.device_id)
        model_dict = model.state_dict()

        preTrainDictTemp = {}
        for k, v in preTrainDict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                preTrainDictTemp[k] = v

        # print('model_dict:')
        # for k in model_dict:
        #     print(k, model_dict[k].shape)
        # print('pretrain:')
        # for k in preTrainDict:
        #     print(k, preTrainDict[k].shape)

        if 0 == len(preTrainDictTemp):
            for k, v in preTrainDict.items():
                kk = k[7:]
                if kk in model_dict and v.size() == model_dict[kk].size():
                    preTrainDictTemp[kk] = v

        if 0 == len(preTrainDictTemp):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        for k in model_dict.keys():
            if k not in preTrainDictTemp:
                print("! [load_model] Key {} in model but not in {}!".format(k, modelname))
                # if k.endswith('weight'):
                #     print('\tinit with kaiming_normal_')
                #     w = torch.rand_like(model_dict[k])
                #     nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
                # else:
                #     print('\tinit to zeros')
                #     w = torch.zeros_like(model_dict[k])
                # preTrainDictTemp[k] = w

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)

        del preTrainDict
        del preTrainDictTemp

        return model


    def run_batch(self, sample, is_train=True):        
        # import ipdb;ipdb.set_trace()
        nb = False
        img0   = sample['img0'].cuda(self.device_id, non_blocking=nb)
        img1   = sample['img1'].cuda(self.device_id, non_blocking=nb)
        intrinsic = sample['intrinsic'].cuda(self.device_id, non_blocking=nb)

        if self.use_stereo==1:
            img0_norm = sample['img0_norm'].cuda(self.device_id, non_blocking=nb)
            img0_r_norm = sample['img0_r_norm'].cuda(self.device_id, non_blocking=nb)
            intrinsic_calib = sample['intrinsic_calib']
            baseline = torch.linalg.norm(sample['extrinsic'][:, :3], dim=1)
            precalc_flow = sample['flow'] if 'flow' in sample else None
        elif self.use_stereo==2.1 or self.use_stereo==2.2:
            extrinsic = sample['extrinsic'].cuda(self.device_id, non_blocking=nb)
            if self.normalize_extrinsic:
                extrinsic_scale = torch.linalg.norm(extrinsic[:, :3], dim=1).view(-1, 1)
                extrinsic[:, :3] /= extrinsic_scale
            img0_r = sample['img0_r'].cuda(self.device_id, non_blocking=nb)

        if is_train:
            self.vonet.train()
        else:
            self.vonet.eval()

        res = {}

        _ = torch.set_grad_enabled(is_train)

        if self.use_stereo==1:
            flow, disp, pose = self.vonet(img0, img1, img0_norm, img0_r_norm, intrinsic)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back

            if not self.correct_scale:

                # pose_gt = sample['motion'].cuda(self.device_id)
                # pose = pose_gt

                if precalc_flow is None:
                    flow *= 5
                    # flow_gt = sample['flow'].cuda(self.device_id)
                    # flow_gt /= 4
                    # flow = flow_gt
                else:
                    flow = precalc_flow
                
                disp *= 50/4
                # depth = sample['depth0'].cuda(self.device_id)
                # disp_gt = 320/4*0.25 / depth
                # disp = disp_gt

                # img0_warp = warp_images('temp', img1, flow)
                # save_images('temp', img0, prefix='', suffix='_orig', fx=1/4, fy=1/4)
                # save_images('temp', img1, prefix='', suffix='_x', fx=1/4, fy=1/4)

                # img0_r = sample['img0_r']
                # disp_warp = warp_images('temp2', img0_r, -disp)
                # save_images('temp2', img0, prefix='', suffix='_orig', fx=1/4, fy=1/4)
                # save_images('temp2', img0_r, prefix='', suffix='_x', fx=1/4, fy=1/4)

                # flow_scale = flow[0] / flow_gt[0]
                # print('flow_scale', flow_scale.mean(), flow_scale.median())

                # disp_scale = disp[0] / disp_gt[0]
                # print('disp_scale', disp_scale.mean(), disp_scale.median())

                # print('depth', torch.min(depth), torch.max(depth), torch.mean(depth))
                # print('flow', torch.min(flow), torch.max(flow), torch.mean(flow))
                # print('disp', torch.min(disp), torch.max(disp), torch.mean(disp), torch.median(disp))
                
                pose_ENU_SE3 = tartan2kitti_pypose(pose)

                # print('pose_ENU_SE3', pose_ENU_SE3)
                # print('baseline', baseline)

                img0_np = img0.cpu().numpy()
                img0_np = img0_np.transpose(0, 2, 3, 1)
                img0_np = (img0_np*255).astype(np.uint8)
                edge = []
                for i in range(img0_np.shape[0]):
                    im = cv2.resize(img0_np[i], None, fx=1/4, fy=1/4)
                    e = cv2.Canny(im, 50, 100)
                    e = cv2.dilate(e, np.ones((5,5), np.uint8))
                    e = e > 0
                    edge.append(e)
                edge = torch.from_numpy(np.stack(edge)).cuda(self.device_id)

                scale = []
                mask = []
                for i in range(pose.shape[0]):
                    fx, fy, cx, cy = intrinsic_calib[i] / 4
                    disp_th_dict = {'kitti':5, 'euroc':1}
                    r, s, m = scale_from_disp_flow(disp[i], flow[i], pose_ENU_SE3[i], fx, fy, cx, cy, baseline[i], 
                                                    mask=edge[i], disp_th=disp_th_dict[sample['datatype'][i]])
                    scale.append(s)
                    mask.append(m)
                scale = torch.stack(scale)
                mask = torch.stack(mask)

                # gt_scale = torch.norm(sample['motion'][:, :3], dim=1)
                # print('scale', scale)
                # print('gt_scale', gt_scale)
                
                trans = torch.nn.functional.normalize(pose[:, :3], dim=1) * scale.view(-1, 1)
                # trans = torch.nn.functional.normalize(pose[:, :3], dim=1)
                pose = torch.cat([trans, pose[:, 3:]], dim=1)

                # for i in range(len(edge)):
                #     mask_img = mask[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)*255
                #     cv2.imwrite('temp/{}_mask.png'.format(i), mask_img)
                
                # gt_pose = sample['motion'].cuda(self.device_id)
                # gt_scale = torch.linalg.norm(gt_pose[:, :3], dim=1).view(-1, 1)
                # print('scale', scale.view(-1))
                # print('gt_scale', gt_scale.view(-1))
                # print('frac', (scale / gt_scale).view(-1))

                # gt_trans_norm = torch.nn.functional.normalize(gt_pose[:, :3], dim=1)
                # trans_norm = torch.nn.functional.normalize(pose[:, :3], dim=1)
                # cross = torch.sum(gt_trans_norm * trans_norm, dim=1)
                # trans_angle = torch.arccos(torch.clamp(cross, min=-1, max=1)) * 180 / 3.14
                # scale_err = torch.abs(gt_scale - scale)
                # scale_err_percent = scale_err / gt_scale
                # print('trans_angle', trans_angle.view(-1))
                # print('scale_err', scale_err.view(-1))
                # print('scale_err_percent', scale_err_percent.view(-1))
                # print('trans', pose)

                res['scale'] = scale

            res['pose'] = pose
            res['flow'] = flow
            res['disp'] = disp

        elif self.use_stereo==2.1 or self.use_stereo==2.2:
            flowAB, flowAC, pose = self.vonet(img0, img0_r, img1, intrinsic, extrinsic)
            pose = pose * self.pose_std # The output is normalized during training, now scale it back
            if self.normalize_extrinsic:
                pose[:, :3] *= extrinsic_scale
            res['pose'] = pose
            res['flowAB'] = flowAB
            res['flowAC'] = flowAC
            
        if self.correct_scale:
            pose = self.handle_scale(sample, pose)
            res['pose'] = pose

        return res


    def handle_scale(self, sample, pose):
        motion_tar = sample['motion']

        scale = torch.norm(motion_tar[:, :3], dim=1).cuda(self.device_id)
        trans_est = torch.nn.functional.normalize(pose[:, :3], dim=1) * scale.view(-1,1)
        pose = torch.cat((trans_est, pose[:, 3:]), dim=1)
        
        return pose


    def pred_flow(self, img0, img1):
        img0 = img0.cuda(self.device_id)
        img1 = img1.cuda(self.device_id)

        batched = len(img0.shape) == 4
        if not batched:
            img0 = img0.unsqueeze(0)
            img1 = img1.unsqueeze(0)

        flow, _ = self.vonet.flowNet(torch.cat([img0, img1], dim=1))
        flow = flow[0] * 5

        if not batched:
            flow = flow.squeeze(0)

        return flow


    def join_flow(self, flow_to_join):
        height, width = flow_to_join[0].shape[-2:]

        u_lin = torch.linspace(0, width-1, width)
        v_lin = torch.linspace(0, height-1, height)
        u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
        uv = torch.stack([u, v]).cuda(self.device_id)

        x = uv.unsqueeze(0)
        flow_to_join.reverse()
        for f in flow_to_join:
            grid = (f + uv).permute(1, 2, 0).unsqueeze(0)
            grid[..., 0] = grid[..., 0] / width * 2 - 1
            grid[..., 1] = grid[..., 1] / height * 2 - 1
            x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        x = x.squeeze(0)
        zero_mask = torch.logical_and(x[0]==0, x[1]==0).repeat(2, 1, 1)
        x = torch.where(zero_mask, -1, x)

        return x - uv


    # def validate_model_result(self, train_step_cnt=None,writer =None):
    #     kitti_ate, kitti_trans, kitti_rot = self.validate_model(count=train_step_cnt, writer=writer,verbose = False, datastr = 'kitti')
    #     euroc_ate = self.validate_model(count=train_step_cnt, writer = writer,verbose = False, datastr = 'euroc')

    #     print("  VAL %s #%d - KITTI-ATE/T/R/EuRoc-ATE: %.4f  %.4f  %.4f %.4f"  % (self.args.exp_prefix[:-1], 
    #     self.val_count, kitti_ate, kitti_trans, kitti_rot, euroc_ate))
    #     score = kitti_ate/ self.kitti_ate_bs/self.kitti_trans_bs + kitti_trans + kitti_rot/self.kitti_rot_bs   + euroc_ate/self.euroc_ate_bs
    #     print('score: ', score)


    # def validate_model(self,writer,count = None, verbose = False, datastr =None,source_dir = '/home/data2'):
    #     euroc_dataset = ['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02', 'V2_03']
    #     if datastr == None:
    #         print('Here is a bug!!!')

    #     args = load_args('args/args_'+datastr+'.pkl')[0]
    #     # read testdir adn posefile from kitti from tarjectory 1 to 10
    #     self.count = count

    #     result_dict = {}

    #     for i in range(11):
    #         # args.test_dir = '/data/azcopy/kitti/10/image_left'
    #         # args.pose_file = '/data/azcopy/kitti/10/pose_left.txt'
    #         if datastr == 'kitti':
    #             args.test_dir = source_dir + '/kitti/'+str(i).zfill(2)+'/image_left'
    #             args.pose_file = source_dir + '/kitti/'+str(i).zfill(2)+'/pose_left.txt'
                
    #             # Specify the path to the KITTI calib.txt file
    #             args.kitti_intrinsics_file = source_dir + '/kitti/'+str(i).zfill(2)+'/calib.txt'
    #             calib_file = source_dir + '/kitti/'+str(i).zfill(2)+'/calib.txt'
    #             focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    #             result_dict['kitti_ate'] = []
    #             result_dict['kitti_trans'] = []
    #             result_dict['kitti_rot'] = []

    #         elif datastr == 'euroc':
    #             args.test_dir = source_dir + '/euroc/'+euroc_dataset[i]+ '/cam0' + '/data2'
    #             args.pose_file = source_dir + '/euroc/'+euroc_dataset[i] + '/cam0' +'/pose_left.txt'
    #             focalx, focaly, centerx, centery = dataset_intrinsics(datastr) 
                
    #             result_dict['euroc_ate'] = []

    #         transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    #         testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
    #                                             focalx=focalx, focaly=focaly, centerx=centerx, centery=centery,verbose = False)
            
    #         testDataset  = MultiTrajFolderDataset(DatasetType=TrajFolderDatasetMultiCam,
    #                                             root=args.data_root, transform=transform, mode = 'test')
            
    #         testDataloader  = DataLoader(testDataset,  batch_size=args.batch_size, shuffle=False,num_workers=args.worker_num)
            
    #         args.batch_size = 64
    #         args.worker_num = 4
    #         testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
    #                                             shuffle=False, num_workers=args.worker_num)
    #         testDataiter = iter(testDataloader)

    #         motionlist = []
    #         testname = datastr + '_' + args.model_name.split('.')[0]
    #         # length = len(testDataiter)

    #         motionlist_array = np.zeros((len(testDataset), 6))
    #         batch_size = args.batch_size
            
    #         for idx in tqdm(range(len(testDataiter))):    
    #             try:
    #                 sample = next(testDataiter)
    #             except StopIteration:
    #                 break
                
    #             # motions, flow = self.validate_test_batch(sample)
    #             res =  self.run_batch(sample)
    #             motions = res['pose']

    #             try:
    #                 motionlist_array[batch_size*idx:batch_size*idx+batch_size,:] = motions
    #             except:
    #                 motionlist_array[batch_size*idx:,:] = motions

    #         # poselist = ses2poses_quat(np.array(motionlist))
    #         poselist = ses2poses_quat( motionlist_array)
            
    #         # calculate ATE, RPE, KITTI-RPE
    #         # if args.pose_file.endswith('.txt'):
    #         evaluator = TartanAirEvaluator()
    #         results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
            
    #         if datastr=='kitti':
    #             result_dict['kitti_ate'].append(results['ate_score'])
    #             result_dict['kitti_trans'].append( results['kitti_score'][1]* 100  )
    #             result_dict['kitti_rot'].append( results['kitti_score'][0] * 100)
    #             print("==> KITTI: %d ATE: %.4f,\t KITTI-T/R: %.4f, %.4f" %(i, results['ate_score'], results['kitti_score'][1]* 100, results['kitti_score'][0]* 100 ))

    #         elif datastr=='euroc':
    #             result_dict['euroc_ate'].append(results['ate_score'])
    #             print("==> EuRoc: %s ATE: %.4f" %(euroc_dataset[i], results['ate_score']))
        
    #     # print average result
    #     if datastr=='euroc':
    #         ate_score = np.mean(result_dict['ate_score'])
    #         print("==> EuRoc: ATE: %.4f" %(ate_score))

    #         if not self.args.not_write_log:
    #             writer.add_scalar('Error/EuRoc_ATE', results['ate_score'], self.count)
    #             wandb.log({"EuRoc_ATE": results['ate_score']}, step=self.count)

    #         return ate_score
        
    #     elif datastr == 'kitti':
    #         ate_score = np.mean(result_dict['kitti_ate'])
    #         trans_score = np.mean(result_dict['kitti_trans'])
    #         rot_score = np.mean(result_dict['kitti_rot'])

    #         print("==> KITTI: ATE: %.4f" %(ate_score))
    #         print("==> KITTI: Trans: %.4f" %(trans_score))
    #         print("==> KITTI: Rot: %.4f" %(rot_score))

    #         if not self.args.not_write_log:
    #             writer.add_scalar('Error/KITTI_ATE', results['ate_score'], self.count)
    #             writer.add_scalar('Error/KITTI_trans', results['kitti_score'][1]* 100, self.count)
    #             writer.add_scalar('Error/KITTI_rot', results['kitti_score'][0]* 100, self.count)
    #             wandb.log({"KITTI_ATE": results['ate_score'], "KITTI_trans": results['kitti_score'][1]* 100, "KITTI_rot": results['kitti_score'][0]* 100 }, step=self.count)

    #         return ate_score, trans_score, rot_score


    # def validate_test_batch(self, sample):
    #     # self.test_count += 1
        
    #     # import ipdb;ipdb.set_trace()
    #     img0   = sample['img1'].cuda()
    #     img1   = sample['img2'].cuda()
    #     intrinsic = sample['intrinsic'].cuda()
    #     inputs = [img0, img1, intrinsic]

    #     self.vonet.eval()

    #     with torch.no_grad():
    #         starttime = time.time()

    #         imgs = torch.cat((inputs[0], inputs[1]), 1)
    #         intrinsic = inputs[2]
    #         # in tartanvo val 
    #         # flow, pose = self.vonet(inputs)
    #         # in tartanvo training
    #         flow_output, pose_output = self.vonet([imgs, intrinsic])
    #         # flow, pose = self.vonet([imgs, intrinsic])

    #         res = self.run_batch(sample)
    #         motion = res['pose']
            
    #         # print(pose)

    #         # Transfer SE3 to translation and rotation
            
    #         if pose.shape[-1] == 7:
    #             posenp,_,_ = SE32ws(pose)

    #         else:
    #             posenp = pose.data.cpu().numpy()

    #         inferencetime = time.time()-starttime
    #         # import ipdb;ipdb.set_trace()
            
    #         # Very very important
    #         posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
    #         flownp = flow.data.cpu().numpy()
    #         # flownp = flownp * self.flow_norm

    #     # calculate scale from GT posefile
    #     if 'motion' in sample:
    #         motions_gt = sample['motion']
    #         scale = np.linalg.norm(motions_gt[:,:3], axis=1)
    #         trans_est = posenp[:,:3]    
            
    #         '''
    #         trans_est_norm = np.linalg.norm(trans_est,axis=1).reshape(-1,1)
    #         eps = 1e-12 * np.ones(trans_est_norm.shape)
    #         '''
            
    #         # trans_est = trans_est/np.max(( trans_est_norm , eps)) * scale.reshape(-1,1)
    #         # trans_est = trans_est/np.max(( trans_est_norm , eps)) * scale.reshape(-1,1)

    #         posenp[:,:3] = trans_est 
    #         # print(posenp)
    #     else:
    #         print('    scale is not given, using 1 as the default scale value..')

    #     return posenp, flownp

