import pypose as pp
import numpy as np
import pandas
import yaml
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

from os import listdir, path
from os.path import isdir, isfile
from torch.utils.data.distributed import DistributedSampler

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer
from .loopDetector import gt_pose_loop_detector, bow_orb_loop_detector, adj_loop_detector
from .stopDetector import gt_vel_stop_detector
from .loopDetector import multicam_frame_selector


def sync_data(ts_src, ts_tar):
    res = []
    j = 0
    for t in ts_tar:
        while j+1 < len(ts_src) and abs(ts_src[j+1]-t) <= abs(ts_src[j]-t):
            j += 1
        res.append(j)
    for i in range(len(res)-1):
        if res[i+1] - res[i] <= 0:
            print('sync_data error', i, ts_tar[i:i+2], ts_src[max(0,res[i]-5):min(len(ts_src), res[i]+5)])
    return np.array(res)


def intrinsic2matrix(intrinsic):
    fx, fy, cx, cy = intrinsic
    return np.array([
        fx, 0, cx,
        0, fy, cy,
        0,  0,  1
    ], dtype=np.float32).reshape(3, 3)

def matrix2intrinsic(m):
    return np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]], dtype=np.float32)


def stereo_rectify(left_intrinsic, left_distortion, right_intrinsic, right_distortion, width, height, right2left_pose):
    left_K = intrinsic2matrix(left_intrinsic).astype(np.float64)
    right_K = intrinsic2matrix(right_intrinsic).astype(np.float64)
    left_distortion = left_distortion.astype(np.float64)
    right_distortion = right_distortion.astype(np.float64)
    R = right2left_pose.Inv().rotation().matrix().numpy().astype(np.float64)
    T = right2left_pose.Inv().translation().numpy().astype(np.float64)

    # print(R, R.dtype, T, T.dtype)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion,
        (width, height), R, T, alpha=0
    )
    
    left_map = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    left_intrinsic_new = matrix2intrinsic(P1)
    right_intrinsic_new = matrix2intrinsic(P2)
    right2left_pose_new = pp.SE3([-P2[0,3]/P2[0,0],0,0, 0,0,0,1]).to(torch.float32)

    # print(right2left_pose, right2left_pose_new)

    return left_intrinsic_new, right_intrinsic_new, right2left_pose_new, left_map, right_map


class TartanAirTrajFolderLoader:
    def __init__(self, datadir, sample_step=1, start_frame=0, end_frame=-1):

        ############################## load images ######################################################################
        imgfolder = datadir + '/image_left'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles)-1, dtype=np.float32) * 0.1

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_right'):
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        ############################## load flow ######################################################################
        if isdir(datadir + '/flow'):
            imgfolder = datadir + '/flow'
            files = listdir(imgfolder)
            self.flowfiles = [(imgfolder +'/'+ ff) for ff in files if ff.endswith('_flow.npy')]
            self.flowfiles.sort()
        else:
            self.flowfiles = None

        ############################## load depth ######################################################################
        if isdir(datadir + '/depth_left'):
            imgfolder = datadir + '/depth_left'
            files = listdir(imgfolder)
            self.depthfiles = [(imgfolder +'/'+ ff) for ff in files if ff.endswith('_depth.npy')]
            self.depthfiles.sort()
        else:
            self.depthfiles = None

        ############################## load calibrations ######################################################################
        self.intrinsic = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0,   0, 0, 0, 1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        ############################## load gt poses ######################################################################
        posefile = datadir + '/pose_left.txt'
        self.poses = np.loadtxt(posefile).astype(np.float32)
        self.vels = None

        ############################## load imu data ######################################################################
        if isdir(datadir + '/imu'):
            imudir = datadir + '/imu'
            # acceleration in the body frame
            accels = np.load(imudir + '/accel_left.npy')
            # angular rate in the body frame
            gyros = np.load(imudir + '/gyro_left.npy')
            # velocity in the world frame
            vels = np.load(imudir + '/vel_left.npy')
            # # accel w/o gravity in body frame
            # accels_nograv = np.load(path.join(imudir, "accel_nograv_body.npy")).astype(np.float32)
            
            # self.accel_bias = -1.0 * np.array([0.01317811, 0.00902851, -0.00521479])
            accel_bias = -1.0 * np.array([-0.02437125, -0.00459115, -0.00392401])
            gyro_bias = -1.0 * np.array([0, 0, 0])
            self.accels = accels - accel_bias
            self.gyros = gyros - gyro_bias

            # imu_fps = 100
            self.imu_dts = np.ones(len(accels)-1, dtype=np.float32) * 0.01
            
            # img_fps = 10
            self.rgb2imu_sync = np.arange(0, len(self.rgbfiles)) * 10

            self.rgb2imu_pose = pp.SE3([0, 0, 0,   0, 0, 0, 1]).to(dtype=torch.float32)

            self.gravity = -9.81

            self.has_imu = True
            self.vels = vels

        else:
            self.has_imu = False
            self.vels = None


class EuRoCTrajFolderLoader:
    def __init__(self, datadir, sample_step=1, start_frame=0, end_frame=-1):
        all_timestamps = []

        ############################## load images ######################################################################
        df = pandas.read_csv(datadir + '/cam0/data.csv')
        timestamps_left = df.values[:, 0].astype(int) // int(1e6)
        all_timestamps.append(timestamps_left)
        self.rgbfiles = datadir + '/cam0/data/' + df.values[:, 1]

        ############################## load stereo right images ######################################################################
        if isfile(datadir + '/cam1/data.csv'):
            df = pandas.read_csv(datadir + '/cam1/data.csv')
            timestamps_right = df.values[:, 0].astype(int) // int(1e6)
            all_timestamps.append(timestamps_right)
            self.rgbfiles_right = datadir + '/cam1/data/' + df.values[:, 1]
        else:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        with open(datadir + '/cam0/sensor.yaml') as f:
            res = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.intrinsic = np.array(res['intrinsics'], dtype=np.float32)
            distortion = np.array(res['distortion_coefficients'], dtype=np.float32)
            T_BL = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)
        
        if self.rgbfiles_right is not None:
            with open(datadir + '/cam1/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.intrinsic_right = np.array(res['intrinsics'], dtype=np.float32)
                distortion_right = np.array(res['distortion_coefficients'], dtype=np.float32)
                T_BR = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)

        if self.rgbfiles_right is not None:
            T_LR = np.matmul(np.linalg.inv(T_BL), T_BR)
            self.right2left_pose = pp.from_matrix(torch.tensor(T_LR), ltype=pp.SE3_type).to(dtype=torch.float32)

            self.require_undistort = True
            img = cv2.imread(self.rgbfiles_right[0])
            h, w = img.shape[:2]
            self.intrinsic, self.intrinsic_right, self.right2left_pose, self.imgmap, self.imgmap_right = stereo_rectify(
                self.intrinsic, distortion, self.intrinsic_right, distortion_right, w, h, self.right2left_pose)
        else:
            self.require_undistort = False

        ############################## load gt poses ######################################################################
        df = pandas.read_csv(datadir + '/state_groundtruth_estimate0/data.csv')
        timestamps_pose = df.values[:, 0].astype(int) // int(1e6)
        all_timestamps.append(timestamps_pose)
        self.poses = df.values[:, (1,2,3, 5,6,7,4)].astype(np.float32)
        self.vels = df.values[:, 8:11].astype(np.float32)
        accel_bias = df.values[:, 14:17].astype(np.float32)
        gyro_bias = df.values[:, 11:14].astype(np.float32)

        ############################## align timestamps ######################################################################
        timestamps = set(all_timestamps[0])
        for i in range(1, len(all_timestamps)):
            timestamps = timestamps.intersection(set(all_timestamps[i]))
        self.rgbfiles = self.rgbfiles[[i for i, t in enumerate(timestamps_left) if t in timestamps]]
        if self.rgbfiles_right is not None:
            self.rgbfiles_right = self.rgbfiles_right[[i for i, t in enumerate(timestamps_right) if t in timestamps]]
        self.poses = self.poses[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        self.vels = self.vels[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        timestamps = np.array(list(timestamps))
        timestamps.sort()
        self.rgb_dts = np.diff(timestamps).astype(np.float32) * 1e-3

        ############################## load imu data ######################################################################
        if isfile(datadir + '/imu0/data.csv'):
            df = pandas.read_csv(datadir + '/imu0/data.csv')
            timestamps_imu = df.values[:, 0].astype(int) // int(1e6)
            accels = df.values[:, 4:7].astype(np.float32)
            gyros = df.values[:, 1:4].astype(np.float32)

            imu2pose_sync = sync_data(timestamps_pose, timestamps_imu)
            self.accels = accels - accel_bias[imu2pose_sync]
            self.gyros = gyros - gyro_bias[imu2pose_sync]
            # self.accels = accels
            # self.gyros = gyros

            self.imu_dts = np.diff(timestamps_imu).astype(np.float32) * 1e-3
            
            self.rgb2imu_sync = sync_data(timestamps_imu, timestamps)

            with open(datadir + '/imu0/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                T_BI = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)
                T_IL = np.matmul(np.linalg.inv(T_BI), T_BL)
                self.rgb2imu_pose = pp.from_matrix(torch.tensor(T_IL), ltype=pp.SE3_type).to(dtype=torch.float32)

            self.gravity = 9.81

            self.has_imu = True

        else:
            self.has_imu = False

        # print("rgbfiles", self.rgbfiles[:10])
        # print("rebfiles_right", self.rgbfiles_right[:10])
        # print("intrinsic", self.intrinsic)
        # print("intrinsic_right", self.intrinsic_right)
        # print("rgb2imu_pose", self.rgb2imu_pose)
        # print("rgb2imu_sync", self.rgb2imu_sync[:10])
        # print("right2left_pose", self.right2left_pose, self.right2left_pose.Log())


class KITTITrajFolderLoader:
    def __init__(self, datadir, sample_step=1, start_frame=0, end_frame=-1):
        import pykitti

        datadir_split = datadir.split('/')
        # Change this to the directory where you store KITTI data
        basedir = '/'.join(datadir_split[:-2])

        # Specify the dataset to load
        date = datadir_split[-2]
        drive = datadir_split[-1].split('_')[-2]

        # Load the data. Optionally, specify the frame range to load.
        dataset = pykitti.raw(basedir, date, drive)
        # dataset = pykitti.raw(basedir, date, drive, frames=range(0, 20, 5))

        # dataset.calib:         Calibration data are accessible as a named tuple
        # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
        # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
        # dataset.camN:          Returns a generator that loads individual images from camera N
        # dataset.get_camN(idx): Returns the image from camera N at idx
        # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
        # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
        # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
        # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
        # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
        # dataset.get_velo(idx): Returns the velodyne scan at idx

        ############################## load times ######################################################################
        # timestamps = np.array([t.timestamp() for t in dataset.timestamps])
        ts_imu = self.load_timestamps(datadir, 'oxts')
        ts_rgb = self.load_timestamps(datadir, 'image_02')
        # self.rgb2imu_sync = np.array([i for i in range(len(self.rgbfiles))])
        self.rgb2imu_sync = sync_data(ts_imu, ts_rgb)

        ############################## load images ######################################################################
        self.rgbfiles = dataset.cam2_files
        self.rgb_dts = np.diff(ts_rgb).astype(np.float32)

        ############################## load stereo right images ######################################################################
        self.rgbfiles_right = dataset.cam3_files

        ############################## load calibrations ######################################################################
        K = dataset.calib.K_cam2
        self.intrinsic = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        K = dataset.calib.K_cam3
        self.intrinsic_right = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])

        T_LI = dataset.calib.T_cam2_imu
        T_RI = dataset.calib.T_cam3_imu
        T_LR = np.matmul(T_LI, np.linalg.inv(T_RI))
        self.right2left_pose = pp.from_matrix(torch.tensor(T_LR), ltype=pp.SE3_type).to(dtype=torch.float32)

        self.require_undistort = False

        ############################## load gt poses ######################################################################
        T_w_imu = np.array([oxts_frame.T_w_imu for oxts_frame in dataset.oxts])
        T_w_imu = T_w_imu[self.rgb2imu_sync]
        self.poses = pp.from_matrix(torch.tensor(T_w_imu), ltype=pp.SE3_type).to(dtype=torch.float32)
        
        vels_local = torch.tensor([[oxts_frame.packet.vf, oxts_frame.packet.vl, oxts_frame.packet.vu] for oxts_frame in dataset.oxts], dtype=torch.float32)
        vels_local = vels_local[self.rgb2imu_sync]
        self.vels = self.poses.rotation() @ vels_local

        self.poses = self.poses.numpy()
        self.vels = self.vels.numpy()

        ############################## load imu data ######################################################################
        # self.accels = np.array([[oxts_frame.packet.af, oxts_frame.packet.al, oxts_frame.packet.au] for oxts_frame in dataset.oxts])
        # self.gyros = np.array([[oxts_frame.packet.wf, oxts_frame.packet.wl, oxts_frame.packet.wu] for oxts_frame in dataset.oxts])
        self.accels = np.array([[oxts_frame.packet.ax, oxts_frame.packet.ay, oxts_frame.packet.az] for oxts_frame in dataset.oxts])
        self.gyros = np.array([[oxts_frame.packet.wx, oxts_frame.packet.wy, oxts_frame.packet.wz] for oxts_frame in dataset.oxts])

        self.imu_dts = np.diff(ts_imu).astype(np.float32)

        T_IL = np.linalg.inv(T_LI)
        self.rgb2imu_pose = pp.from_matrix(torch.tensor(T_IL), ltype=pp.SE3_type).to(dtype=torch.float32)

        self.gravity = 9.81

        self.has_imu = True

    def load_timestamps(self, datapath, subfolder):
        import datetime as dt

        """Load timestamps from file."""
        timestamp_file = path.join(
            datapath, subfolder, 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t.timestamp())
        timestamps.sort()

        return timestamps


class TrajFolderDataset(Dataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1, loader=None):
        if loader is None:
            if datatype == 'tartanair':
                loader = TartanAirTrajFolderLoader(datadir)
            elif datatype == 'euroc':
                loader = EuRoCTrajFolderLoader(datadir)
            elif datatype == 'kitti':
                loader = KITTITrajFolderLoader(datadir)

        if end_frame <= 0:
            end_frame += len(loader.rgbfiles)

        self.datatype = datatype
        
        self.rgbfiles = loader.rgbfiles[start_frame:end_frame]
        self.rgb_dts = loader.rgb_dts[start_frame:end_frame-1]
        self.num_img = len(self.rgbfiles)

        try:
            self.rgbfiles_right = loader.rgbfiles_right[start_frame:end_frame]
        except:
            self.rgbfiles_right = None

        try:
            self.flowfiles = loader.flowfiles[start_frame:end_frame-1]
        except:
            self.flowfiles = None

        try:
            self.depthfiles = loader.depthfiles[start_frame:end_frame]
        except:
            self.depthfiles = None

        self.intrinsic = loader.intrinsic
        try:
            self.intrinsic_right = loader.intrinsic_right
            self.right2left_pose = loader.right2left_pose
        except:
            pass

        self.poses = loader.poses[start_frame:end_frame]

        try:
            self.vels = loader.vels[start_frame:end_frame]
        except:
            self.vels = None

        if loader.has_imu:
            self.rgb2imu_sync = loader.rgb2imu_sync[start_frame:end_frame]
            start_imu = self.rgb2imu_sync[0]
            end_imu = self.rgb2imu_sync[-1] + 1
            self.rgb2imu_sync -= start_imu

            self.accels = loader.accels[start_imu:end_imu]
            self.gyros = loader.gyros[start_imu:end_imu]
            self.imu_dts = loader.imu_dts[start_imu:end_imu-1]
            
            self.rgb2imu_pose = loader.rgb2imu_pose
            self.imu_init = {'rot':self.poses[0, 3:], 'pos':self.poses[0, :3], 'vel':self.vels[0]}
            self.gravity = loader.gravity

            self.imu_motions = None
            self.has_imu = True

        else:
            self.has_imu = False

        if loader.require_undistort:
            self.imgmap = loader.imgmap
            try:
                self.imgmap_right = loader.imgmap_right
            except:
                pass
            self.require_undistort = True
        else:
            self.require_undistort = False

        self.transform = transform

        self.links = None
        self.num_link = 0

        del loader

    def imu_pose2motion(self, imu_poses):
        SEs = pos_quats2SEs(imu_poses)
        matrix = pose2motion(SEs, links=self.links)
        self.imu_motions = SEs2ses(matrix).astype(np.float32)

    def calc_motions_by_links(self, links):
        if self.poses is None:
            return None

        SEs = pos_quats2SEs(self.poses)
        matrix = pose2motion(SEs, links=links)
        motions = SEs2ses(matrix).astype(np.float32)
        return motions

    def undistort(self, img, is_right=False):
        if not self.require_undistort:
            return img
        imgmap = self.imgmap_right if is_right else self.imgmap
        dst = cv2.remap(img, imgmap[0], imgmap[1], cv2.INTER_AREA)
        return dst

    def __len__(self):
        return self.num_link


class TrajFolderDatasetPVGO(TrajFolderDataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1, loader=None,
                    use_loop_closure=False, use_stop_constraint=False, batch_size=None):

        super(TrajFolderDatasetPVGO, self).__init__(datadir, datatype, transform, start_frame, end_frame, loader)

        ############################## generate links ######################################################################
        self.links = []
        # # [loop closure] gt pose
        # if use_loop_closure and self.poses is not None:
        #     loop_min_interval = 100
        #     trans_th = np.average([np.linalg.norm(self.poses[i+1, :3] - self.poses[i, :3]) for i in range(len(self.poses)-1)]) * 5
        #     self.links.extend(gt_pose_loop_detector(self.poses, loop_min_interval, trans_th, 5))
        # # [loop closure] bag of word (to do)
        # self.links = bow_orb_loop_detector(self.rgbfiles, loop_min_interval)
        # # [loop closure] adjancent
        # loop_range = 2
        # loop_interval = 1
        # self.links.extend(adj_loop_detector(self.num_img, loop_range, loop_interval))
        if batch_size == None:
            self.links = [[i, i+1] for i in range(self.num_img-1)]
        else:
            for current_idx in range(0, self.num_img-batch_size-1, batch_size):
                for i in range(current_idx, current_idx+batch_size):
                    self.links.append([i, i+1])
                for i in range(current_idx, current_idx+batch_size-1):
                    self.links.append([i, i+2])


        self.num_link = len(self.links)

        ############################## calc motions ######################################################################
        self.motions = self.calc_motions_by_links(self.links)

        ############################## pick stop frames ######################################################################
        self.stop_frames = []
        if use_stop_constraint and self.vels_world is not None:
            self.stop_frames = gt_vel_stop_detector(self.vels_world[::imu_mul], 0.02)

    def __getitem__(self, idx):
        res = {}

        img0 = cv2.imread(self.rgbfiles[self.links[idx][0]], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.rgbfiles[self.links[idx][1]], cv2.IMREAD_COLOR)
        img0 = self.undistort(img0)
        img1 = self.undistort(img1)
        # cv2.imwrite('temp/{}_img0.png'.format(idx), img0)
        # cv2.imwrite('temp/{}_img1.png'.format(idx), img1)
        res['img0'] = [img0]
        res['img1'] = [img1]
        res['path_img0'] = self.rgbfiles[self.links[idx][0]]
        res['path_img1'] = self.rgbfiles[self.links[idx][1]]

        if self.rgbfiles_right is not None:
            img0_r = cv2.imread(self.rgbfiles_right[self.links[idx][0]], cv2.IMREAD_COLOR)
            img0_r = self.undistort(img0_r, True)
            # cv2.imwrite('temp/{}_img0_r.png'.format(idx), img0_r)
            res['img0_r'] = [img0_r]
            res['path_img0_r'] = self.rgbfiles_right[self.links[idx][0]]
            # res['blxfx'] = np.array([self.focalx * self.baseline], dtype=np.float32) # used for convert disp to depth
            
        if self.flowfiles is not None:
            flow = np.load(self.flowfiles[self.links[idx][0]])
            res['flow'] = [flow]
            res['path_flow'] = self.flowfiles[self.links[idx][0]]

        if self.depthfiles is not None:
            depth = np.load(self.depthfiles[self.links[idx][0]])
            res['depth0'] = [depth]
            res['path_depth0'] = self.depthfiles[self.links[idx][0]]

        h, w, _ = img0.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.intrinsic[0], self.intrinsic[1], self.intrinsic[2], self.intrinsic[3])
        res['intrinsic'] = [intrinsicLayer]

        res['intrinsic_calib'] = self.intrinsic.copy()

        res['link'] = np.array(self.links[idx])

        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]

        if self.has_imu and self.imu_motions is not None:
            res['imu_motion'] = self.imu_motions[idx]

        if self.right2left_pose != None:
            res['extrinsic'] = self.right2left_pose.Log().numpy()

        res['datatype'] = self.datatype

        return res


class TrajFolderDatasetMultiCam(TrajFolderDataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1):

        super(TrajFolderDatasetMultiCam, self).__init__(datadir, datatype, transform, start_frame, end_frame)

        ############################## generate links ######################################################################
        angle_range = (0, 5)
        trans_range = (0.1, 0.5)
        
        # debug
        # angle_range = (0, 0.01)
        # trans_range = (0.0, 0.05)
        self.links = multicam_frame_selector(self.poses, trans_range, angle_range)

        self.num_link = len(self.links)

        ############################## calc extrinsics & motions ######################################################################
        self.motions = self.calc_motions_by_links(self.links[:, (0,2)])
        self.extrinsics = self.calc_motions_by_links(self.links[:, (0,1)])

    def __getitem__(self, idx):
        res = {}

        imgA = cv2.imread(self.rgbfiles[self.links[idx][0]], cv2.IMREAD_COLOR)
        imgB = cv2.imread(self.rgbfiles[self.links[idx][1]], cv2.IMREAD_COLOR)
        imgC = cv2.imread(self.rgbfiles[self.links[idx][2]], cv2.IMREAD_COLOR)
        res['img0'] = [imgA]
        res['img1'] = [imgC]
        res['img0_r'] = [imgB]

        res['path_img0'] = self.rgbfiles[self.links[idx][0]]
        res['path_img1'] = self.rgbfiles[self.links[idx][2]]
        res['path_img0_r'] = self.rgbfiles[self.links[idx][1]]

        h, w, _ = imgA.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.intrinsic[0], self.intrinsic[1], self.intrinsic[2], self.intrinsic[3])
        res['intrinsic'] = [intrinsicLayer]

        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]

        if self.has_imu and self.imu_motions is not None:
            res['imu_motion'] = self.imu_motions[idx]

        if self.extrinsics is not None:
            res['extrinsic'] = self.extrinsics[idx]
        return res


class MultiTrajFolderDataset(Dataset):
    def __init__(self, DatasetType, datatype_root, transform=None, mode='train', debug=False):
        self.datatype_root = datatype_root
        self.mode = mode

        folder_list = []
        folder_list.extend(self.list_tartanair_folders())
        folder_list.extend(self.list_kitti_folders())
        folder_list.extend(self.list_euroc_folders())
        folder_list.sort()
        if debug:
            folder_list = [folder_list[0]]

        self.datasets = []
        self.accmulatedDataSize = [0]

        print('Loading dataset for {} dirs ...'.format(len(folder_list)))

        from tqdm import tqdm
        for folder, datatype in tqdm(folder_list):
            if isinstance(DatasetType, list) or isinstance(DatasetType, tuple):
                for DS in DatasetType:
                    dataset = DS(datadir=folder, datatype=datatype, transform=transform)
                    self.datasets.append(dataset)
                    self.accmulatedDataSize.append(self.accmulatedDataSize[-1] + len(dataset))
            else:
                dataset = DatasetType(datadir=folder, datatype=datatype, transform=transform)
                self.datasets.append(dataset)
                self.accmulatedDataSize.append(self.accmulatedDataSize[-1] + len(dataset))

        print('Find {} datasets. Have {} frames in total.'.format(len(self.datasets), self.accmulatedDataSize[-1]))

    def list_tartanair_folders(self):
        if 'tartanair' not in self.datatype_root:
            return []
        else:
            dataroot = self.datatype_root['tartanair']

        scenedirs = [
            'abandonedfactory',    'abandonedfactory_night',   'amusement',        'carwelding',   'ocean',
            'gascola',             'hospital',                 'japanesealley',    'neighborhood', 'seasonsforest',
            'office',              'office2',                  'oldtown',          'seasidetown',  'seasonsforest_winter',
            'soulcity',            'westerndesert',            'endofworld'
        ]
        level_set = ['Easy', 'Hard']

        if self.mode == 'train':
            print('\nLoading Training dataset')
            scenedirs = scenedirs[0:16]
        elif self.mode == 'test':
            scenedirs = scenedirs[16:18]
            print('\nLoading Testing dataset')
                
        res = []

        for scene in scenedirs:
            for level in level_set:
                trajdirs = listdir('{}/{}/{}'.format(dataroot, scene, level))
                trajdirs.sort()
                for traj in trajdirs:
                    if not (len(traj)==4 and traj.startswith('P0')):
                        continue
                    folder = '{}/{}/{}/{}'.format(dataroot, scene, level, traj)
                    res.append([folder, 'tartanair'])
                    # only load one traj per env level!
                    break
        
        return res

    def list_kitti_folders(self):
        if 'kitti' not in self.datatype_root:
            return []
        else:
            dataroot = self.datatype_root['kitti']

        date_drive = {
            '2011_09_30': [
                '2011_09_30_drive_0016_sync',
                '2011_09_30_drive_0018_sync',
                '2011_09_30_drive_0020_sync',
                '2011_09_30_drive_0027_sync',
                '2011_09_30_drive_0028_sync',
                '2011_09_30_drive_0033_sync',
                '2011_09_30_drive_0034_sync'
            ],
            '2011_10_03': [
                '2011_10_03_drive_0027_sync',
                '2011_10_03_drive_0034_sync',
                '2011_10_03_drive_0042_sync'
            ]
        }

        if self.mode == 'train':
            print('\nLoading Training dataset')
        elif self.mode == 'test':
            print('\nLoading Testing dataset')

        res = []

        for date, drive_list in date_drive.items():
            for drive in drive_list:
                folder = '{}/{}/{}'.format(dataroot, date, drive)
                res.append([folder, 'kitti'])
        
        return res
    
    def list_euroc_folders(self):
        if 'euroc' not in self.datatype_root:
            return []
        else:
            dataroot = self.datatype_root['euroc']

        trajs = [
            'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult',
            'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
            'V2_01_easy', 'V2_02_medium', 'V2_03_difficult'
        ]

        res = []

        for traj in trajs:
            folder = '{}/{}/mav0'.format(dataroot, traj)
            res.append([folder, 'euroc'])
        
        return res

    def __len__(self):
        return self.accmulatedDataSize[-1]

    def __getitem__(self, idx):
        ds_idx = 0
        while idx >= self.accmulatedDataSize[ds_idx]:
            ds_idx += 1
        ds_idx -= 1

        return self.datasets[ds_idx][idx - self.accmulatedDataSize[ds_idx]]

    # def list_all_frames(self):
    #     all_frames = []
    #     for ds in self.datasets:
    #         rgb = np.array([fname.replace(self.dataroot, '') for fname in ds.rgbfiles])
    #         all_frames.extend(rgb[ds.links].tolist())
    #     return all_frames


class LoopDataSampler:
    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=4, distributed=True):
        self.dataset = dataset
        pin = False
        if distributed:
            self.dist_sampler = DistributedSampler(dataset, shuffle=shuffle)
            self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=self.dist_sampler, pin_memory=pin)
        else:
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
        self.dataiter = iter(self.dataloader)
        self.epoch_cnt = 0
        self.distributed = distributed

        self.first_sample = self.next()
    
    def next(self):
        try:
            sample = next(self.dataiter)

        except StopIteration:
            self.epoch_cnt += 1
            if self.distributed:
                self.dist_sampler.set_epoch(self.epoch_cnt)

            self.dataiter = iter(self.dataloader)
            sample = next(self.dataiter)

        return sample

    def first(self):
        return self.first_sample
        