import torch
import pypose as pp

from Network.IMUDenoiseNet import IMUCorrector_CNN_GRU


def prase_init(self, init=None, motion_mode=False):
        dtype = self.accels.dtype

        if init is not None:
            if motion_mode:
                init_pos = torch.zeros(3, dtype=dtype).to(self.device)
                init_rot = pp.SO3(init['rot']).to(dtype).to(self.device)
                init_vel = torch.zeros(3, dtype=dtype).to(self.device)
            else:
                init_pos = torch.tensor(init['pos'], dtype=dtype).to(self.device)
                init_rot = pp.SO3(init['rot']).to(dtype).to(self.device)
                init_vel = torch.tensor(init['vel'], dtype=dtype).to(self.device)
        else:
            init_pos = torch.zeros(3, dtype=dtype).to(self.device)
            init_rot = pp.identity_SO3().to(dtype).to(self.device)
            init_vel = torch.zeros(3, dtype=dtype).to(self.device)

        return init_pos, init_rot, init_vel


class IMUModule:
    def __init__(self, accels, gyros, dts, init=None, gravity=9.81007, rgb2imu_sync=None, 
                 device='cuda:0', denoise_model_name=None, denoise_accel=True, denoise_gyro=True):
        
        self.device = device
        self.last_frame_dt = 0.1

        if rgb2imu_sync is None:
            self.rgb2imu_sync = [i for i in range(len(accels))]
        else:
            self.rgb2imu_sync = rgb2imu_sync

        dtype = torch.get_default_dtype()
        self.accels = torch.tensor(accels, dtype=dtype).to(device)
        self.gyros = torch.tensor(gyros, dtype=dtype).to(device)
        self.dts = torch.tensor(dts, dtype=dtype).unsqueeze(-1).to(device)

        self.use_denoise_model = denoise_model_name is not None and denoise_model_name != '' and (denoise_accel or denoise_gyro)
        self.optm_bias = not use_denoise_model and (denoise_accel or denoise_gyro)

        init_pos, init_rot, init_vel = prase_init(init)
        self.integrator = pp.module.IMUPreintegrator(
            init_pos, init_rot, init_vel, gravity=float(gravity)).to(device)

        if self.optm_bias:
            self.accel_bias = torch.zeros(3)
            self.gyro_bias = torch.zeros(3)

        if self.use_denoise_model:
            self.denoiser = IMUCorrector_CNN_GRU()
            pretrain = torch.load(denoise_model_name)
            self.denoiser.load_state_dict(pretrain)
            self.denoiser = self.denoiser.to(device)

            self.denoise_accel = denoise_accel
            self.denoise_gyro = denoise_gyro


    def integrate(self, st, end, init=None, motion_mode=False):
        '''
        motion_mode False: 
            pos, rot, vel in world frame
        motion_mode True : 
            rot = relative rotation from t to t+1 in t's frame
            vel = delta velocity from t to t+1 in wolrd frame
            pos = relative translation cased only by acceleration (assume zero initial speed) in world frame

        rgb2imu_sync[rgb_frame_idx] = imu_frame_idx at the same time
        '''
        
        init_pos, init_rot, init_vel = prase_init(init, motion_mode)

        if motion_mode: 
            poses, rots, covs, vels = [], [], [], []
        else:
            poses = [init_pos.cpu()]
            rots = [init_rot.rotation().cpu()]
            covs = []
            vels = [init_vel.cpu()]

        state = {'pos':init_pos.unsqueeze(0), 'rot':init_rot.unsqueeze(0), 'vel':init_vel.unsqueeze(0)}
        last_state = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}

        imu_batch_st = self.rgb2imu_sync[st]
        imu_batch_end = self.rgb2imu_sync[end] + 1

        dts = self.dts[imu_batch_st:imu_batch_end].clone()
        gyros = self.gyros[imu_batch_st:imu_batch_end].clone()
        accels = self.accels[imu_batch_st:imu_batch_end].clone()

        if self.optm_bias:
            if self.denoise_accel:
                accels -= self.accel_bias.view(1, 3)
            if self.denoise_gyro:
                gyros -= self.gyros_bias.view(1, 3)

        if self.use_denoise_model and imu_batch_end - imu_batch_st >= 10:
            data = {'acc':accels, 'gyro':gyros}
            denoised_accels, denoised_gyros, acc_cov, gyro_cov = self.denoiser(data, eval=True)
            if self.denoise_accel:
                accels = denoised_accels
            if self.denoise_gyro:
                gyros = denoised_gyros

        # has_imu = torch.ones(end-st, dtype=bool)
        for i in range(st, end):
            imu_frame_st = self.rgb2imu_sync[i] - imu_batch_st
            imu_frame_end = self.rgb2imu_sync[i+1] - imu_batch_st
            
            # if imu_frame_st == imu_frame_end:
            #     has_imu[i-st] = False
            #     dtype = accels.dtype
            #     dt = torch.ones((1, 1), dtype=dtype).to(self.device) * self.last_frame_dt
            #     gyro = torch.zeros((1, 3), dtype=dtype).to(self.device)
            #     acc = torch.zeros((1, 3), dtype=dtype).to(self.device)
            # else:
            #     dt = dts[imu_frame_st:imu_frame_end]
            #     self.last_frame_dt = torch.sum(dt)
            #     gyro = gyros[imu_frame_st:imu_frame_end]
            #     acc = accels[imu_frame_st:imu_frame_end]
            
            # state = self.integrator(dt=dt, gyro=gyro, acc=acc, init_state=last_state)

            if imu_frame_st == imu_frame_end:
                dtype = accels.dtype
                if motion_mode:
                    state['pos'] = torch.zeros((1, 3), dtype=dtype).to(self.device)
                    state['vel'] = torch.zeros((1, 3), dtype=dtype).to(self.device)
                else:
                    state['vel'] = torch.zeros((1, 3), dtype=dtype).to(self.device)
            else:
                dt = dts[imu_frame_st:imu_frame_end]
                gyro = gyros[imu_frame_st:imu_frame_end]
                acc = accels[imu_frame_st:imu_frame_end]
                state = self.integrator(dt=dt, gyro=gyro, acc=acc, init_state=last_state)

            poses.append(state['pos'][..., -1, :].squeeze().cpu())
            vels.append(state['vel'][..., -1, :].squeeze().cpu())
            if motion_mode:
                rots.append(last_state['rot'].Inv().cpu() @ state['rot'][..., -1, :].squeeze().cpu())
            else:
                rots.append(state['rot'][..., -1, :].squeeze().cpu())
            
            last_state['rot'] = state['rot'][..., -1, :].squeeze()
            if not motion_mode:
                last_state['pos'] = state['pos'][..., -1, :].squeeze()
                last_state['vel'] = state['vel'][..., -1, :].squeeze()
                
        poses = torch.stack(poses, axis=0)
        rots = torch.stack(rots, axis=0)
        vels = torch.stack(vels, axis=0)

        return poses, rots, covs, vels


class IMUFwd(nn.Module):
    def __init__(self, accels, gyros, accel_bias, gyro_bias, dts, init, gravity, device):
        dtype = torch.get_default_dtype()
        self.accels = torch.tensor(accels, dtype=dtype).to(device)
        self.gyros = torch.tensor(gyros, dtype=dtype).to(device)
        self.dts = torch.tensor(dts, dtype=dtype).unsqueeze(-1).to(device)

        self.accel_bias = nn.Parameter(accel_bias).to(device)
        self.gyro_bias = nn.Parameter(gyro_bias).to(device)

        init_pos, init_rot, init_vel = prase_init(init)
        self.integrator = pp.module.IMUPreintegrator(
            init_pos, init_rot, init_vel, gravity=float(gravity)).to(device)

    def forward(self, poses, sync):
        dts = self.dts
        accels = self.accels - self.accel_bias.view(1, 3)
        gyros = self.gyros - self.gyro_bias(1, 3)

        state = self.integrator(dt=dts, gyro=gyros, acc=accels)
        print(state['rot'][..., sync, :].squeeze().shape)

        roterr = (poses.rotation().Inv() @ state['rot'][..., sync, :].squeeze()).Log().norm()
        transerr = torch.nn.functional.mse_loss((poses.translation(), state['pos'][..., sync, :].squeeze()))

        return roterr + transerr
                

def optm_bias(lr, epoch, poses, sync, accels, gyros, accel_bias, gyro_bias, dts, init, gravity, device='cuda:0'):
    imu = IMUFwd(accels, gyros, accel_bias, gyro_bias, dts, init, gravity, device)
    optimizer = torch.optim.Adam(imu.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)

    for epoch_i in range(epoch):
        loss = imu(pose, sync)
        scheduler.step(loss)
        paint('IMU loss:', loss.itme())

    return imu.accel_bias.detach(), imu.gyro_bias.detach()


if __name__ == '__main__':
    from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim
    from Datasets.TrajFolderDataset import TrajFolderDataset

    # data_root = '/projects/academic/cwx/kitti_raw/2011_09_30/2011_09_30_drive_0033_sync'
    # data_root = '/home/data2/kitti_raw/2011_09_30/2011_09_30_drive_0033_sync'
    data_root = '/home/data2/euroc_raw/MH_01_easy/mav0'
    data_type = 'euroc'
    start_frame = 0
    end_frame = -1
    trainroot = './train_results/test_imudenoise'
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

    imu_module = IMUModule(dataset.accels, dataset.gyros, dataset.imu_dts, dataset.imu_init, 
                           dataset.gravity, dataset.rgb2imu_sync, use_denoiser=True)
    
    imu_module.integrate(0, 1, dataset.imu_init, motion_mode=False)
