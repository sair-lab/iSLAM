import torch
import numpy as np
import pypose as pp


def run_imu_preintegrator(accels, gyros, dts, init=None, gravity=9.81007, 
                          device='cuda:0', motion_mode=False, rgb2imu_sync=None):

    '''
    motion_mode False: 
        pos, rot, vel in world frame
    motion_mode True : 
        rot = rotation in world frame
        vel = delta velocity in wolrd frame
        pos = relative translation cased only by acceleration (assume zero initial speed) in world frame

    rgb2imu_sync[rgb_frame_idx] = imu_frame_idx at the same time
    '''

    dtype = torch.get_default_dtype()
    
    if init is not None:
        if motion_mode:
            init_pos = torch.zeros(3, dtype=dtype).to(device)
            init_rot = pp.SO3(init['rot']).to(dtype).to(device)
            init_vel = torch.zeros(3, dtype=dtype).to(device)
        else:
            init_pos = torch.tensor(init['pos'], dtype=dtype).to(device)
            init_rot = pp.SO3(init['rot']).to(dtype).to(device)
            init_vel = torch.tensor(init['vel'], dtype=dtype).to(device)
    else:
        init_pos = torch.zeros(3, dtype=dtype).to(device)
        init_rot = pp.identity_SO3().to(dtype).to(device)
        init_vel = torch.zeros(3, dtype=dtype).to(device)
    
    if motion_mode:
        last_state = {'pos':init_pos, 'rot':init_rot, 'vel':init_vel}

    integrator = pp.module.IMUPreintegrator(init_pos, init_rot, init_vel, gravity=float(gravity)).to(device)

    accels = torch.tensor(accels, dtype=dtype).to(device)
    gyros = torch.tensor(gyros, dtype=dtype).to(device)
    dts = torch.tensor(dts, dtype=dtype).unsqueeze(-1).to(device)

    if rgb2imu_sync is None:
        N = accels.shape[0] - 1
    else:
        N = len(rgb2imu_sync) - 1

    if motion_mode:
        poses, rots, covs, vels = [], [], [], []
    else:
        poses = [init_pos.cpu().numpy()]
        rots = [init_rot.rotation().cpu().numpy()]
        covs = []
        vels = [init_vel.cpu().numpy()]

    state = {'pos':init_pos.unsqueeze(0), 'rot':init_rot.unsqueeze(0), 'vel':init_vel.unsqueeze(0)}

    for i in range(N):
        if rgb2imu_sync is None:
            if motion_mode:
                state = integrator(dt=dts[i].reshape(1, -1), gyro=gyros[i].reshape(1, -1), acc=accels[i].reshape(1, -1), init_state=last_state)
            else:
                state = integrator(dt=dts[i].reshape(1, -1), gyro=gyros[i].reshape(1, -1), acc=accels[i].reshape(1, -1))
        else:
            st = rgb2imu_sync[i]
            end = rgb2imu_sync[i+1]
            if st == end:
                if motion_mode:
                    state['pos'] = torch.zeros((1, 3), dtype=dtype).to(device)
                    state['vel'] = torch.zeros((1, 3), dtype=dtype).to(device)
                else:
                    state['vel'] = torch.zeros((1, 3), dtype=dtype).to(device)
            elif st+1 == end:
                if motion_mode:
                    state = integrator(dt=dts[st].reshape(1, -1), gyro=gyros[st].reshape(1, -1), acc=accels[st].reshape(1, -1), init_state=last_state)
                else:
                    state = integrator(dt=dts[st].reshape(1, -1), gyro=gyros[st].reshape(1, -1), acc=accels[st].reshape(1, -1))
            else:
                if motion_mode:
                    state = integrator(dt=dts[st:end], gyro=gyros[st:end], acc=accels[st:end], init_state=last_state)
                else:
                    state = integrator(dt=dts[st:end], gyro=gyros[st:end], acc=accels[st:end])

        poses.append(state['pos'][..., -1, :].squeeze().cpu().numpy())
        if motion_mode:
            rot = last_state['rot'].Inv() @ state['rot'][..., -1, :].squeeze()
            rots.append(rot.cpu().numpy())
        else:
            rots.append(state['rot'][..., -1, :].squeeze().cpu().numpy())
        vels.append(state['vel'][..., -1, :].squeeze().cpu().numpy())

        if motion_mode:
            last_state['rot'] = state['rot'][..., -1, :].squeeze()

    poses = np.stack(poses, axis=0)
    rots = np.stack(rots, axis=0)
    vels = np.stack(vels, axis=0)

    return poses, rots, covs, vels
