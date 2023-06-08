import cv2
import numpy as np
import pypose as pp

import torch
from torch.masked import masked_tensor

from timer import Timer

from Datasets.utils import ToTensor, Compose, CropCenter, DownscaleFlow, Normalize, SqueezeBatchDim, RandomResizeCrop, RandomHSV, save_images


def is_inside_image_1D(u, width):
    return torch.logical_and(u >= 0, u <= width)

def is_inside_image(uv, width, height):
    return torch.logical_and(is_inside_image_1D(uv[0], width), is_inside_image_1D(uv[1], height))

def proj(x):
    return x / x[..., -1:]

def scale_from_disp_flow(disp, flow, motion, fx, fy, cx, cy, baseline, depth=None, mask=None, disp_th=1):
    height, width = flow.shape[-2:]
    device = motion.device

    if motion.shape[-1] == 7:
        T = pp.SE3(motion)
    else:
        T = pp.se3(motion).Exp()

    u_lin = torch.linspace(0, width-1, width)
    v_lin = torch.linspace(0, height-1, height)
    u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
    u = u.to(device=device)
    v = v.to(device=device)
    uv = torch.stack([u, v])
    uv1 = torch.stack([u, v, torch.ones_like(u)])

    flow_norm = torch.linalg.norm(flow, dim=0)
    flow_mask = torch.logical_and(is_inside_image(flow + uv, width, height), flow_norm > 0)
    # flow_mask = flow_norm > 0
    if mask is None:
        mask = flow_mask
    else:
        mask = torch.logical_and(flow_mask, mask)

    if depth is None:
        disp_mask = torch.logical_and(is_inside_image_1D(-disp + u, width), disp >= disp_th)
        # disp_mask = disp >= 5
        mask = torch.logical_and(disp_mask, mask)

        z = torch.where(disp_mask, fx*baseline / disp, 0)

    else:
        depth_th = fx*baseline
        depth_mask = torch.logical_and(depth <= depth_th, depth > 0)
        mask = torch.logical_and(depth_mask, mask)

        z = torch.where(depth_mask, depth, 0)

    # z_gray = to_image(z.numpy()*10)
    # cv2.imwrite('z_gray.png', z_gray)

    # u_gray = to_image(u.numpy()*0.5)
    # v_gray = to_image(v.numpy()*0.5)
    # cv2.imwrite('u_gray.png', u_gray)
    # cv2.imwrite('v_gray.png', v_gray)  

    if torch.sum(mask) < 1000:
        print('Warning! mask contains too less points!', torch.sum(mask)) 

    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3).to(device=device)
    K_inv = torch.linalg.inv(K)

    P = z.unsqueeze(-1) * (K_inv.unsqueeze(0).unsqueeze(0) @ uv1.permute(1, 2, 0).unsqueeze(-1)).squeeze()

    R = T.Inv().rotation()
    t = T.Inv().translation()
    # scale = torch.linalg.norm(t)
    # t_norm = t / scale
    t_norm = torch.nn.functional.normalize(t, dim=0)
    # t_norm = t  # the input trans is normalized
    a = (K @ t_norm.view(3, 1)).squeeze().unsqueeze(0).unsqueeze(0)
    b = (K.unsqueeze(0).unsqueeze(0) @ (R.unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)).squeeze()
    # print(a.shape, b.shape)

    f = (flow + uv).permute(1, 2, 0)
    M1 = a[..., 2] * f[..., 0] - a[..., 0]
    w1 = b[..., 0] - b[..., 2] * f[..., 0]
    M2 = a[..., 2] * f[..., 1] - a[..., 1]
    w2 = b[..., 1] - b[..., 2] * f[..., 1]
    # print(M.shape, w.shape)

    # print(torch.sum(mask))
    m_M1 = M1.view(-1)[mask.view(-1)]
    m_M2 = M2.view(-1)[mask.view(-1)]
    m_w1 = w1.view(-1)[mask.view(-1)]
    m_w2 = w2.view(-1)[mask.view(-1)]

    M = torch.stack([m_M1, m_M2]).view(-1, 1)
    w = torch.stack([m_w1, m_w2]).view(-1, 1)
    s = 1 / torch.sum(M * M) * M.t() @ w
    s = s.view(1)
    # s = s.item()

    # print(s, scale)

    T = pp.SE3(torch.cat([s * t, R.tensor()]))

    reproj = K.unsqueeze(0).unsqueeze(0) @ proj(T.Inv().unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)
    reproj = reproj.squeeze().permute(2, 0, 1)[:2, ...]

    # reproj_rgb = np.concatenate([reproj.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj.shape[1:]), axis=-1)], axis=-1)*0.5
    # reproj_rgb = to_image(reproj_rgb)
    # uv_rgb = np.concatenate([uv.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(uv.shape[1:]), axis=-1)], axis=-1)*0.5
    # uv_rgb = to_image(uv_rgb)
    # cv2.imwrite('reproj_rgb.png', reproj_rgb)
    # cv2.imwrite('uv_rgb.png', uv_rgb)

    r = reproj - (flow + uv)

    # flow_dest = flow + uv
    # flow_dest_rgb = np.concatenate([flow_dest.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(flow_dest.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('flow_dest_rgb.png', flow_dest_rgb)
    # reproj_flow = reproj - uv
    # reproj_flow_rgb = np.concatenate([reproj_flow.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj_flow.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('reproj_flow_rgb.png', reproj_flow_rgb)

    return r, s, mask


path = '/user/taimengf/projects/tartanair/TartanAir/abandonedfactory/Easy/P000'

def to_image(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def create_dataset():
    # transform = Compose([   CropCenter((args.image_height, args.image_width), fix_ratio=True), 
    #                         DownscaleFlow(), 
    #                         Normalize(), 
    #                         ToTensor(),
    #                         SqueezeBatchDim()
    #                     ])

    transformlist = []
    transformlist = [ CropCenter( size=(448, 640), 
                                    fix_ratio=False, scale_w=1.0, scale_disp=False)]
    transformlist.append(DownscaleFlow())
    transformlist.append(RandomHSV((10,80,80), random_random=0.2))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transformlist.append(Normalize(mean=mean, std=std, keep_old=True))
    transformlist.extend([ToTensor(), SqueezeBatchDim()])
    transform = Compose(transformlist)

    return transform


if __name__ == '__main__':
    from Datasets.transformation import tartan2kitti_pypose

    timer = Timer()

    idx = 269
    flow = np.load(path+'/flow/%06d_%06d_flow.npy'%(idx, idx+1))
    depth = np.load(path+'/depth_left/%06d_left_depth.npy'%idx)
    disp = 320*0.25 / depth

    transform = create_dataset()
    sample = {}
    sample['depth0'] = [depth]
    sample['flow'] = [flow]
    sample['disp0'] = [disp]
    transform(sample)
    depth = sample['depth0']
    flow = sample['flow']
    disp = sample['disp0']
    if flow.shape[-1] == 160:
        print('scale flow /4')
        flow /= 4
        disp /= 4

    # img1 = cv2.imread(path+'/image_left/000000_left.png')
    # img2 = cv2.imread(path+'/image_left/000001_left.png')
    # cv2.imwrite('img1.png', img1)
    # cv2.imwrite('img2.png', img2)

    # img1_warp = warp_flow(img1, flow)
    # cv2.imwrite('img1_warp.png', img1_warp)

    # flow_rgb = np.concatenate([flow*10, np.expand_dims(np.zeros(flow.shape[:-1]), axis=-1)], axis=2)
    # flow_rgb = to_image(flow_rgb)
    # depth_gray = to_image(depth*10)
    # disp_gray = to_image(disp)
    # cv2.imwrite('flow_rgb.png', flow_rgb)
    # cv2.imwrite('depth_gray.png', depth_gray)
    # cv2.imwrite('disp_gray.png', disp_gray)

    # print('disp max min', np.max(disp), np.min(disp))
    # print('depth max min', np.max(depth), np.min(depth))

    # flow = torch.from_numpy(flow).permute(2, 0, 1)
    # disp = torch.from_numpy(disp)
    # depth = torch.from_numpy(depth)
            
    print('flow', flow.shape, 'depth', depth.shape)
    print('depth', torch.min(depth), torch.max(depth), torch.mean(depth))
    print('flow', torch.min(flow), torch.max(flow), torch.mean(flow))
    print('disp', torch.min(disp), torch.max(disp), torch.mean(disp))

    poses = np.loadtxt(path+'/pose_left.txt')
    poses = pp.SE3(poses)
    # poses = tartan2kitti(poses)
    motion = poses[idx].Inv() @ poses[idx+1]
    motion = tartan2kitti_pypose(motion)
    print(motion)
    scale = torch.linalg.norm(motion.translation())

    motion[:3] = torch.nn.functional.normalize(motion[:3], dim=0)    

    # print(motion)

    timer.tic('fwd')
    fx = fy = cx = flow.shape[-1] / 2
    cy = flow.shape[-2] / 2
    print('intrinsic', fx, fy, cx, cy)
    r, s = scale_from_disp_flow(disp, flow, motion, fx, fy, cx, cy, 0.25)
    timer.toc('fwd')

    # img1_reproj = warp_flow(img1, reproj_flow.permute(1, 2, 0).numpy())
    # cv2.imwrite('img1_reproj.png', img1_reproj)

    print('r.shape', r.shape)
    print('max', torch.max(r), 'min', torch.min(r))
    print('s', s, 'scale', scale)
    print('fwd time', timer.last('fwd'))

    # r_np = r.detach().permute(1, 2, 0).numpy()
    # r_gray = np.linalg.norm(r_np, axis=2)
    # r_gray = to_image(r_gray)
    # r_rgb = np.concatenate([r_np, np.expand_dims(np.zeros(r_np.shape[:-1]), axis=-1)], axis=2)
    # r_rgb = to_image(r_rgb)
    # cv2.imwrite('r_rgb.png', r_rgb)
    # cv2.imwrite('r_gray.png', r_gray)


