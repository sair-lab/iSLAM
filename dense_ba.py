import cv2
import torch
import numpy as np
import pypose as pp
from pypose.function.geometry import pixel2point, reprojerr


def is_inside_image_1D(u, width):
    return torch.logical_and(u >= 0, u <= width)

def is_inside_image(uv, width, height):
    if len(uv.shape) == 3:
        return torch.logical_and(is_inside_image_1D(uv[0, ...], width), is_inside_image_1D(uv[1, ...], height))
    else:
        return torch.logical_and(is_inside_image_1D(uv[:, 0, ...], width), is_inside_image_1D(uv[:, 1, ...], height)) 

def proj(x, return_mask=False):
    if return_mask:
        mask = x[..., -1:] > 0.1
        proj = torch.where(mask, x / x[..., -1:], 0)
        mask = torch.where(mask, torch.logical_and(
            torch.logical_and(proj[..., 0:1]>=-1, proj[..., 0:1]<=1),
            torch.logical_and(proj[..., 1:2]>=-1, proj[..., 1:2]<=1)
        ), False)
        proj = torch.where(mask, proj, 0)
        return proj, mask.squeeze()
    else:
        return x / x[..., -1]


def scale_from_disp_flow(disp, flow, motion, fx, fy, cx, cy, baseline, depth=None, mask=None, disp_th=1):
    height, width = flow.shape[-2:]
    device = motion.device

    if motion.shape[-1] == 7:
        T = pp.SE3(motion)
    else:
        T = pp.se3(motion).Exp()

    # build UV map
    u_lin = torch.linspace(0, width-1, width)
    v_lin = torch.linspace(0, height-1, height)
    u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
    u = u.to(device)
    v = v.to(device)
    uv = torch.stack([u, v])
    uv1 = torch.stack([u, v, torch.ones_like(u)])

    # flow mask: inside after warp and magnitude > 0
    flow_norm = torch.linalg.norm(flow, dim=0)
    flow_mask = torch.logical_and(is_inside_image(flow + uv, width, height), flow_norm > 0)
    if mask is None:
        mask = flow_mask
    else:
        mask = torch.logical_and(flow_mask, mask)

    if depth is None:
        # use disparity map for depth
        # disp mask: inside after warp and magnitude > disp_th
        disp_mask = torch.logical_and(is_inside_image_1D(-disp + u, width), disp >= disp_th)
        mask = torch.logical_and(disp_mask, mask)

        # disp to depth
        z = torch.where(disp_mask, fx*baseline / disp, 0)
        depth_mask = disp_mask

    else:
        # use depth input
        depth_th = fx * baseline
        depth_mask = torch.logical_and(depth <= depth_th, depth > 0)
        mask = torch.logical_and(depth_mask, mask)

        z = torch.where(depth_mask, depth, 0) 

    if torch.sum(mask) < 500:
        print('Warning! mask contains too less points!', torch.sum(mask)) 

    # intrinsic matrix
    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3).to(device)
    K_inv = torch.linalg.inv(K)

    # back-project to 3D point
    P = z.unsqueeze(-1) * (K_inv.unsqueeze(0).unsqueeze(0) @ uv1.permute(1, 2, 0).unsqueeze(-1)).squeeze()

    R = T.Inv().rotation()
    t = T.Inv().translation()
    t_norm = torch.nn.functional.normalize(t, dim=0)

    # build linear system: Ms = w
    a = (K @ t_norm.view(3, 1)).squeeze().unsqueeze(0).unsqueeze(0)
    b = (K.unsqueeze(0).unsqueeze(0) @ (R.unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)).squeeze()
    f = (flow + uv).permute(1, 2, 0)

    M1 = a[..., 2] * f[..., 0] - a[..., 0]
    w1 = b[..., 0] - b[..., 2] * f[..., 0]
    M2 = a[..., 2] * f[..., 1] - a[..., 1]
    w2 = b[..., 1] - b[..., 2] * f[..., 1]

    # apply mask
    m_M1 = M1.view(-1)[mask.view(-1)]
    m_M2 = M2.view(-1)[mask.view(-1)]
    m_w1 = w1.view(-1)[mask.view(-1)]
    m_w2 = w2.view(-1)[mask.view(-1)]

    M = torch.stack([m_M1, m_M2]).view(-1, 1)
    w = torch.stack([m_w1, m_w2]).view(-1, 1)

    # least square solution
    s = 1 / torch.sum(M * M) * M.t() @ w
    s = s.view(1)

    T = pp.SE3(torch.cat([s * t, R.tensor()]))

    # perform reprojection and calculate residual
    # reproj = K.unsqueeze(0).unsqueeze(0) @ proj(T.Inv().unsqueeze(0).unsqueeze(0) @ P).unsqueeze(-1)
    # reproj = reproj.squeeze().permute(2, 0, 1)[:2, ...]
    # r = reproj - (flow + uv)

    # debug

    # z_gray = to_image(z.numpy()*10)
    # cv2.imwrite('z_gray.png', z_gray)

    # reproj_rgb = np.concatenate([reproj.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj.shape[1:]), axis=-1)], axis=-1)*0.5
    # reproj_rgb = to_image(reproj_rgb)
    # uv_rgb = np.concatenate([uv.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(uv.shape[1:]), axis=-1)], axis=-1)*0.5
    # uv_rgb = to_image(uv_rgb)
    # cv2.imwrite('reproj_rgb.png', reproj_rgb)
    # cv2.imwrite('uv_rgb.png', uv_rgb)

    # flow_dest = flow + uv
    # flow_dest_rgb = np.concatenate([flow_dest.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(flow_dest.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('flow_dest_rgb.png', flow_dest_rgb)
    # reproj_flow = reproj - uv
    # reproj_flow_rgb = np.concatenate([reproj_flow.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj_flow.shape[1:]), axis=-1)], axis=-1)*0.5
    # cv2.imwrite('reproj_flow_rgb.png', reproj_flow_rgb)

    # print(z.shape, z.device, mask.shape, mask.device)
    # print(torch.min(z[mask]), torch.max(z[mask]))

    return s, z.squeeze().cpu(), mask.squeeze().cpu(), depth_mask.squeeze().cpu()


class ReprojectionLoss:
    def __init__(self, depth, flow, fx, fy, cx, cy, mask, rgb2imu_pose, device='cuda:0'):
        # (batch, channel, height, width)
        assert len(flow.shape) == 4
        # (batch, height, width)
        assert len(depth.shape) == 3
        assert mask is None or len(mask.shape) == 3

        height, width = flow.shape[-2:]
        batch_size = depth.shape[0]

        self.z = depth.to(device)
        self.flow = flow.to(device)
        self.mask = mask.to(device)
        self.rgb2imu_pose = rgb2imu_pose.to(device)

        # build UV map
        u_lin = torch.linspace(0, width-1, width)
        v_lin = torch.linspace(0, height-1, height)
        u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
        u = u.to(device)
        v = v.to(device)
        self.uv = torch.stack([u, v]).repeat(batch_size, 1, 1, 1)
        self.uv1 = torch.stack([u, v, torch.ones_like(u)]).repeat(batch_size, 1, 1, 1)

        # intrinsic matrix
        self.K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3).to(device)
        self.K_inv = torch.linalg.inv(self.K)


    def __call__(self, motion):
        T = self.rgb2imu_pose.Inv() @ motion @ self.rgb2imu_pose
        z = self.z
        K = self.K
        uv1 = self.uv1
        mask = self.mask
        K_inv = self.K_inv

        # back-project to 3D point
        P = z.unsqueeze(-1) * (K_inv.view(1, 1, 1, 3, 3) @ uv1.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
        # (): (b, x, y, c, 1)
        # P: (b, x, y, c)

        # perform reprojection and calculate residual
        P = T.Inv().lview(-1, 1, 1) @ P
        p, reproj_mask = proj(P, return_mask=True)
        mask = torch.logical_and(mask, reproj_mask)
        reproj = K.view(1, 1, 1, 3, 3) @ p.unsqueeze(-1)
        reproj = reproj.squeeze(-1).permute(0, 3, 1, 2)[:, :2, ...]
        # reproj: (b, c, x, y)
        r = reproj - (self.flow + self.uv)

        bs = r.shape[0]
        l1loss = torch.sum(torch.abs(r), dim=1)
        # l1loss: (b, x, y)
        # mask = torch.logical_and(mask, l1loss <= 10)
        loss = []
        for i in range(bs):
            loss.append(torch.mean((l1loss[i])[mask[i]]))
        loss = torch.stack(loss)
        # loss: (b,)

        # debug

        def to_image(x):
            return np.clip(np.abs(x), 0, 255).astype(np.uint8)
        
        for i in range(bs):
            z_gray = to_image(z[i].cpu().numpy()*10)
            cv2.imwrite(f'temp/{i}_z_gray.png', z_gray)
            mask_gray = to_image(self.mask[i].cpu().numpy()*200)
            cv2.imwrite(f'temp/{i}_mask_gray.png', mask_gray)
            reproj_rgb = np.concatenate([reproj[i].detach().cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj.shape[2:]), axis=-1)], axis=-1)
            reproj_rgb = to_image(reproj_rgb)
            uv_rgb = np.concatenate([self.uv[i].cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(self.uv.shape[2:]), axis=-1)], axis=-1)
            uv_rgb = to_image(uv_rgb)
            cv2.imwrite(f'temp/{i}_reproj_rgb.png', reproj_rgb)
            cv2.imwrite(f'temp/{i}_uv_rgb.png', uv_rgb)
            flow_rgb = np.concatenate([self.flow[i].cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(self.flow.shape[2:]), axis=-1)], axis=-1)*10
            flow_rgb = to_image(flow_rgb)
            cv2.imwrite(f'temp/{i}_flow.png', flow_rgb)
            flow_dest = self.flow[i].cpu() + self.uv[i].cpu()
            flow_dest_rgb = np.concatenate([flow_dest.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(flow_dest.shape[1:]), axis=-1)], axis=-1)
            flow_dest_rgb = to_image(flow_dest_rgb)
            cv2.imwrite(f'temp/{i}_flow_dest_rgb.png', flow_dest_rgb)
            reproj_flow = reproj[i].detach().cpu() - self.uv[i].cpu()
            reproj_flow_rgb = np.concatenate([reproj_flow.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj_flow.shape[1:]), axis=-1)], axis=-1)*10
            reproj_flow_rgb = to_image(reproj_flow_rgb)
            cv2.imwrite(f'temp/{i}_reproj_flow_rgb.png', reproj_flow_rgb)
            l1loss_gray = to_image(l1loss[i].detach().cpu().numpy()*10)
            cv2.imwrite(f'temp/{i}_l1loss_gray.png', l1loss_gray)

        quit()

    def __call__(self, motion):
        T = self.rgb2imu_pose.Inv() @ motion @ self.rgb2imu_pose
        z = self.z
        K = self.K
        uv1 = self.uv1
        mask = self.mask
        K_inv = self.K_inv

        # back-project to 3D point
        P = z.unsqueeze(-1) * (K_inv.view(1, 1, 1, 3, 3) @ uv1.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1)
        # (): (b, x, y, c, 1)
        # P: (b, x, y, c)

        # perform reprojection and calculate residual
        P = T.Inv().lview(-1, 1, 1) @ P
        p, reproj_mask = proj(P, return_mask=True)
        mask = torch.logical_and(mask, reproj_mask)
        reproj = K.view(1, 1, 1, 3, 3) @ p.unsqueeze(-1)
        reproj = reproj.squeeze(-1).permute(0, 3, 1, 2)[:, :2, ...]
        # reproj: (b, c, x, y)
        r = reproj - (self.flow + self.uv)

        bs = r.shape[0]
        l1loss = torch.sum(torch.abs(r), dim=1)
        # l1loss: (b, x, y)
        # mask = torch.logical_and(mask, l1loss <= 10)
        loss = []
        for i in range(bs):
            loss.append(torch.mean((l1loss[i])[mask[i]]))
        loss = torch.stack(loss)
        # loss: (b,)

        # debug

        def to_image(x):
            return np.clip(np.abs(x), 0, 255).astype(np.uint8)
        
        for i in range(bs):
            z_gray = to_image(z[i].cpu().numpy()*10)
            cv2.imwrite(f'temp/{i}_z_gray.png', z_gray)
            mask_gray = to_image(self.mask[i].cpu().numpy()*200)
            cv2.imwrite(f'temp/{i}_mask_gray.png', mask_gray)
            reproj_rgb = np.concatenate([reproj[i].detach().cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj.shape[2:]), axis=-1)], axis=-1)
            reproj_rgb = to_image(reproj_rgb)
            uv_rgb = np.concatenate([self.uv[i].cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(self.uv.shape[2:]), axis=-1)], axis=-1)
            uv_rgb = to_image(uv_rgb)
            cv2.imwrite(f'temp/{i}_reproj_rgb.png', reproj_rgb)
            cv2.imwrite(f'temp/{i}_uv_rgb.png', uv_rgb)
            flow_rgb = np.concatenate([self.flow[i].cpu().permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(self.flow.shape[2:]), axis=-1)], axis=-1)*10
            flow_rgb = to_image(flow_rgb)
            cv2.imwrite(f'temp/{i}_flow.png', flow_rgb)
            flow_dest = self.flow[i].cpu() + self.uv[i].cpu()
            flow_dest_rgb = np.concatenate([flow_dest.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(flow_dest.shape[1:]), axis=-1)], axis=-1)
            flow_dest_rgb = to_image(flow_dest_rgb)
            cv2.imwrite(f'temp/{i}_flow_dest_rgb.png', flow_dest_rgb)
            reproj_flow = reproj[i].detach().cpu() - self.uv[i].cpu()
            reproj_flow_rgb = np.concatenate([reproj_flow.permute(1, 2, 0).numpy(), np.expand_dims(np.zeros(reproj_flow.shape[1:]), axis=-1)], axis=-1)*10
            reproj_flow_rgb = to_image(reproj_flow_rgb)
            cv2.imwrite(f'temp/{i}_reproj_flow_rgb.png', reproj_flow_rgb)
            l1loss_gray = to_image(l1loss[i].detach().cpu().numpy()*10)
            cv2.imwrite(f'temp/{i}_l1loss_gray.png', l1loss_gray)

        quit()

        return loss
