import cv2
import torch
import numpy as np
import pypose as pp


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

    # build UV map
    u_lin = torch.linspace(0, width-1, width)
    v_lin = torch.linspace(0, height-1, height)
    u, v = torch.meshgrid(u_lin, v_lin, indexing='xy')
    u = u.to(device=device)
    v = v.to(device=device)
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
        depth_mask = torch.logical_and(is_inside_image_1D(-disp + u, width), disp >= 1)

    else:
        # use depth input
        depth_th = fx * baseline
        depth_mask = torch.logical_and(depth <= depth_th, depth > 0)
        mask = torch.logical_and(depth_mask, mask)

        z = torch.where(depth_mask, depth, 0) 

    if torch.sum(mask) < 1000:
        print('Warning! mask contains too less points!', torch.sum(mask)) 

    # intrinsic matrix
    K = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=torch.float32).view(3, 3).to(device=device)
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

    return s, z.squeeze().cpu(), depth_mask.squeeze().cpu()
