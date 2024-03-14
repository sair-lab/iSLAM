import torch
import numpy as np
import pypose as pp
from scipy.spatial.transform import Rotation as R


# ----- scipy functions -----

def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def pose2motion(data, skip=0, links=None):
    if links is None:
        links = [(i, i+skip+1) for i in range(data.shape[0]-skip-1)]
    all_motion = np.zeros((len(links), 12))
    for i, l in enumerate(links):
        pose_curr = line2mat(data[l[0],:])
        pose_next = line2mat(data[l[1],:])
        motion = pose_curr.I * pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion

def SO2so(SO_data):
    return R.from_matrix(SO_data).as_rotvec()

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def SE2se(SE_data):
    result = np.zeros((6))
    result[0:3] = np.array(SE_data[0:3,3].T)
    result[3:6] = SO2so(SE_data[0:3,0:3]).T
    return result

def se2SE(se_data):
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat
    
def SEs2ses(motion_data):
    data_size = motion_data.shape[0]
    ses = np.zeros((data_size,6))
    for i in range(0,data_size):
        SE = np.matrix(np.eye(4))
        SE[0:3,:] = motion_data[i,:].reshape(3,4)
        ses[i,:] = SE2se(SE)
    return ses

def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    SE = np.array(SE[0:3,:]).reshape(1,12)
    return SE

def pos_quats2SEs(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,12))
    for i_data in range(0,data_len):
        SE = pos_quat2SE(quat_datas[i_data,:])
        SEs[i_data,:] = SE
    return SEs


# ----- pypose functions -----

def cvtSE3_pypose(motion):
    if isinstance(motion, pp.LieTensor):
        if motion.ltype == pp.SE3_type:
            return motion.clone()
        elif motion.ltype == pp.se3_type:
            return motion.Exp()
    else:
        if not isinstance(motion, torch.Tensor):
            motion = torch.tensor(motion)
        if motion.shape[-1] == 6:
            trans = motion[..., :3]
            rot = pp.so3(motion[..., 3:]).Exp().tensor()
            return pp.SE3(torch.cat([trans, rot], dim=-1))
        elif motion.shape[-1] == 7:
            return pp.SE3(motion)
    assert False, "Not valid input."

def tartan2kitti_pypose(motion):
    motion = cvtSE3_pypose(motion)
    
    T= [[0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [1.,0.,0.,0.],
        [0.,0.,0.,1.]]
    T = pp.from_matrix(T, ltype=pp.SE3_type).to(motion.device)

    return T @ motion @ T.Inv()

def motion2pose_pypose(motion, T=None):
    motion = cvtSE3_pypose(motion)

    if T is None:
        T = pp.SE3([0,0,0, 0,0,0,1]).to(motion.device)
    else:
        T = cvtSE3_pypose(T).to(motion.device)

    pose = [T]
    for m in motion:
        T = T @ m
        pose.append(T)

    pose = pp.SE3(torch.stack(pose))
    return pose

def pose2motion_pypose(pose):
    pose = cvtSE3_pypose(pose)

    motion = []
    for i in range(len(pose)-1):
        motion.append(pose[i].Inv() @ pose[i+1])

    motion = pp.SE3(torch.stack(motion))
    return motion
