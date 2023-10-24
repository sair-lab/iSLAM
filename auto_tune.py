import os

data_root = '/data/tartanair'

data_name = [
    'soulcity_Hard_P000',
    'soulcity_Hard_P001',
    'soulcity_Hard_P002',
    'soulcity_Hard_P003',
    'soulcity_Hard_P004',
    'soulcity_Hard_P005',
    'soulcity_Hard_P008',
    'soulcity_Hard_P009',
    # 'japanesealley_Hard_P000', 
    # 'japanesealley_Hard_P001', 
    # 'japanesealley_Hard_P002', 
    # 'japanesealley_Hard_P003', 
    # 'japanesealley_Hard_P004', 
    # 'japanesealley_Hard_P005', 
    'ocean_Hard_P000',
    'ocean_Hard_P001',
    'ocean_Hard_P002',
    'ocean_Hard_P003',
    'ocean_Hard_P004',
    'ocean_Hard_P005',
    'ocean_Hard_P006',
    'ocean_Hard_P007',
    'ocean_Hard_P008',
    'ocean_Hard_P009'
]

# data_name = [
#     'soulcity_Easy_P000',
#     'soulcity_Easy_P001',
#     'soulcity_Easy_P002',
#     'soulcity_Easy_P003',
#     'soulcity_Easy_P004',
#     'soulcity_Easy_P005',
#     'soulcity_Easy_P006',
#     'soulcity_Easy_P007',
#     'soulcity_Easy_P008',
#     'soulcity_Easy_P009',
#     'soulcity_Easy_P010',
#     'soulcity_Easy_P011',
#     'soulcity_Easy_P012',
#     'ocean_Easy_P000',
#     'ocean_Easy_P001',
#     'ocean_Easy_P002',
#     'ocean_Easy_P004',
#     'ocean_Easy_P005',
#     'ocean_Easy_P006',
#     'ocean_Easy_P008',
#     'ocean_Easy_P009',
#     'ocean_Easy_P010',
#     'ocean_Easy_P011',
#     'ocean_Easy_P012',
#     'ocean_Easy_P013'
# ]

# data_name = [
#     'seasidetown_Easy_P000',
#     'seasidetown_Easy_P001',
#     'seasidetown_Easy_P002',
#     'seasidetown_Easy_P003',
#     'seasidetown_Easy_P004',
#     'seasidetown_Easy_P005',
#     'seasidetown_Easy_P006',
#     'seasidetown_Easy_P007',
#     'seasidetown_Easy_P008',
#     'seasidetown_Easy_P009',
#     'seasidetown_Hard_P000',
#     'seasidetown_Hard_P001',
#     'seasidetown_Hard_P002',
#     'seasidetown_Hard_P004',
# ]

import numpy as np
from evaluate_ate_scale import calc_ate
import random
import copy

from multiprocessing import Process

def eval(dir, dn, lw_str):
    gt_poses = np.loadtxt(dir+'/pose_left.txt')
    vo_poses = np.loadtxt(f'train_results/{dn}_optmbias/exp_bs=8_lr=1e-6_lw=(0.05,10,10,3)_stereo/1/vo_pose.txt')
    pgo_poses = np.loadtxt(f'train_results/{dn}_optmbias/tune_lw={lw_str}_stereo/1/pgo_pose.txt')
    gt_poses = gt_poses[:len(vo_poses)]
    vo_ate = calc_ate(vo_poses, gt_poses)
    pgo_ate = calc_ate(pgo_poses, gt_poses)
    return pgo_ate - vo_ate

results = {}

def run(lw):
    lw_tp = tuple(lw)
    if lw_tp in results:
        return results[lw_tp]
    
    pool = []
    for i, dn in enumerate(data_name):
        dir = data_root + '/' + dn.replace('_', '/')
        res_name = dn + '_optmbias'
        lw_str = str(tuple(lw)).replace(' ', '')
        cmd = "sh run_tartanair_lw.sh {} {} {} > /dev/null".format(dir, res_name, '\''+lw_str+'\'')
        p = Process(target=os.system, args=(cmd,))
        p.start()
        pool.append(p)
        if (i+1) % 3 == 0:
            for p in pool:
                p.join()
            pool = []
    for p in pool:
        p.join()

    loss = 0
    for i, dn in enumerate(data_name):
        dir = data_root + '/' + dn.replace('_', '/')
        res_name = dn + '_optmbias'
        lw_str = str(tuple(lw)).replace(' ', '')
        loss += eval(dir, dn, lw_str)
    results[lw_tp] = loss
    
    return loss

lw = [1.5, 0.25, 3.0, 0.025]
changed=True

bestlw = copy.copy(lw)
bestloss = run(lw)
print('init loss', bestloss)

while changed:
    order = [0,1,2,3]
    random.shuffle(order)
    print('order', order)
    changed = False
    for idx in order:
        print('\tidx', idx)
        old = lw[idx]
        for rate in [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]:
            lw[idx] = old * rate
            loss = run(lw)
            if loss < bestloss:
                bestloss = loss
                bestlw = copy.copy(lw)
                changed=True
                print('>>> New Best <<<')
            print('\t\trun', lw, loss)
        lw = copy.copy(bestlw)

print('best', bestlw, bestloss)
