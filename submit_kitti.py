import os

machine = 'labserver'

if machine == 'ccr':
    data_root = '/user/taimengf/projects/cwx/kitti_raw/'
elif machine == 'labserver'
    # 10Hz IMU !!!
    data_root = '/home/data2/kitti_raw'

data_name = [
    '2011_10_03_drive_0027',
    '2011_10_03_drive_0042',
    '2011_10_03_drive_0034',
    '2011_09_30_drive_0016',
    '2011_09_30_drive_0018',
    '2011_09_30_drive_0020',
    '2011_09_30_drive_0027',
    '2011_09_30_drive_0028',
    '2011_09_30_drive_0033',
    '2011_09_30_drive_0034'
]

for dn in data_name:
    date = dn[:10]
    dir = data_root + date + '/' + dn + '_sync'
    res_name = dn + '_loop1-5'

    if machine == 'ccr':
        cmd = "sbatch run_kitti.sh {} {}".format(dir, res_name)
    else:
        cmd = "sh run_kitti.sh {} {}".format(dir, res_name)

    print('\n>>>>>', cmd, '<<<<<\n')

    os.system(cmd)
