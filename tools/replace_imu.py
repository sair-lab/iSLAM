from os import system
from os.path import isdir


data_list = [
    '2011_09_30_drive_0016',
    '2011_09_30_drive_0018',
    '2011_09_30_drive_0020',
    '2011_09_30_drive_0027',
    '2011_09_30_drive_0028',
    '2011_09_30_drive_0033',
    '2011_09_30_drive_0034',
    '2011_10_03_drive_0027',
    '2011_10_03_drive_0034',
    '2011_10_03_drive_0042',
    # '2011_09_26_drive_0001',
    # '2011_09_26_drive_0002',
    # '2011_09_26_drive_0005',
    # '2011_09_26_drive_0009',
    # '2011_09_26_drive_0013',
    # '2011_09_26_drive_0014',
    # '2011_09_26_drive_0015',
    # '2011_09_26_drive_0017',
    # '2011_09_26_drive_0018',
    # '2011_09_26_drive_0019',
]

data_root_sync = '/data/kitti'
data_root_extract = '/data/kitti_extract'

for dname in data_list:
    ddate = dname[:len('2011_09_30')]
    sync_root = data_root_sync + '/' + ddate + '/' + dname+'_sync'
    extract_root = data_root_extract + '/' + ddate + '/' + dname+'_extract'

    if isdir('{}/oxts_sync'.format(sync_root)):
        print('Has a oxts_sync folder at {}!, Skip ...'.format(sync_root))
    else:
        print('Replacing {} ...'.format(dname))
        system('mv {}/oxts {}/oxts_sync'.format(sync_root, sync_root))
        system('cp -r {}/oxts {}/oxts'.format(extract_root, sync_root))
        print('Done.')