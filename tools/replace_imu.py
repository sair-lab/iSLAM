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
    '2011_10_03_drive_0042'
]

data_root = '.'

for dname in data_list:
    ddate = dname[:len('2011_09_30')]
    sync_root = data_root + '/' + ddate + '/' + dname+'_sync'
    extract_root = data_root + '/' + ddate + '/' + dname+'_extract'

    if isdir('{}/oxts_sync'.format(sync_root)):
        print('Has a oxts_sync folder at {}!, Skip ...'.format(sync_root))
    else:
        print('Replacing {} ...'.format(dname))
        system('mv {}/oxts {}/oxts_sync'.format(sync_root, sync_root))
        system('cp -r {}/oxts {}/oxts'.format(extract_root, sync_root))
        print('Done.')