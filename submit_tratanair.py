import os

data_root = '/user/taimengf/projects/cwx/tartanair/TartanAir/'

data_name = [
    'soulcity_Hard_P000',
    'soulcity_Hard_P001',
    'soulcity_Hard_P002',
    'soulcity_Hard_P003',
    'soulcity_Hard_P004',
    'soulcity_Hard_P005',
    'soulcity_Hard_P008',
    'soulcity_Hard_P009',
    'japanesealley_Hard_P000', 
    'japanesealley_Hard_P001', 
    'japanesealley_Hard_P002', 
    'japanesealley_Hard_P003', 
    'japanesealley_Hard_P004', 
    'japanesealley_Hard_P005', 
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

for dn in data_name:
    dir = data_root + dn.replace('_', '/')
    cmd = "sbatch run_tartanair.sh {} {}".format(dir, dn)

    print('\n>>>>>', cmd, '<<<<<\n')

    os.system(cmd)